import argparse
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from dockgame.utils.diffusion import t_to_sigma, get_t_schedule
from dockgame.data.transforms import construct_score_transform, construct_reward_transform
from dockgame.game.agents import get_agent_cls, Agent, ActionDict
from scipy.spatial.transform import Rotation
from dockgame.utils.geometry import axis_angle_to_matrix, matrix_to_axis_angle

Tensor = torch.Tensor
AgentDict = dict[str, Agent]
Metrics = dict[str, float]

# Orthonormal basis of SO(3) with shape [3, 3, 3]
basis = torch.tensor([
[[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]],
[[0.,0.,1.],[0.,0.,0.],[-1.,0.,0.]],
[[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]]])
# hat map from vector space Rˆ3 to Lie algebra so(3)
def hat(v): return torch.einsum("...i,ijk->...jk", v, basis.to(v.device))
# Exponential map from so(3) to SO(3), this is the matrix exponential
def exp(A): return torch.linalg.matrix_exp(A)
# Exponential map from tangent space at R0 to SO(3)
def expmap(R0, tangent):
    skew_sym = torch.einsum("...ij,...ik->...jk", R0, tangent)
    return torch.einsum("...ij,...jk->...ik", R0, exp(skew_sym))


class BaseStrategy:

    def __init__(
        self, n_rounds: int, 
        model: torch.nn.Module,
        transform: BaseTransform,
        device: str = 'cpu',
        **kwargs
    ):
        self.model = model
        self.transform = transform
        self.n_rounds = n_rounds
        self.device = device
        self.model.eval()

    def setup_game(self):
        pass
    
    def compute_actions(self, agent_dict, agent_keys, player_keys):
        raise NotImplementedError("Subclasses must implement for themselves")
    
    def gather_actions(self, updates, agent_keys, player_keys):
        raise NotImplementedError("Subclasses must implement for themselves")
    
    def apply_actions(self, agent_dict: AgentDict, action_dict: ActionDict) -> AgentDict:
        for key in action_dict:
            agent = agent_dict[key]
            agent.update_pose(action_dict[key])
        
        return agent_dict
    
    def play_round(self):
        raise NotImplementedError("Subclasses must implement for themselves")
    
    def to(self, device: str):
        self.model = self.model.to(device)

    def check_for_termination(self, running_logs, verbose: bool = False):
        return False


class RewardGradient(BaseStrategy):

    def __init__(
        self, n_prot_copies: int = 1,
        perturbation_mag: float = 0.5,
        distance_penalty: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_prot_copies = n_prot_copies
        self.perturbation_mag = perturbation_mag
        self.distance_penalty = distance_penalty
        self.n_grads = 2

    def setup_game(self, 
                   data: HeteroData, 
                   agent_keys: list[str], 
                   player_keys: list[str], 
                   agent_params: dict[str, float] = None) -> AgentDict:
        agent_dict = {}
        agent_cls = get_agent_cls(cls_name='reward')

        # receptor center 
        #center_receptor = torch.mean(data[agent_keys[-1]].pos, axis=0, keepdims=True)

        for key in agent_keys:
            is_player = False if key not in player_keys else True

            if agent_params is None:
                if is_player:
                    rot_lr, tr_lr = 1.0, 1.0
                else:
                    rot_lr, tr_lr = 0.0, 0.0
            else:
                rot_lr, tr_lr = agent_params[key]['rot_lr'], agent_params[key]['tr_lr']
            
            # overlap agents with receptor
            #center_agent = torch.mean(data[key].pos, axis=0, keepdims=True)
            #new_pos = data[key].pos - center_agent + center_receptor
            new_pos = data[key].pos

            agent_info = agent_cls(
                name=key, x=data[key].x, 
                edge_attr=data[key].edge_attr if "edge_attr" in data[key] else None,
                pos=new_pos, 
                is_player=is_player,
                pos_ref=None, 
                rot_lr=rot_lr, tr_lr=tr_lr
            )
            
            agent_dict[key] = agent_info

        return agent_dict

    def play_round(self, 
                   agent_dict: AgentDict, 
                   agent_keys: list[str], 
                   player_keys: list[str],
                   round_id: int) -> tuple[AgentDict, Metrics]:
        action_dict, metrics = self.compute_actions(
            agent_dict=agent_dict, agent_keys=agent_keys, 
            player_keys=player_keys,
        )

        agent_dict = self.apply_actions(agent_dict=agent_dict, action_dict=action_dict)
        agent_dict = self.update_lr(agent_dict=agent_dict, round_id=round_id)
        return agent_dict, metrics

    def update_lr(self, agent_dict: AgentDict, round_id: int) -> AgentDict:
        # Polynomial decay lr schedule"
        poly_dgr = 0.5
        for key in agent_dict:
            agent_dict[key].rot_lr *= ((round_id + 1) / (round_id + 2))**poly_dgr
            agent_dict[key].tr_lr *= ((round_id + 1) / (round_id + 2))**poly_dgr
        return agent_dict
    
    def compute_actions(self, 
                        agent_dict: AgentDict, 
                        agent_keys: list[str], 
                        player_keys: list[str]) -> tuple[ActionDict, Metrics]:
        
        grad_inputs = self._prepare_grad_inputs(agent_dict=agent_dict, player_keys=player_keys)
        
        complex_data = self.transform(agent_dict, agent_keys)

        (score, score_ref, score_diff), _ = self.model(complex_data)
        metrics = {'score_diff': score_diff.item()}

        if self.n_prot_copies > 1:
            agent_copies = self._make_agent_copies(
                agent_dict=agent_dict, 
                agent_keys=agent_keys, 
                player_keys=player_keys
            )

        poses = [agent_dict[key].pos for key in agent_keys]
        device = agent_dict[agent_keys[0]].pos.device
        distance_score = torch.zeros(1, device=device)
        if self.distance_penalty > 0:
            for i in range(round(len(poses)/2)):
                # minimum distance between agent-i and any other agent i+1,...,N
                min_dist = torch.cdist(poses[i].repeat(len(poses)-i-1,1), torch.cat(poses[i+1:],dim=0)).min()
                distance_score += self.distance_penalty * F.relu((min_dist -3*torch.ones_like(min_dist, device=device))**2)

        # Compute gradients
        grads = torch.autograd.grad(outputs=score_diff + distance_score, inputs=grad_inputs, retain_graph=False)
        grads = self._Riemannian_projections(grads, agent_dict, player_keys)
        
        scores = score_diff
        if self.n_prot_copies > 1:
            for agents_i in agent_copies:
                # Make sure the complex is prepared on the agents and not just on players
                complex_i = self.transform(agents=agents_i, agent_keys=agent_keys)
                (_, _, score_diff_i), _ = self.model(complex_i)
                
                # Gradient inputs always on the players
                grad_inputs_i = self._prepare_grad_inputs(agents=agents_i, player_keys=player_keys)
                grads_i = torch.autograd.grad(outputs=score_diff_i, inputs=grad_inputs_i, retain_graph=False)
                grads_i = self._Riemannian_projections(grads_i, agents_i, player_keys)

                grads = tuple(g1+g2 for g1,g2 in zip(grads, grads_i))
                scores = torch.cat((scores, score_diff_i))
        
            score_diff = torch.mean(scores)
            score_diff_var = torch.var(scores)

            metrics['score_diff_var'] = score_diff_var.item()
        grads = tuple(1/(self.n_prot_copies+1) * g for g in grads)
        
        grad_agents = self.gather_actions(updates=grads, agent_keys=agent_keys, player_keys=player_keys)

        for key in player_keys:
            grad_rot, grad_tr = grad_agents[key]['rot_vec'], grad_agents[key]['tr_vec']

            metrics[f"{key}_grad_rot_norm"] = torch.linalg.norm(grad_rot).item()
            metrics[f"{key}_grad_tr_norm"] = torch.linalg.norm(grad_tr).item()
            metrics[f"{key}_rot_norm"] = torch.linalg.norm(agent_dict[key].rot_vec).item()
            metrics[f"{key}_tr_norm"] = torch.linalg.norm(agent_dict[key].tr_vec).item()

        return grad_agents, metrics
    

    def _Riemannian_projections(self, grads: tuple, 
                               agent_dict: AgentDict, 
                               player_keys: list[str]) -> tuple[Tensor]:
        """ Projects gradients from Euclidean to Riemannian space, for each player """
        grads = list(grads)
        for idx, key in enumerate(player_keys):
            rot_grad = grads[idx * self.n_grads: (idx + 1) * self.n_grads][0]
            grads[idx * self.n_grads: (idx + 1) * self.n_grads][0] = torch.einsum("...ij,...jk->...ik", agent_dict[key].rot_vec_current, hat(rot_grad))
        grads = tuple(grads)
        return grads


    def _prepare_grad_inputs(self, 
                             agent_dict: AgentDict, 
                             player_keys: list[str]) -> tuple[Tensor]:
        grad_inputs = ()
        for player_key in player_keys:
            agent = agent_dict[player_key]

            # Rotation inputs
            # store current rot_vec for final Riemannian projection
            agent.rot_vec_current = agent.rot_vec.clone().detach().to(agent.rot_vec.device)
            
            rot_mat = axis_angle_to_matrix(agent.rot_vec_current)
            rot_coeff = torch.zeros(list(rot_mat.shape[:-2])+[3], device=agent.rot_vec.device, requires_grad=True)
            agent.rot_vec = matrix_to_axis_angle(expmap(rot_mat, torch.einsum("...ij,...jk->...ik", rot_mat, hat(rot_coeff).to(rot_mat.device))))
            #agent.rot_vec += rot_coeff # this would correspond to simple Eucledian gradients
            
            # Traslation inputs
            tr_coeff = torch.zeros(list(agent.tr_vec.shape[:-2])+[3], device=agent.tr_vec.device, requires_grad=True)
            agent.tr_vec += tr_coeff
            
            # Use above coeffs to modify agent.pose
            agent.update_pose()

            grad_inputs += (rot_coeff, tr_coeff)
        return grad_inputs
    
    def _make_agent_copies(self, 
                           agent_dict: AgentDict, 
                           agent_keys: list[str], 
                           player_keys: list[str]) -> list[AgentDict]:
        agent_cls = get_agent_cls(cls_name='reward')

        agent_copies = []
        for _ in range(self.n_prot_copies):
            agents_copy_i = {}

            for key in agent_keys:
                is_player = False if key not in player_keys else True

                agents_copy_i[key] = agent_cls(
                    name=agent_dict[key].name, 
                    x=agent_dict[key].x,
                    edge_attr=agent_dict[key].edge_attr, 
                    pos=agent_dict[key].pos, 
                    pos_ref=agent_dict[key].pos_ref,
                    is_player=agent_dict[key].is_player,
                    rot_lr=agent_dict[key].rot_lr, 
                    tr_lr=agent_dict[key].tr_lr
                )
            
            agent_copies.append(agents_copy_i)
        
        return agent_copies

    def gather_actions(self, 
                       updates: Tensor, 
                       agent_keys: list[str], 
                       player_keys: list[str]) -> ActionDict:
        action_updates = {}
        assert self.n_grads == 2

        for idx, key in enumerate(player_keys):
            updates_agent = updates[idx * self.n_grads: (idx + 1) * self.n_grads]
            assert len(updates_agent) == self.n_grads
            action_agent = {'rot_vec': updates_agent[0], 'tr_vec': updates_agent[1]}
            action_updates[key] = action_agent
        
        for key in agent_keys:
            if key not in action_updates:
                action_agent = {
                    'rot_vec': updates[0].new_zeros((1, 3)), 
                    'tr_vec': updates[0].new_zeros((1, 3))
                }
                action_updates[key] = action_agent
        
        return action_updates
    
    def check_for_termination(self, running_logs, verbose: bool = False):
        mean_thsld = 0.5
        t_window = 6
        std_thsld = 0.05
        if len(running_logs[list(running_logs.keys())[0]]) < t_window:
            return False
        gradient_log_keys = [key for key in running_logs.keys() if 'grad' in key]
        for key in gradient_log_keys:
            mean_last = np.mean(running_logs[key][-t_window:])
            std_last = np.std(running_logs[key][-t_window:])

            if verbose:
                print(f'last {key}: {running_logs[key][-1]}')
                print(f'mean of last {t_window} {key}: {mean_last}')
                print(f'std of last {t_window} {key}: {std_last}')
            
            scale = 5 if 'rot' in key else 1 # rotation gradients are larger
            if mean_last > scale*mean_thsld or std_last > scale*std_thsld:
                return False
        print('Game terminates!')
        return True


class RoundRobinRewardGradient(RewardGradient):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.player_idx = 0
    
    def play_round(self, 
                   agent_dict: AgentDict, 
                   agent_keys: list[str], 
                   player_keys: list[str],
                   round_id: int) -> tuple[AgentDict, Metrics]:
        
        if self.player_idx > len(player_keys)-1:
            self.player_idx = 0 # reset player index
            
        action_dict, metrics = self.compute_actions(
            agent_dict=agent_dict, agent_keys=agent_keys, 
            player_keys= [player_keys[self.player_idx]],
        )

        agent_dict = self.apply_actions(
            agent_dict=agent_dict, 
            action_dict={player_keys[self.player_idx]: action_dict[player_keys[self.player_idx]]}
        )
        
        # Update idx of playing agent (Round Robin)
        self.player_idx = (self.player_idx + 1) % len(player_keys)
        if self.player_idx == 0:
            agent_dict = self.update_lr(agent_dict=agent_dict, round_id=round_id % len(player_keys))
        return agent_dict, metrics


class ScoreMatching(BaseStrategy):

    def __init__(self,
        t_to_sigma: Callable,
        tr_sigmas: tuple[float],
        rot_sigmas: tuple[float],
        t_schedule: list[float],
        ode: bool = False,
        no_final_noise: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.t_to_sigma_fn = t_to_sigma
        self.t_schedule = t_schedule
        self.ode = ode
        self.tr_sigma_min, self.tr_sigma_max = tr_sigmas
        self.rot_sigma_min, self.rot_sigma_max = rot_sigmas
        self.no_final_noise = no_final_noise

    def setup_game(self, 
                   data: HeteroData, 
                   agent_keys: list[str], 
                   player_keys: list[str], 
                   agent_params: dict[str, float] = None) -> AgentDict:
        agent_dict = {}
        agent_cls = get_agent_cls(cls_name='score')

        for agent_key in agent_keys:

            is_player = agent_key in player_keys
            agent_info = agent_cls(
                name=agent_key, x=data[agent_key].x,
                edge_attr=data[agent_key].edge_attr if "edge_attr" in data[agent_key] else None,
                pos=data[agent_key].pos,
                is_player=is_player,
                tr_sigma_max=self.tr_sigma_max
            )

            agent_dict[agent_key] = agent_info

        return agent_dict
    
    def play_round(self, 
                   agent_dict: AgentDict, 
                   agent_keys: list[str], 
                   player_keys: list[str], 
                   round_id: int) -> tuple[AgentDict, dict[str, float]]:
        t_tr = self.t_schedule[round_id]
        t_rot = t_tr
        t_agents = {agent: (t_tr, t_rot) for agent in agent_keys}
        
        action_dict = self.compute_actions(
            agent_dict=agent_dict, agent_keys=agent_keys, 
            player_keys=player_keys, t_agents=t_agents,
            round_id=round_id
        )
        agent_dict = self.apply_actions(agent_dict=agent_dict, action_dict=action_dict)
        return agent_dict, {}

    def compute_actions(self, 
                        agent_dict: AgentDict, 
                        agent_keys: list[str], 
                        player_keys: list[str], 
                        t_agents: dict[str, tuple[float, float]],
                        round_id: int) -> ActionDict:
        complex_data = self.transform(
            complex_data=agent_dict, 
            t_agents=t_agents, 
            agents=agent_keys, 
            players=player_keys
        )

        complex_data.to(self.device)

        dt_tr = self.t_schedule[round_id] - self.t_schedule[round_id + 1] \
                if round_id < self.n_rounds - 1 else self.t_schedule[round_id]

        t_tr, t_rot = t_agents[agent_keys[0]]
        dt_rot = self.t_schedule[round_id] - self.t_schedule[round_id + 1] \
            if round_id < self.n_rounds- 1 else self.t_schedule[round_id]

        with torch.no_grad():
            tr_score, rot_score = self.model(complex_data)
            assert tr_score.size(0) == len(player_keys)
            assert rot_score.size(0) == len(player_keys)

        device = complex_data.x.device
        tr_sigma, rot_sigma = self.t_to_sigma_fn(t_tr, t_rot)

        tr_g = tr_sigma * torch.sqrt(
            torch.tensor(
                2 * np.log(self.tr_sigma_max / self.tr_sigma_min
            ), device=device)
        )
        rot_g = 2 * rot_sigma * torch.sqrt(
            torch.tensor(np.log(self.rot_sigma_max / self.rot_sigma_min
            ), device=device)
        )

        if self.ode:
            tr_update = (0.5 * tr_g ** 2 * dt_tr * tr_score)
            rot_update = (0.5 * rot_score * dt_rot * rot_g ** 2)
        else:
            if self.no_final_noise and round_id == self.n_rounds - 1:
                tr_z = torch.zeros(size=(len(player_keys), 3), device=device)
                rot_z = torch.zeros(size=(len(player_keys), 3), device=device)
            else:
                tr_z = torch.normal(mean=0, std=1, size=(len(player_keys), 3), device=device)
                rot_z = torch.normal(mean=0, std=1, size=(len(player_keys), 3), device=device)
        
            tr_update = (tr_g ** 2 * dt_tr * tr_score + tr_g * torch.sqrt(dt_tr) * tr_z)
            rot_update = (rot_score * dt_rot * rot_g ** 2 + rot_g * torch.sqrt(dt_rot) * rot_z)

        updates = (rot_update, tr_update)
        action_updates = self.gather_actions(
            updates=updates, agent_keys=agent_keys, player_keys=player_keys
        )

        return action_updates

    def gather_actions(self, 
                       updates: tuple[Tensor], 
                       agent_keys: list[str], 
                       player_keys: list[str]) -> ActionDict:
        rot_update, tr_update = updates
        action_updates = {}
        idx = 0

        for key in agent_keys:
            if key not in player_keys: # Agent is stationary, no need to update
                action_updates[key] = {
                    'tr_vec': rot_update.new_zeros((1, 3)), 
                    'rot_vec': rot_update.new_zeros((1, 3))
                }
            
            else:
                tr_agent, rot_agent = tr_update[idx:idx+1], rot_update[idx: idx+1]
                action_updates[key] = {'tr_vec': tr_agent, 'rot_vec': rot_agent}
                idx += 1

        return action_updates
    

class RoundRobinLangevin(ScoreMatching):

    def play_round(self, 
                   agent_dict: AgentDict, 
                   agent_keys: list[str], 
                   player_keys: list[str], 
                   round_id: int) -> tuple[AgentDict, dict[str, float]]:
        t_tr = self.t_schedule[round_id]
        t_rot = t_tr
        t_agents = {agent: (t_tr, t_rot) for agent in agent_keys}

        dt_tr = self.t_schedule[round_id] - self.t_schedule[round_id + 1] \
                if round_id < self.n_rounds - 1 else self.t_schedule[round_id]

        t_tr, t_rot = t_agents[agent_keys[0]]
        dt_rot = self.t_schedule[round_id] - self.t_schedule[round_id + 1] \
            if round_id < self.n_rounds- 1 else self.t_schedule[round_id]

        for idx, player in enumerate(player_keys):
            complex_data = self.transform(
                complex_data=agent_dict, 
                t_agents=t_agents, 
                agents=agent_keys, 
                players=player_keys
            )

            with torch.no_grad():
                tr_score, rot_score = self.model(complex_data)
                assert tr_score.size(0) == len(player_keys)
                assert rot_score.size(0) == len(player_keys)

            device = complex_data.x.device
            tr_sigma, rot_sigma = self.t_to_sigma_fn(t_tr, t_rot)

            tr_g = tr_sigma * torch.sqrt(
                    torch.tensor(
                        2 * np.log(self.tr_sigma_max / self.tr_sigma_min
                    ), device=device)
                )
            rot_g = 2 * rot_sigma * torch.sqrt(
                torch.tensor(np.log(self.rot_sigma_max / self.rot_sigma_min
                ), device=device)
            )

            if self.ode:
                tr_update = (0.5 * tr_g ** 2 * dt_tr * tr_score)
                rot_update = (0.5 * rot_score * dt_rot * rot_g ** 2)
            else:
                if self.no_final_noise and round_id == self.n_rounds - 1:
                    tr_z = torch.zeros(size=(len(player_keys), 3), device=device)
                    rot_z = torch.zeros(size=(len(player_keys), 3), device=device)
                else:
                    tr_z = torch.normal(mean=0, std=1, size=(len(player_keys), 3), device=device)
                    rot_z = torch.normal(mean=0, std=1, size=(len(player_keys), 3), device=device)
            
                tr_update = (tr_g ** 2 * dt_tr * tr_score + tr_g * torch.sqrt(dt_tr) * tr_z)
                rot_update = (rot_score * dt_rot * rot_g ** 2 + rot_g * torch.sqrt(dt_rot) * rot_z)

            # Mask to zero out all other updates except the one we want
            mask = torch.zeros(rot_update.size(0), dtype=torch.bool, device=rot_update.device)
            mask[idx] = 1

            rot_update = rot_update * mask.unsqueeze(1)
            tr_update = tr_update * mask.unsqueeze(1)
            updates = (rot_update, tr_update)

            action_dict = self.gather_actions(
                updates=updates, agent_keys=agent_keys, player_keys=player_keys
            )

            agent_dict = self.apply_actions(
                agent_dict=agent_dict, 
                action_dict=action_dict
            )

        return agent_dict, {}


def get_strategy_from_args(
        model: torch.nn.Module, 
        model_args: argparse.Namespace, 
        strategy_type: str, n_rounds: int,
        ode: bool = False,
        distance_penalty: float = 0.0,
        device: str = 'cpu'
    ) -> BaseStrategy:

    common_args = {
            'model': model, 'n_rounds': n_rounds, 'device': device
    }

    if strategy_type == "langevin":
        transform = construct_score_transform(
            args=model_args, mode='inference'
        )
        t_to_sigma_fn = partial(
            t_to_sigma,
            tr_sigma_min=model_args.tr_sigma_min,
            tr_sigma_max=model_args.tr_sigma_max,
            rot_sigma_min=model_args.rot_sigma_min,
            rot_sigma_max=model_args.rot_sigma_max
        )
        t_schedule = get_t_schedule(inference_steps=n_rounds)

        strategy = ScoreMatching(
            t_to_sigma=t_to_sigma_fn,
            tr_sigmas=(model_args.tr_sigma_min, model_args.tr_sigma_max),
            rot_sigmas=(model_args.rot_sigma_min, model_args.rot_sigma_max),
            t_schedule=t_schedule,
            ode=ode,
            transform=transform,
            **common_args
        )

    elif strategy_type == "reward_grad":

        transform = construct_reward_transform(model_args)
        
        strategy = RewardGradient(
            n_prot_copies=0,
            perturbation_mag=0, 
            distance_penalty=distance_penalty,
            transform=transform,
            **common_args
        )
    
    elif strategy_type == "round_robin_reward_grad":

        transform = construct_reward_transform(model_args)
        
        strategy = RoundRobinRewardGradient(
            n_prot_copies=0,
            perturbation_mag=0, 
            distance_penalty=distance_penalty,
            transform=transform,
            **common_args
        )

    elif strategy_type == "round_robin_langevin":
        transform = construct_score_transform(
            args=model_args, mode='inference'
        )
        t_to_sigma_fn = partial(
            t_to_sigma,
            tr_sigma_min=model_args.tr_sigma_min,
            tr_sigma_max=model_args.tr_sigma_max,
            rot_sigma_min=model_args.rot_sigma_min,
            rot_sigma_max=model_args.rot_sigma_max
        )
        t_schedule = get_t_schedule(inference_steps=n_rounds)

        strategy = RoundRobinLangevin(
            t_to_sigma=t_to_sigma_fn,
            tr_sigmas=(model_args.tr_sigma_min, model_args.tr_sigma_max),
            rot_sigmas=(model_args.rot_sigma_min, model_args.rot_sigma_max),
            t_schedule=t_schedule,
            ode=ode,
            transform=transform,
            **common_args
        )

    else:
        raise ValueError(f"Strategy of type {strategy} is not supported.")

    return strategy
