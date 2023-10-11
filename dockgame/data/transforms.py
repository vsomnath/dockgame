from itertools import combinations
from functools import partial
import dataclasses
from typing import Callable, Any

import torch
import numpy as np
from torch_cluster import radius, radius_graph
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import SparseTensor

from dockgame.utils import geometry as geometry_ops
import dockgame.utils.so3 as so3
from dockgame.utils.diffusion import t_to_sigma
from dockgame.common.constants import DEVICE
import dockgame.analysis.metrics as metrics


# Type aliases
Tensor = torch.Tensor
Array = np.ndarray


@dataclasses.dataclass
class GameTransform(BaseTransform):
    """
    Base Transform class used in this project. This transform acts on game.agents
    or on data objects and induces appropriate constructions of self and cross
    graphs.
    """
    max_radius: float = 10.0
    max_neighbors: int = 24
    cross_max_radius: float = 10.0
    cross_max_neighbors: int = 24

    def build_self_graph(self, complex_data) -> Tensor:
        if isinstance(complex_data, HeteroData):
            agent_keys = complex_data.agent_keys
        elif isinstance(complex_data, dict):
            agent_keys = list(complex_data.keys())
        else:
            raise ValueError(f"Complex base type {type(complex_data)} is not supported.")
        
        device = complex_data[agent_keys[0]].x.device

        assert self.max_radius is not None

        if self.max_neighbors is None:
            self.max_neighbors = 32

        edge_index = torch.zeros((2, 0), device=device).long()

        n_nodes = 0
        for agent_key in agent_keys:

            radius_edges = radius_graph(
                x=complex_data[agent_key].pos,
                r=self.max_radius,
                max_num_neighbors=self.max_neighbors
            )

            edge_index = torch.cat([edge_index, radius_edges.long() + n_nodes], dim=1)
            n_nodes += complex_data[agent_key].x.size(0)
        
        return edge_index

    def build_cross_graph(self,
                          complex_data, 
                          cross_cutoff: float = None,
                          pos_attr: str = 'pos') -> Tensor:
        if isinstance(complex_data, HeteroData):
            agent_keys = complex_data.agent_keys
        elif isinstance(complex_data, dict):
            agent_keys = list(sorted(complex_data.keys()))
        else:
            raise ValueError(f"Complex base type {type(complex_data)} is not supported.")
        
        device = complex_data[agent_keys[0]].x.device

        if cross_cutoff is None:
            cross_cutoff = self.cross_max_radius
            
        if self.cross_max_neighbors is None:
            self.cross_max_neighbors = 32

        cross_edge_index = torch.zeros((2, 0), device=device).long()
        start_idxs, start_idx = [], 0

        for agent_key in agent_keys:
            start_idxs.append(start_idx)
            start_idx += complex_data[agent_key].x.size(0)

        comb = list(combinations(range(len(agent_keys)), r=2))

        for idx_a, idx_b in comb:
            if pos_attr == 'pos':
                agent_a_pos = complex_data[agent_keys[idx_a]].pos
                agent_b_pos = complex_data[agent_keys[idx_b]].pos
            
            elif pos_attr == 'ref':
                agent_a_pos = complex_data[agent_keys[idx_a]].pos_ref
                agent_b_pos = complex_data[agent_keys[idx_b]].pos_ref
            
            elif pos_attr == 'bound':
                agent_a_pos = complex_data[agent_keys[idx_a]].pos_bound
                agent_b_pos = complex_data[agent_keys[idx_b]].pos_bound
            
            else:
                raise ValueError(f"{pos_attr} pos type not supported.")

            cross_edges = radius(
                x=agent_a_pos,
                y=agent_b_pos,
                r=cross_cutoff,
                max_num_neighbors=self.cross_max_neighbors
            )
            
            src, dst = cross_edges
            src = src + start_idxs[idx_b]
            dst = dst + start_idxs[idx_a]

            cross_edges = torch.stack([src, dst], dim=0)
            cross_edge_index = torch.cat(
                [cross_edge_index, 
                    cross_edges, 
                    torch.flip(cross_edges, dims=[0])], 
                dim=1
            )
            
        return cross_edge_index


# ==============================================================================
# Transforms and Data objects used for Reward gradient style gameplay
# ==============================================================================


@dataclasses.dataclass
class RewardTransform(GameTransform):

    def __call__(self, 
                 complex_data: HeteroData, 
                 agents: list[str] = None, 
                 players: list[str] = None) -> Data:
        if agents is None:
            if isinstance(complex_data, HeteroData):
                agents = complex_data.agent_keys
            elif isinstance(complex_data, dict):
                agents = list(sorted(complex_data.keys()))

        complex_out = Data()
        complex_out.x = torch.cat(
            [complex_data[key].x for key in agents], dim=0
        )
        complex_out.pos = torch.cat(
            [complex_data[key].pos for key in agents], dim=0
        )
        complex_out.pos_ref = torch.cat(
            [complex_data[key].pos_ref for key in agents], dim=0
        )

        self_edge_index = self.build_self_graph(complex_data=complex_data)
        complex_out.edge_index = self_edge_index

        cross_edge_index = self.build_cross_graph(complex_data=complex_data)
        complex_out.cross_edge_index = cross_edge_index

        ref_cross_edge_index = self.build_cross_graph(
            complex_data=complex_data, pos_attr='ref')
        complex_out.ref_cross_edge_index = ref_cross_edge_index

        # (TODO): This part is still experimental and untested.
        if isinstance(complex_data, HeteroData):

            complex_out.y = torch.tensor(complex_data.y).float()
            complex_out.y_ref = torch.tensor(complex_data.y_ref).float()
            complex_out.y_diff = torch.tensor(complex_data.y - complex_data.y_ref).float()
        
            # if self.norm_method is not None and "sqrt_diff" in self.norm_method:
            #     sign_diff = torch.sign(complex_out.y_diff)
            #     complex_out.y_diff = sign_diff * torch.sqrt(torch.abs(complex_out.y_diff))

            complex_out.pos_bound = torch.cat(
            [complex_data[key].pos_bound for key in agents], dim=0
            )

            complex_rmsd, _ = metrics.compute_complex_rmsd_torch(
                complex_pred=complex_out.pos, 
                complex_true=complex_out.pos_bound
            )
            complex_rmsd_ref, _ =  metrics.compute_complex_rmsd_torch(
                complex_pred=complex_out.pos_ref,
                complex_true=complex_out.pos_bound
            )
        
            complex_out.rmsd = complex_rmsd
            complex_out.rmsd_ref = complex_rmsd_ref
            complex_out.rmsd_diff = complex_rmsd - complex_rmsd_ref

        complex_out.agent_keys = agents
        complex_out.protein_keys = ["ligand", "receptor"] # Hardcoded for now!
        num_nodes = sum(complex_data[agent].x.size(0) for agent in agents)
        complex_out.batch = torch.zeros((num_nodes,), dtype=torch.long)
        return complex_out.to(DEVICE)


# ==============================================================================
# Transforms and Data objects used for Score-Matching style gameplay
# ==============================================================================


class MultiAgentData(Data):

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "batch" in key:
            return int(value.max()) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        elif key == 'agent_membership':
            return self.num_agents
        elif key == 'agent_center_pos':
            return 0
        elif key == 'center_src':
            return self.num_nodes
        else:
            return 0
        
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or 'face' in key:
            return -1
        elif key == "agent_center_edges":
            return -1
        return 0


@dataclasses.dataclass
class ScoreMatchingTransform(GameTransform):
    """
    Transform class used on Data object during training and inference. This is 
    used for training and gameplay on the langevin dynamics based equilibria 
    computation method. 
    """
    t_to_sigma: Callable = None
    cross_cutoff_threshold: float = 40.0
    dynamic_max_cross: bool = False
    pert_strategy: str = "all-but-one"
    same_t_for_agent: bool = False

    def __call__(self, 
                 complex_data: HeteroData, 
                 t_agents: list[str] = None, 
                 agents: list[str] = None, 
                 players: list[str] = None) -> Data:
        if agents is None:
            if isinstance(complex_data, HeteroData):
                agents = complex_data.agent_keys
            elif isinstance(complex_data, dict):
                agents = list(sorted(complex_data.keys()))

        # During training, players are chosen arbitrarily based on strategy
        if self.pert_strategy is not None:
            if self.pert_strategy == "all-but-one":
                players = agents[:-1]
            elif self.pert_strategy == "one":
                players_all = agents[:-1]
                players = [np.random.choice(players_all)]
            elif self.pert_strategy == "all":
                players = agents

        if t_agents is None:
            if self.same_t_for_agent:
                t = np.random.uniform(low=1e-5, high=1.0)
                t_agents = {agent: (t, t) for agent in agents}
            else:
                ts =  {agent: np.random.uniform(low=1e-5, high=1.0) for agent in players}
                ts.update({agent: 0.0 for agent in agents if agent not in players})
                t_agents = {agent: (t, t) for agent, t in ts.items()}

        return self.apply_transform(complex_data, t_agents=t_agents, 
                                    agents=agents, players=players)

        
    def apply_transform(self, 
                        complex_data: HeteroData, 
                        t_agents: dict[str, tuple[float, float]], 
                        agents: list[str], 
                        players: list[str]) -> Data:
        if self.pert_strategy is not None:
            tr_score, rot_score = [], []
        node_t_tr, node_t_rot = [], []
        agent_t_tr, agent_t_rot = [], []

        center_src = []
        agent_membership = []

        t_max = max(t_agents[agent][0] for agent in players)

        if self.same_t_for_agent:
            assert all(t_max == t_agents[agent][0] for agent in players)

        n_nodes, agent_idx = 0, 0

        for agent in agents:
            t_tr, t_rot = t_agents[agent]

            node_t_tr.append(t_tr * torch.ones(complex_data[agent].num_nodes))
            node_t_rot.append(t_rot * torch.ones(complex_data[agent].num_nodes))

            if agent in players:

                agent_t_tr.append(t_tr)
                agent_t_rot.append(t_rot)

                center_src.append(torch.arange(complex_data[agent].num_nodes) + n_nodes)
                agent_membership.extend([agent_idx] * complex_data[agent].num_nodes)
                agent_idx += 1

                tr_sigma, rot_sigma = self.t_to_sigma(t_tr, t_rot)

                if self.pert_strategy is not None:
                    tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3), dtype=torch.float)
                    rot_update = so3.sample_vec(eps=rot_sigma)

                    _modify_agent(complex_data=complex_data, 
                                tr_update=tr_update, 
                                rot_update=rot_update,
                                agent_key=agent)
                    
                    tr_score_player = -tr_update / tr_sigma ** 2
                    rot_score_player = torch.tensor(so3.score_vec(vec=rot_update, 
                                                    eps=rot_sigma), dtype=torch.float32).unsqueeze(0)
                    
                    tr_score.append(tr_score_player)
                    rot_score.append(rot_score_player)

            n_nodes += complex_data[agent].num_nodes

        # Building different types of graphs
        self_edges = self.build_self_graph(complex_data)

        # tr_sigma max of tr_sigma for all agents
        tr_sigma_max_agents, _ = self.t_to_sigma(t_max, t_max)
        if self.dynamic_max_cross:
            cross_cutoff = (tr_sigma_max_agents * 3 + self.cross_cutoff_threshold)
        else:
            cross_cutoff = self.cross_max_radius

        cross_edges = self.build_cross_graph(complex_data, cross_cutoff=cross_cutoff) 

        center_pos = torch.cat([
            torch.mean(complex_data[agent].pos, dim=0, keepdim=True)
            for agent in agents if agent in players
        ], dim=0)

        complex_out = MultiAgentData()
        complex_out.x = torch.cat([complex_data[key].x for key in agents], dim=0)
        complex_out.pos = torch.cat([complex_data[key].pos for key in agents], dim=0)
        complex_out.num_agents = len(players)   

        # Setting the time
        complex_out.node_t_tr = torch.cat(node_t_tr, dim=0).float()
        complex_out.node_t_rot = torch.cat(node_t_rot, dim=0).float()
        complex_out.t_tr = torch.tensor(agent_t_tr).float()
        complex_out.t_rot = torch.tensor(agent_t_rot).float()

        # Adding edges and center position
        complex_out.edge_index = self_edges
        complex_out.cross_edge_index = cross_edges
        complex_out.agent_center_pos = center_pos

        complex_out.agent_membership = torch.tensor(agent_membership).long()
        complex_out.center_src = torch.cat(center_src, dim=0).long()

        # Computing the true scores
        if self.pert_strategy is not None:
            complex_out.tr_score = torch.cat(tr_score, dim=0)
            complex_out.rot_score = torch.cat(rot_score, dim=0)
            assert complex_out.tr_score.size(0) == len(players)
            assert complex_out.rot_score.size(0) == len(players)

        return complex_out


# ==============================================================================
# Transform construction functions
# ==============================================================================

def _modify_agent(
    complex_data: HeteroData, 
    tr_update: Tensor, 
    rot_update: Array, 
    agent_key: str):
    pos_orig = complex_data[agent_key].pos
    rot_vec = torch.tensor(rot_update, dtype=torch.float32)

    pos_updated = geometry_ops.apply_rigid_transform(
        pos=pos_orig,
        rot_vec=rot_vec,
        tr_vec=tr_update
    )

    complex_data[agent_key].pos = pos_updated
    assert not torch.allclose(pos_orig, complex_data[agent_key].pos)


def construct_score_transform(args, mode: str = "train") -> ScoreMatchingTransform:  

    t_to_sigma_fn = partial(
            t_to_sigma,
            tr_sigma_min=args.tr_sigma_min,
            tr_sigma_max=args.tr_sigma_max,
            rot_sigma_min=args.rot_sigma_min,
            rot_sigma_max=args.rot_sigma_max
        )
    
    # TODO: What should be done for same_t_for_agent
    if mode == "inference":
        pert_strategy = None
        same_t_for_agent = None
    else:
        pert_strategy = args.pert_strategy \
            if "pert_strategy" in args else "all-but-one"
        same_t_for_agent = args.same_t_for_agent \
            if "same_t_for_agent" in args else True
    
    if args.transform is None:
        print("Transform was given as none, using default ma_noise transform")

    if 'cross_cutoff_threshold' not in args:
        args.cross_cutoff_threshold = args.cross_max_radius

    if args.transform is None or args.transform == "ma_noise":
        transform = ScoreMatchingTransform(
            t_to_sigma=t_to_sigma_fn,
            max_radius=args.max_radius,
            max_neighbors=args.max_neighbors,
            cross_cutoff_threshold=args.cross_cutoff_threshold,
            cross_max_radius=args.cross_max_radius,
            cross_max_neighbors=args.cross_max_neighbors,
            dynamic_max_cross=args.dynamic_max_cross,
            pert_strategy=pert_strategy,
            same_t_for_agent=same_t_for_agent,
        )
    else:
        raise ValueError(f"{args.transform} is not supported.")

    return transform


def construct_reward_transform(args, mode: str = 'train') -> RewardTransform:

    transform = RewardTransform(
        max_radius=args.max_radius,
        max_neighbors=args.max_neighbors,
       # cross_max_radius=args.cross_max_radius,
       # cross_max_neighbors=args.cross_max_neighbors
    )
    return transform
