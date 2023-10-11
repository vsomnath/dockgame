import copy
import dataclasses
import wandb
from typing import Sequence

import torch
from torch_geometric.data import HeteroData
import numpy as np

from dockgame.common.structure import Protein, write_to_pdb
from dockgame.game import strategy, logging
from dockgame.analysis.metrics import compute_complex_rmsd_torch


RunningLogs = dict[str, list[float]]
AgentDict = strategy.AgentDict
GameOutputs = tuple[AgentDict, RunningLogs]


@dataclasses.dataclass
class DockingEngine:

    n_rounds: int
    strategy: strategy.BaseStrategy
    agent_type: str = 'protein'
    debug: bool = False
    log_every: int = 1
    save_trajectory: bool = False

    def play(
        self, data: HeteroData,
        player_keys: Sequence[str] = None,  
        agent_params: dict[str, str] = None, 
        monitor: logging.GameMonitor = None,
        structures: Sequence[Protein] = None,
        save_init_structures_to: str = None,
    ) -> GameOutputs:
        
        agent_keys = data.agent_keys
        complex_id = data.pdb_id[0] if isinstance(data.pdb_id, list) else data.pdb_id

        if player_keys is None:
            player_keys = agent_keys

        agent_dict = self.strategy.setup_game(
            data=data, agent_keys=agent_keys, 
            player_keys=player_keys, 
            agent_params=agent_params
        )

        if structures is not None:
            if self.agent_type == 'chain':
                structure_dict = {}

                for protein in structures:
                    for chain in protein.get_chains():
                        structure_dict[chain.name] = chain
            else:
                structure_dict = {
                    protein.name: protein
                    for protein in structures
                }

        if monitor is not None:
            monitor.display(" ")

        if monitor is not None and self.debug:
            if structures is not None:
                structures = [
                    structure_dict[agent_key] for agent_key in agent_keys
                ]

            filenames_structures = {
                structure.name: f"{monitor.out_dir}/{complex_id}-bound-{structure.name}.pdb"
                for structure in structures
            }

            filenames_joint = {
                'joint': f"{monitor.out_dir}/{complex_id}-bound.pdb"
            }

            monitor.write_to_pdb(
                structures=structures,
                filenames=filenames_joint
            )

            monitor.write_to_pdb(
                structures=structures,
                filenames=filenames_structures
            )

            log_dict = {}
            log_dict[f"{complex_id}_bound"] = wandb.Molecule(filenames_joint['joint'])
            wandb.log(log_dict)

            init_metrics = {}
            if monitor.score_fn is not None:
                score_init = monitor.compute_pyrosetta_scores(
                        filenames=filenames_structures, agent_keys=agent_keys
                    )
                score_joint = monitor.compute_pyrosetta_scores(
                    filenames=filenames_joint, agent_keys=None
                )

                init_metrics['pyrosetta_bound_score_indi'] = np.round(score_init, 4)
                init_metrics['pyrosetta_bound_score_joint'] = np.round(score_joint, 4)

            print_msg = f"Bound complex metrics: {init_metrics} \n"
            monitor.display(print_msg)

        # Initialize all agents with structures if structures is not None
        if structures is not None:
            print("Structures were supplied. Adding it to the agents for visualization...", flush=True)
            for key in agent_keys:
                assert key in structure_dict, f'{key} not found in {structure_dict.keys()}'
                agent_dict[key].add_structure(structure_dict[key])

            if save_init_structures_to is not None:
                print(f"Saving randomized structures to: {save_init_structures_to}", flush=True)
                structures_init = [
                    copy.deepcopy(agent_dict[agent].get_structure()) for agent in agent_keys
                ]
                write_to_pdb(
                    structures=structures_init, 
                    filename=save_init_structures_to
                )

        # Logging setup at the start of the game
        if monitor is not None:
            print_msg = "Initialized Game: \n"

            if self.debug:
                for key in player_keys:
                    agent = agent_dict[key]
                    rot_norm_init = torch.linalg.norm(agent.rot_init).item()
                    tr_norm_init = torch.linalg.norm(agent.tr_init).item()

                    print_msg += f"{key}: Rot Norm B->UB: {np.round(rot_norm_init, 4)}, "
                    print_msg += f"Tr Norm B->UB: {np.round(tr_norm_init, 4)}\n"

                    if hasattr(agent, 'rot_vec'):
                        rot_norm = torch.linalg.norm(agent.rot_vec).item()
                        tr_norm = torch.linalg.norm(agent.tr_vec).item()

                        print_msg += f"{key}: Rot Norm t=0: {np.round(rot_norm, 4)}, "
                        print_msg += f"Tr Norm t=0: {np.round(tr_norm, 4)} \n"

            monitor.display(print_msg)

        running_logs = {}
        if monitor is not None and self.debug:
            # Logs after Round 0
            pos_bound = torch.cat(
                [data[key].pos_bound for key in agent_keys], dim=0
            )
            _, metrics = compute_complex_rmsd_torch(
                    complex_pred=torch.cat([agent_dict[key].pos for key in agent_keys], dim=0),
                    complex_true=pos_bound
                )
            
            com_dists = []
            min_dists = []

            for key in agent_keys[:-1]:
                last_agent = agent_keys[-1]
                com_agent = agent_dict[key].pos.mean(dim=0)
                com_stationary_agent = agent_dict[last_agent].pos.mean(dim=0)

                com_dist = torch.sqrt(((com_agent - com_stationary_agent)**2).sum())
                com_dists.append(com_dist.item())

                min_dists.append(torch.cdist(agent_dict[key].pos, agent_dict[last_agent].pos).min().item())

            metrics['min_com_dist'] = min(com_dists)
            metrics['max_com_dist'] = max(com_dists)
            metrics['min_res_dist'] = min(min_dists)

            structures_to_log = None
            if self.save_trajectory and structures is not None:
                structures_to_log = [
                    copy.deepcopy(agent_dict[agent].get_structure()) for agent in agent_keys
                ]

            log_dict = monitor.log_at_round(
                metrics=metrics, round_id=0, complex_id=complex_id,
                save=self.debug, structures=structures_to_log
            )
        
            monitor.update_running_logs(
                log_dict=log_dict, running_logs=running_logs
            )
            monitor.update_running_logs(
                log_dict=init_metrics, running_logs=running_logs
            )


        if monitor is not None:
            monitor.display("====================")
            monitor.display("Starting gameplay ")
            monitor.display("==================== \n")

        for round_id in range(self.n_rounds):
            agent_dict, metrics = self.strategy.play_round(
                agent_dict=agent_dict, agent_keys=agent_keys, 
                player_keys=player_keys, round_id=round_id
            )

            if self.debug:
                _, rmsd_metrics = compute_complex_rmsd_torch(
                    complex_pred=torch.cat([agent_dict[key].pos for key in agent_keys], dim=0),
                    complex_true=pos_bound
                )
                metrics.update(rmsd_metrics)

                com_dists = []
                min_dists = []

                for key in agent_keys[:-1]:
                    last_agent = agent_keys[-1]
                    com_agent = agent_dict[key].pos.mean(dim=0)
                    com_stationary_agent = agent_dict[last_agent].pos.mean(dim=0)

                    com_dist = torch.sqrt(((com_agent - com_stationary_agent)**2).sum())
                    com_dists.append(com_dist.item())

                    min_dists.append(torch.cdist(agent_dict[key].pos, agent_dict[last_agent].pos).min().item())

                metrics['min_com_dist'] = min(com_dists)
                metrics['max_com_dist'] = max(com_dists)
                metrics['min_res_dist'] = min(min_dists)

            if monitor is not None:
                structures_to_log = None
                if self.save_trajectory and structures is not None:
                    structures_to_log = [
                        copy.deepcopy(agent_dict[agent].get_structure()) 
                        for agent in agent_keys
                    ]

                if round_id % self.log_every == 0:
                    log_dict = monitor.log_at_round(
                        metrics=metrics, round_id=round_id + 1, complex_id=complex_id,
                        save=self.debug, structures=structures_to_log
                    )
                    
                monitor.update_running_logs(
                    log_dict=log_dict, running_logs=running_logs
                )

            terminate_game = self.strategy.check_for_termination(running_logs=running_logs)

            if terminate_game:
                break

        if monitor is not None:

            if structures is not None and self.debug:
                structures_to_log = [
                    copy.deepcopy(agent_dict[agent].get_structure()) 
                    for agent in agent_keys
                ]

                filename = f'{monitor.out_dir}/{complex_id}-round={self.n_rounds}.pdb'
                monitor.write_to_pdb(
                    structures=structures_to_log, 
                    filenames={'joint': filename}
                )

                log_dict = {}
                log_dict[f"{complex_id}_final"] = wandb.Molecule(filename)
                wandb.log(log_dict)

                if monitor.score_fn is not None:

                    energy = monitor.compute_pyrosetta_scores(filenames={'joint': filename})
                    monitor.display(f"PyRosetta Score of final complex: {np.round(energy, 4)}")
                    monitor.display("")

        return agent_dict, running_logs
