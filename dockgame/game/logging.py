"""Logging and saving utilities for game."""

import numpy as np
import wandb
import os

try:
    from dockgame.analysis.metrics import compute_pyrosetta_scores, get_score_fn
except Exception as e:
    print(e)
    pass

from dockgame.common.structure import write_to_pdb, Structure


class GameMonitor:

    def __init__(self,
        score_fn_name: str = 'dock_low_res',
        out_dir = "game_outputs"
    ):  
        try:
            self.score_fn = get_score_fn(score_fn_name=score_fn_name)
        except Exception as e:
            print(e)
            self.score_fn = None
        self.score_fn_name = score_fn_name
        self.out_dir = out_dir

    def log_at_round(self,
                     metrics: dict[str, float], 
                     round_id: int, 
                     complex_id: str,
                     structures=None,
                     save: bool = False) -> dict[str, float]:
        log_dict = {}
        if metrics is not None:
            log_dict = metrics.copy()

        if save and structures is not None:
            os.makedirs(f"{self.out_dir}/{complex_id}", exist_ok=True)
            files_structures = {
                structure.name: f"{self.out_dir}/{complex_id}/{complex_id}-round={round_id}-{structure.name}.pdb"
                for structure in structures
            }

            files_joint = {
                'joint': f"{self.out_dir}/{complex_id}/{complex_id}-round={round_id}.pdb"
            }

            print(f"Saving structures after Round {round_id}", flush=True)
            self.write_to_pdb(
                structures=structures, filenames=files_structures
            )

            self.write_to_pdb(
                structures=structures, filenames=files_joint
            )

            log_dict[f"{complex_id}_plot"] = wandb.Molecule(files_joint['joint'])

            if self.score_fn is not None:
                keys = [structure.name for structure in structures]
                score_indi = self.compute_pyrosetta_scores(
                    filenames=files_structures, agent_keys=keys
                )
                score_joint = self.compute_pyrosetta_scores(
                    filenames=files_joint, agent_keys=keys
                )
                #rg = self.compute_radius_of_gyration(filename=filename)

                metrics['pyrosetta_score_indi'] = score_indi
                metrics['pyrosetta_score_joint'] = score_joint
                #metrics['radius_gyration'] = rg
            
                log_dict[f"pyrosetta_score_indi"] = score_indi
                log_dict[f"pyrosetta_score_joint"] = score_joint
                #log_dict[f"radius_gyration"] = rg

            log_dict["step"] = round_id
        
        wandb.log(log_dict)

        if metrics is not None:
            base_metrics = [
                "score_diff", 
                "complex_rmsd", 
                "pyrosetta_score_indi", 
                "pyrosetta_score_joint"
            ]

            print_msg = f"Metrics computed after Round {round_id}: "
            for metric in base_metrics:
                if metric in metrics.keys():
                    print_msg += f"{metric}: {np.round(metrics[metric], 4)} "

            print_msg += "\nOther metrics: "
            for metric, metric_val in metrics.items():
                if metric in base_metrics:
                    continue
                print_msg += f"{metric}: {np.round(metric_val, 4)} "
            self.display(print_msg)
            self.display(" ")
            
        return log_dict

    def write_to_pdb(self, 
                     structures: list[Structure], 
                     filenames: dict[str, str]):
        if len(filenames) > 1:
            for structure in structures:
                filename = filenames[structure.name]
                write_to_pdb(structures=structure, filename=filename)
        
        else:
            write_to_pdb(
                structures=structures,
                filename=filenames['joint']
            )

    def compute_pyrosetta_scores(self, 
                                 filenames: dict[str, str], 
                                 agent_keys: list[str] = None) -> float:
        return compute_pyrosetta_scores(
                filenames=filenames, agent_keys=agent_keys, 
                score_fn=self.score_fn
            )
        
    def display(self, msg: str):
        print(msg, flush=True)

    def update_running_logs(self, 
                            running_logs: dict[str, list[float]], 
                            log_dict: dict[str]):
        # append log_dict statistics to running_logs
        for key in log_dict.keys():
            if 'plot' in key or 'bound' in key and 'score' not in key: 
                continue # do not log wandb.Molecules
            if f"{key}s" not in running_logs.keys():
                running_logs[f"{key}s"] = []
            running_logs[f"{key}s"].append(log_dict[key])
        # for key in metrics.keys():
        #     if f"{key}s" not in running_logs.keys():
        #         running_logs[f"{key}s"] = []
        #     running_logs[f"{key}s"].append(metrics[key])
        return
