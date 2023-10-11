import os
import copy
import argparse
import traceback
import time

import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from dockgame.common.structure import write_to_pdb
from dockgame.data.parser import ProteinParser
from dockgame.data.featurize import construct_featurizer
from dockgame.models import load_model_from_args
from dockgame.game.engine import DockingEngine
from dockgame.game.strategy import get_strategy_from_args

from dockgame.utils.setup import read_complexes_from_txt
from dockgame.common.constants import DEVICE


def construct_filenames(complex_dir, complex_id, dataset: str = 'db5'):
    if dataset == 'db5':
        filenames = {
            "ligand": f"{complex_dir}/{complex_id}_l_b.pdb",
            "receptor": f"{complex_dir}/{complex_id}_r_b.pdb" 
        }

    elif dataset == "dips":
        # A simple fix checking for whether .dill is already part of the id
        if ".dill" in complex_id:
            filenames = {
                complex_id: f'{complex_dir}/{complex_id}'
            }
        else:
            filenames = {
                complex_id: f"{complex_dir}/{complex_id}.dill"
            }
        
    elif dataset == "multimer":
        filenames = {
            complex_id: f"{complex_dir}/{complex_id}-assembly1.cif"
        }
    
    return filenames


def run_inference(args):
    complex_dir = f"{args.data_dir}/raw/{args.dataset}/{args.complex_dir}"
    full_complex_list_path = f"{args.data_dir}/raw/{args.dataset}/{args.complex_list_file}"

    start_time = time.time()

    model, model_args = load_model_from_args(args, mode='test', return_model_args=True)
    
    if "center_complex" not in args:
        # Keeping the center_complex same as trained model if nothing specified
        args.center_complex = model_args.center_complex
    
    if "agent_type" not in args:
        # Keeping the agent_type same as trained model if nothing specified
        args.agent_type = model_args.agent_type
    
    if "featurizer" not in args:
        args.featurizer = model_args.featurizer

    # Used for evaluation later
    if args.task_name is None:
        args.task_name = f'dataset={args.dataset}_strategy={args.strategy}_agent={args.agent_type}_run={model_args.run_name}'

    complex_ids = read_complexes_from_txt(full_complex_list_path)

    print(f"Task Name: {args.task_name}", flush=True)
    print(f"Running gameplay for {len(complex_ids)} complexes", flush=True)
    print(f"Saving outputs of game to {args.out_dir}/{args.task_name}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print(flush=True)
    
    print("============================", flush=True)
    print(f"Game Args: {args}", flush=True)
    print(flush=True)
    
    print("============================", flush=True)
    print(f"Args from Restored Checkpoint: {model_args}", flush=True)
    print(flush=True)
    
    # Setting up the docking engine
    strategy = get_strategy_from_args(
        model=model, model_args=model_args,
        strategy_type=args.strategy, n_rounds=args.n_rounds,
        ode=args.use_ode if "use_ode" in args else False,
        distance_penalty=args.distance_penalty \
            if "distance_penalty" in args else 0.0,
        device=DEVICE
    )

    strategy.to(DEVICE)

    engine = DockingEngine(
        strategy=strategy,
        n_rounds=args.n_rounds, 
        agent_type=args.agent_type, 
        debug=False, log_every=None
    )

    # Setting up parser and featurizer
    parser = ProteinParser()
    featurizer = construct_featurizer(args=args)

    for complex_id in tqdm(complex_ids):
        try:
            filenames = construct_filenames(complex_dir=complex_dir,
                                            complex_id=complex_id, 
                                            dataset=args.dataset)

            structures = parser.parse(filenames)
            complex_data = HeteroData()
            complex_data = featurizer.featurize(
                structures=structures, graph=complex_data
            )
            complex_data.pdb_id = complex_id

            if args.center_complex:
                stationary_agent = complex_data.agent_keys[-1]
                agent_center = complex_data[stationary_agent].pos.mean(dim=0, keepdims=True)

                for agent in complex_data.agent_keys:
                    complex_data[agent].pos -= agent_center
                    complex_data[agent].pos_bound -= agent_center
                
                assert torch.allclose(
                    complex_data[stationary_agent].pos.mean(dim=0, keepdims=True),
                    torch.zeros(1, 3), atol=1e-3
                )

            player_keys = complex_data.agent_keys[:-1]

            for equilibrium_id in tqdm(range(args.n_equilibria)):
                os.makedirs(f'{args.out_dir}/{args.task_name}/{complex_id}', exist_ok=True)

                complex_data_copy = copy.deepcopy(complex_data)
                complex_data_copy = complex_data_copy.to(DEVICE)

                try:
                    file_prefix = f'{args.out_dir}/{args.task_name}/{complex_id}/{complex_id}_{equilibrium_id}'
                    file_init = f'{args.out_dir}/{args.task_name}/{complex_id}/{complex_id}-init-{equilibrium_id}.pdb'
                    proteins, _ = engine.play(
                        data=complex_data_copy, agent_params=None,
                        player_keys=player_keys,
                        monitor=None,
                        structures=None if not args.save_visualization else structures,
                        save_init_structures_to=file_init
                    )

                    if args.save_visualization:
                        structures_to_log = [
                            copy.deepcopy(proteins[agent].get_structure()) 
                            for agent in complex_data.agent_keys
                        ]

                        write_to_pdb(
                            structures=structures_to_log,
                            filename=f'{file_prefix}.pdb'
                        )

                    torch.save(proteins, f'{file_prefix}.pt')
                    del complex_data_copy

                except Exception as e:
                    print(f'Failed to generate equilibrium {equilibrium_id} for complex {complex_id} due to {e}')
                    traceback.print_exc()
                    del complex_data_copy
                    continue
        except Exception as e:
            print(f'Complex generation failed for {complex_id} with {e}. Skipping')
            traceback.print_exc()
            continue

    end_time = time.time()
    print(f'Total time spent: {end_time-start_time}', flush=True)


def parse_inference_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--out_dir", type=str, default="game_outputs")

    parser.add_argument("--task_name", type=str, default=None)

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str)

    parser.add_argument("--complex_dir", type=str, default='complexes')
    parser.add_argument("--complex_list_file", type=str, default='complexes.txt',
                        help="List of complexes to play game on")

    # Common args for game
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_rounds", type=int, default=50)
    parser.add_argument("--resolution",type=str, default="c_alpha")
    parser.add_argument("--n_equilibria", type=int, default=1)
    parser.add_argument("--save_visualization", action='store_true')
    parser.add_argument("--agent_type", default="protein", 
                        choices=["protein", "chain"])
    parser.add_argument("--score_fn_name", default="dock_low_res", 
                        choices=["dock_low_res", "dock_high_res"])
    parser.add_argument("--strategy", default="langevin")

    # Langevin specific args
    parser.add_argument("--use_ode", action='store_true')

    # Reward_grad specific args
    parser.add_argument("--perturbation_mag", type=float, default=0.0)
    parser.add_argument("--num_protein_copies", type=int, default=0)
    parser.add_argument("--distance_penalty", type=float, default=1.0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_inference_args()
    run_inference(args=args)
