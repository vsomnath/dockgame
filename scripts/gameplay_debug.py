import os
import time
import torch
from tqdm import tqdm
import copy

from dockgame.game.utils import (
    setup_game, print_game_time_statistics, plot_game_logs
)
from dockgame.utils.setup import wandb_setup, parse_game_args, read_complexes_from_txt
from dockgame.game.logging import GameMonitor
from dockgame.data.parser import ProteinParser
from dockgame.common.constants import DEVICE



def construct_filenames(complex_id: str, dataset: str = 'db5'):
    if dataset == 'db5':
        filenames = {
            "ligand": f"{complex_id}_l_b.pdb",
            "receptor": f"{complex_id}_r_b.pdb" 
        }

    elif dataset == "dips":
        # A simple fix checking for whether .dill is already part of the id
        if ".dill" in complex_id:
            filenames = {
                complex_id: complex_id
            }
        else:
            filenames = {
                complex_id: f"{complex_id}.dill"
            }
        
    elif dataset == "multimer":
        filenames = {
            complex_id: f"{complex_id}.cif"
        }
    
    return filenames


def compute_equilibria_for_complex(game, complex_id, args):
    
    dataset = args.dataset
    agent_type = args.agent_type
    resolution = args.resolution if 'resolution' in args else 'c_alpha'
    output_dir = args.save_dir

    # Constructing the full path to processed directory
    processed_data_dir = f"{args.data_dir}/processed/{dataset}/agent={agent_type}_resolution={resolution}"
    if 'featurizer' in args and args.featurizer is not None:
        processed_data_dir += f"_featurizer={args.featurizer}"

    if 'center_complex' in args and args.center_complex:
        processed_data_dir += "_centered"

    all_logs = []

    print(f"Computing {args.n_equilibria} equilibria for {complex_id}")

    for equilibrium_id in tqdm(range(args.n_equilibria)):
        complex_file = f"{processed_data_dir}/{complex_id}.pt"
        complex_base = torch.load(complex_file, map_location='cpu')
        complex_base.pdb_id = complex_id

        monitor = None
        if args.debug or args.save_vis:
            monitor = GameMonitor(
                score_fn_name=args.score_fn_name,
                out_dir=output_dir
            )

        agent_params = None
        if args.strategy == "reward_grad":
            agent_params = {
                agent: {"rot_lr": args.rot_lr[idx], "tr_lr": args.tr_lr[idx]}
                for idx, agent in enumerate(complex_base.agent_keys)
            }

            print(agent_params)

        if agent_type == "protein":
            player_keys = ["ligand"]
        else:
            player_keys = complex_base.agent_keys[:-1]

        structures = None
        save_init_structures_to = None

        if args.save_vis:
            full_complex_dir = f"{args.data_dir}/raw/{dataset}/{args.complex_dir}/"
            filenames = construct_filenames(complex_id=complex_id, dataset=dataset)

            for key, value in filenames.items():
                filenames[key] = f"{full_complex_dir}/{value}"
            
            parser = ProteinParser()
            structures = parser.parse(filenames=filenames)
            save_init_structures_to = f'{output_dir}/{complex_id}-init-{equilibrium_id+1}.pdb'

        if len(player_keys) > 0:
            complex_base_copy = copy.deepcopy(complex_base)
            complex_base_copy = complex_base_copy.to(DEVICE)

            proteins, game_logs = game.play(
                data=complex_base_copy,
                agent_params=agent_params,
                player_keys=player_keys,
                monitor=monitor, 
                structures=structures,
                save_init_structures_to=save_init_structures_to
            )

            os.makedirs(f"{output_dir}/{complex_id}", exist_ok=True)
            torch.save(proteins, f"{output_dir}/{complex_id}/structures_{equilibrium_id+1}.pt")
            torch.save(game_logs, f"{output_dir}/{complex_id}/game_logs_{equilibrium_id+1}.pt")
            print(flush=True)
            all_logs.append(game_logs)

            del complex_base_copy
 
    return all_logs


def run_gameplay(args):     
    # Build game config and game
    game = setup_game(args)

    # Start the game
    start_time = time.time()

   # Read complex ids from file
    full_complex_id_path = f"{args.data_dir}/raw/{args.dataset}/{args.complex_list_file}"
    pdb_ids = read_complexes_from_txt(full_complex_id_path)
    print(f"Running evaluation for {len(pdb_ids)} complexes", flush=True)
    print(flush=True)

    to_plot = ['pyrosetta_score_indis', 'pyrosetta_score_joints', 'score_diffs', 'min_res_dists', 'complex_rmsds']
    
    game_logs = {}

    for i, complex_id in enumerate(pdb_ids):
        all_logs = compute_equilibria_for_complex(
            game=game, complex_id=complex_id, args=args
        )
        game_logs[complex_id] = all_logs
        if i % 3 == 0:
            plot_game_logs(game_logs, savedir=args.save_dir, to_plot=to_plot)
        
        torch.save(game_logs, f"{args.save_dir}/game_logs.pt")

    print_game_time_statistics(
        n_complexes=len(pdb_ids), 
        start_time=start_time, 
        n_rounds=args.n_rounds
    )
    
    torch.save(game_logs, f"{args.save_dir}/game_logs.pt")
    plot_game_logs(game_logs=game_logs, savedir=args.save_dir, to_plot=to_plot)


def main():
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=4)

    # Load args from command line and replace values with those from config
    args = parse_game_args()
    print("Args after reading from config file: ")
    print(args, flush=True)
    wandb_setup(args)

    run_gameplay(args=args)


if __name__ == "__main__":
    main()
