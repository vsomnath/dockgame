import argparse
import os
import time

import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

from dockgame.models import load_model_from_args
from dockgame.game import DockingEngine, get_strategy_from_args
from dockgame.common.constants import DEVICE

Args = argparse.Namespace
Array = np.ndarray


# ==============================================================================
# Game Setup 
# ==============================================================================


def setup_game(args: Args) -> DockingEngine:
    model, model_args = load_model_from_args(
        args, mode='test', return_model_args=True)
    
    if "center_complex" not in args:
        # Keeping the center_complex same as trained model if nothing specified
        args.center_complex = model_args.center_complex
    if "agent_type" not in args:
        # Keeping the agent_type same as trained model if nothing specified
        args.agent_type = model_args.agent_type
    
    if "featurizer" not in args:
        args.featurizer = model_args.featurizer

    strategy = get_strategy_from_args(
        model=model, model_args=model_args,
        strategy_type=args.strategy, n_rounds=args.n_rounds,
        ode=args.use_ode if "use_ode" in args else False,
        distance_penalty=args.distance_penalty \
            if "distance_penalty" in args else 0.0
    )

    strategy.to(DEVICE)

    engine = DockingEngine(
        n_rounds=args.n_rounds, 
        strategy=strategy,
        agent_type=args.agent_type,
        debug=args.debug,
        log_every=args.log_every,
        save_trajectory=args.save_trajectory if 'save_trajectory' in args else False
    )

    save_dir = f"{args.out_dir}/{args.run_name}/{args.dataset}"
    args.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    config_file = f"{save_dir}/game_config.yml"
    yaml_dump = yaml.dump(args.__dict__)
    with open(config_file, "w") as f:
        f.write(yaml_dump)
    
    print(f"Saved game config to {config_file}", flush=True)
    print(flush=True)

    return engine


# ==============================================================================
# Game Logging Related
# ==============================================================================


def print_game_time_statistics(
    n_complexes: int, 
    n_rounds: int, 
    start_time: time.time
):
    end_time = time.time()
    time_taken = end_time - start_time
    time_taken_str = str(timedelta(seconds=end_time - start_time))
    avg_time_per_complex = time_taken / n_complexes
    avg_time_per_round = time_taken / (n_complexes * n_rounds)

    msg = f"Time taken: {time_taken_str}"
    msg += f" | Avg. Time per complex (in s): {avg_time_per_complex:.4f}"
    msg += f" | Avg. Time per round (in s): {avg_time_per_round:.4f}"

    print(f"Finished playing game for {n_complexes} complexes", flush=True)
    print(msg, flush=True)


def tolerant_stats(arrs: Array) -> tuple[Array, Array]:
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


def plot_game_logs(
    game_logs: dict[str, float], 
    savedir: str = None, 
    to_plot: list[str] = None
):        
    pdb_ids = list(game_logs.keys())

    # Plot absolute pyrosetta scores, score_diffs, and min_dists over game rounds
    for key in to_plot:
        try:
            plt.figure()
            values = []
            for pdb_id in pdb_ids:
                for idx_games in range(len(game_logs[pdb_id])):
                    values.append(game_logs[pdb_id][idx_games][key])
            mean, std = tolerant_stats(values)
            plt.plot(mean)
            plt.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.3)
            #plt.legend()
            plt.title(key)
            if savedir:
                plt.savefig(f"{savedir}/{key}.png")
            plt.close()
        except:
            print(f"Could not plot {key}")

    pdb_id = pdb_ids[0]
    if 'pyrosetta_score_indis' in game_logs[pdb_id][0]:
        # Plot % of decrease/increase of pyrosetta scores
        plt.figure()
        values = []

        for pdb_id in pdb_ids:
            for idx_games in range(len(game_logs[pdb_id])):
                scores = game_logs[pdb_id][idx_games]['pyrosetta_score_indis']
                values.append((np.array(scores)-scores[0])/abs(scores[0]))
        mean, std = tolerant_stats(values)
        plt.plot(mean)
        plt.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.3)
        #plt.legend()
        plt.title('Game-normalized Pyrosetta score')
        if savedir:
            plt.savefig(f"{savedir}/norm_pyrosetta_scores.png")
        plt.close()

    # Scatter RMSDs for each complex 
    fig = plt.figure()
    for i, pdb_id in enumerate(pdb_ids):
        rmsd_final = []
        for idx_games in range(len(game_logs[pdb_id])):
            rmsds = game_logs[pdb_id][idx_games]['complex_rmsds']
            plt.scatter(i*np.ones(len(rmsds)), rmsds, marker='s', color='black')
            plt.scatter(i, rmsds[-1], marker='s', color='red', label='final RMSD')
            rmsd_final.append(rmsds[-1])
        plt.scatter(i*np.ones(len(rmsd_final)), rmsd_final, marker='s', color='red', label='final RMSD')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys()) 
    plt.title('RMSDs observed')
    if savedir:
        plt.savefig(f"{savedir}/rmsds.png")
    plt.close()

    fig = plt.figure()
    for i, pdb_id in enumerate(pdb_ids):
        for idx_games in range(len(game_logs[pdb_id])):
            duration = len(game_logs[pdb_id][idx_games]['complex_rmsds'])
            plt.scatter(i, duration, marker='s', color='black')
    plt.ylabel('# of rounds')
    plt.title('Game durations')
