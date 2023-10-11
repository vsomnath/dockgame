import os
import time
import datetime
import json
import argparse
import pickle
from multiprocessing import Pool
import traceback

import numpy as np
import matplotlib.pyplot as plt
try:
    from pyrosetta.distributed import packed_pose
except Exception as e:
    pass

from dockgame.data.tools.decoys import RosettaDecoys
from dockgame.data.tools.decoys.pyrosetta_decoys import get_score_fn
from dockgame.data.parser import ProteinParser
from dockgame.common.structure import to_pdb
from dockgame.utils.setup import read_complexes_from_txt


def get_decoy_generator(decoy_type: str):
    if decoy_type == "rosetta":
        decoy_generator = RosettaDecoys
    else:
        raise ValueError(f"Decoy generator of type {decoy_type} is not supported.")
    return decoy_generator
    

def get_filenames_from_dataset(args, pdb_ids, full_complex_dir, dataset):
    if dataset == "db5":
        filenames = [{
            'ligand': f"{full_complex_dir}/{pdb_id}_l_b.pdb",
            'receptor': f"{full_complex_dir}/{pdb_id}_r_b.pdb"
        } for pdb_id in pdb_ids]
        non_failing_ids = pdb_ids
    
    elif dataset.lower() == "dips":
        filenames = []
        non_failing_ids = []
        for pdb_id in pdb_ids:
            try:
                if ".dill" in pdb_id:
                    base_file = {
                        pdb_id: f"{full_complex_dir}/{pdb_id}"
                    }

                else:
                    base_file = {
                        pdb_id: f"{full_complex_dir}/{pdb_id}.dill"
                    }

                structures = ProteinParser().parse(base_file)

                filename = {}

                for structure in structures:
                    lines = to_pdb(structure=structure)
                    if '.dill' in pdb_id:
                        pdb_id_no_ext = os.path.splitext(pdb_id)[0]
                    else:
                        pdb_id_no_ext = pdb_id 

                    if 'tmp_dir' not in args or args.tmp_dir is None:
                        tmp_dir = f"{args.data_dir}/tmp-pyrosetta"
                    else:
                        tmp_dir = args.tmp_dir

                    tmp_save_file = f"{tmp_dir}/dips/{pdb_id_no_ext}-{structure.name}.pdb"
                    os.makedirs(os.path.dirname(tmp_save_file), exist_ok=True)
                    print(tmp_save_file)

                    filename[structure.name] = tmp_save_file

                    with open(f"{tmp_save_file}", "w") as f:
                        f.write(lines)

                filenames.append(filename)
                non_failing_ids.append(pdb_id)
            except Exception as e:
                print(f"Getting filename failed for {pdb_id} with {e}", flush=True)
                traceback.print_exc()

    elif dataset.lower() == "multimer":
        filenames = [
            {pdb_id: f"{full_complex_dir}/{pdb_id}.cif"} for pdb_id in pdb_ids
        ]
        non_failing_ids = pdb_ids
    
    return filenames, non_failing_ids


def print_statistics(inputs, prefix: str = None):
    print_msg = ""
    if prefix is not None:
        print_msg += prefix
    
    mean = np.round(np.mean(inputs), 4)
    std = np.round(np.std(inputs), 4)
    min_val = np.round(np.min(inputs), 4)
    max_val = np.round(np.max(inputs), 4)

    print_msg += f" Mean={mean}, Std={std}, Min={min_val}, Max={max_val}"
    print(print_msg, flush=True)


def remove_existing_decoys(args, filenames, pdb_ids, mode):
    non_existing_filenames = []
    non_existing_pdb_ids = []
    for i, (filename, pdb_id) in enumerate(zip(filenames, pdb_ids)):
        decoy_dir = f"{args.data_dir}/raw/{args.dataset}/decoys"
        decoy_info_file = f"{pdb_id}-agent={args.agent_type}-decoys={args.num_decoys}"
        decoy_info_file += f"-score={args.score_fn_name}"
        decoy_info_file += f"-max_tr={args.max_tr}"
        decoy_info_file += f"-{mode}.decoy"
        decoy_info_file = f"{decoy_dir}/{decoy_info_file}"

        if os.path.exists(decoy_info_file):
            msg = f"Decoy file already exists for {pdb_id}. Skipping."
            msg += "Please use --overwrite to overwrite this file"
            print(msg, flush=True)
        else:
            non_existing_filenames.append(filename)
            non_existing_pdb_ids.append(pdb_id)
    return non_existing_filenames, non_existing_pdb_ids


def decoy_generation_loop(args, mode, pdb_ids, map_fn, decoy_generator, score_fn):
    # Setup decoy directory
    base_dir = f"{args.data_dir}/raw/{args.dataset}/"
    decoy_dir = f"{args.data_dir}/raw/{args.dataset}/decoys"
    os.makedirs(decoy_dir, exist_ok=True)

    if args.save_histograms:
        histogram_dir = f"{base_dir}/histograms"
        os.makedirs(histogram_dir, exist_ok=True)

    filenames, pdb_ids = get_filenames_from_dataset(
        args=args,
        pdb_ids=pdb_ids,
        full_complex_dir=f"{base_dir}/{args.complex_dir}",
        dataset=args.dataset
    )

    if not args.overwrite:
        filenames, pdb_ids = remove_existing_decoys(args=args, filenames=filenames, pdb_ids=pdb_ids, mode=mode)

    for decoy_info in map_fn(decoy_generator.generate_decoys, zip(pdb_ids, filenames)):
        if decoy_info is not None:
            decoy_id = decoy_info[0]["id"]

            decoy_info_file = f"{decoy_id}-agent={args.agent_type}-decoys={args.num_decoys}"
            decoy_info_file += f"-score={args.score_fn_name}"
            decoy_info_file += f"-max_tr={args.max_tr}"
            decoy_info_file += f"-{mode}.decoy"

            decoy_info_file = f"{decoy_dir}/{decoy_info_file}"
            
            if args.dataset.lower() == "dips":
                decoy_dirname = os.path.dirname(decoy_info_file)
                os.makedirs(decoy_dirname, exist_ok=True)

            if os.path.exists(decoy_info_file) and not args.overwrite:
                msg = f"Decoy file already exists for {decoy_id}. Skipping."
                msg += "Please use --overwrite to overwrite this file"
                print(msg, flush=True)
                continue

            print('Scoring decoys...')
            try:
                score_decoys = []
                for info in decoy_info:
                    info["score"] = score_fn(packed_pose.to_pose(info["pose"]))
                    if not args.save_pose:
                        #print('Deleting pose from decoy info')
                        del info["pose"]
                        for key in info['agent_keys']:
                            if 'pose' in info[key]:
                                del info[key]["pose"]
                    score_decoys.append(info["score"])
                score_decoys = np.asarray(score_decoys)

                print_statistics(score_decoys, prefix=f"{decoy_id}")

                if args.save_histograms:
                    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(8, 8))
                    ax[0].hist(score_decoys, bins=50)
                    ax[0].set_title(f"{decoy_id} Scores: Before normalization")
                    score_bound = decoy_info[0]["score"]
                    ax[0].axvline(x=score_bound, color="k")
                    
                    scores_mean = np.mean(score_decoys)
                    scores_std = np.std(score_decoys)
                    scores_normed = (score_decoys - scores_mean) / scores_std

                    ax[1].hist(scores_normed, bins=50)
                    ax[1].set_title(f"{decoy_id} Scores: After normalization")
                    score_bound_norm = (decoy_info[0]["score"] - scores_mean) / scores_std
                    ax[1].axvline(x=score_bound_norm, color="k")

                    fig.tight_layout()
                    if '/' in decoy_id:
                        idx = decoy_id.index('/')
                        fig.savefig(f"{histogram_dir}/{decoy_id[idx+1:]}_decoys={args.num_decoys}.png", dpi=400)
                    else:
                        fig.savefig(f"{histogram_dir}/{decoy_id}_decoys={args.num_decoys}.png", dpi=400)
                    plt.close()

                print(f"Saving decoy info to {decoy_info_file}", flush=True)
                if os.path.exists(decoy_info_file):
                    print(f"Removing existing file", flush=True)
                    os.remove(decoy_info_file)
                with open(f"{decoy_info_file}", "wb") as f:
                    pickle.dump(decoy_info, f)
                print(flush=True)

            except Exception as e:
                print(f"Scoring failed for {decoy_id} with {e}", flush=True)
                traceback.print_exc()


def parse_args_from_cmdline():
    parser = argparse.ArgumentParser()

    # Directory and setup args
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--dataset", default="db5", type=str)
    parser.add_argument("--complex_dir", default="complexes", type=str)
    parser.add_argument("--tmp_dir", default=None)
    parser.add_argument("--complex_list_file", default="complexes.txt", type=str)

    parser.add_argument("--agent_type", default="protein", type=str)
    
    # Decoy generation
    parser.add_argument("--decoy_type", default="rosetta", type=str)
    parser.add_argument("--score_fn_name", default="dock_low_res", type=str, help="score function name")
    parser.add_argument("--num_decoys", default=100, type=int, help="Number of decoys to generate per pdb id")
    parser.add_argument("--max_tr", default=10.0, type=float, help="Maximum translation magnitude for sampling decoys")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of workers used for decoy generation")
    parser.add_argument("--chunk_size", default=10, type=int, help="Chunk size to use multiprocessing with.")

    parser.add_argument("--save_histograms", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files with same name")
    parser.add_argument("--save_pose", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args_from_cmdline()

    # Load pdb ids
    base_dir = f"{args.data_dir}/raw/{args.dataset}"

    # Read complex ids from file
    full_complex_id_path = f"{base_dir}/{args.complex_list_file}"
    pdb_ids = read_complexes_from_txt(full_complex_id_path)

    if args.dataset == "db5":
        # Failing ids during stats computation for PyRosetta
        # TODO: See if they can be fixed
        failing_ids = ["2CFH", "4CPA", "3R9A", "1ZM4", "1TMQ", "3H11"]
        pdb_ids = list(set(pdb_ids).difference(set(failing_ids)))
    
    if not args.overwrite:
        pdb_ids = [pdb_id for pdb_id in pdb_ids 
                   if not os.path.exists(f"{base_dir}/decoys/{pdb_id}-decoys={args.num_decoys}.decoy")]

    print(f"Generating {args.num_decoys} decoys for {len(pdb_ids)} complexes.", flush=True)
    print(flush=True)

    max_tr = args.max_tr

    # Decoy generator and scoring
    decoy_gen_cls = get_decoy_generator(args.decoy_type)
    decoy_generator = decoy_gen_cls(max_tr=max_tr,
                                    num_decoys=args.num_decoys,
                                    agent_type=args.agent_type)
    score_fn = get_score_fn(args.score_fn_name)
    
    start_time = time.time()

    if args.num_workers > 1:
        chunk_size = args.chunk_size

        for i in range(len(pdb_ids) // chunk_size + 1):
            for mode in ["train", "val"]: # No test since we want to play game:
                
                pdb_ids_chunk = pdb_ids[chunk_size * i: chunk_size * (i+1)]
                p = Pool(processes=args.num_workers, maxtasksperchild=1)
                map_fn = p.imap_unordered

                decoy_generation_loop(args=args, mode=mode, pdb_ids=pdb_ids_chunk, map_fn=map_fn,
                                    decoy_generator=decoy_generator, score_fn=score_fn)

                p.__exit__(None, None, None)

    else:

        for mode in ["train", "val", "test"]:
            decoy_generation_loop(args=args, mode=mode, pdb_ids=pdb_ids, map_fn=map, 
                                decoy_generator=decoy_generator, score_fn=score_fn)

    end_time = time.time()
    time_taken = end_time - start_time
    time_taken_str = str(datetime.timedelta(seconds=end_time - start_time))
    avg_time_per_complex = time_taken / len(pdb_ids)
    avg_time_per_decoy = time_taken / (len(pdb_ids) * args.num_decoys)

    msg = f"Time taken: {time_taken_str}"
    msg += f" | Avg. Time per complex (in s): {avg_time_per_complex:.4f}"
    msg += f" | Avg. Time per decoy (in s): {avg_time_per_decoy:.4f}"

    print("Finished generating decoys for complexes", flush=True)
    print(msg, flush=True)
    
if __name__ == "__main__":
    main()
