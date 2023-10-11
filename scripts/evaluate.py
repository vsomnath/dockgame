import os
import json
import argparse
from collections import defaultdict
from typing import Callable

import numpy as np
from tqdm import tqdm

from dockgame.data.parser import ProteinParser
from dockgame.common.structure import Protein
from dockgame.utils.setup import read_complexes_from_txt

from dockgame.analysis.metrics import (
    compute_complex_rmsd, 
    compute_pyrosetta_scores, get_score_fn, 
    compute_tm_score
)

Args = argparse.Namespace


def gather_filenames_gt(complex_id: str, args: Args) -> dict[str, str]:
    gt_complex_dir = (
        f'{args.data_dir}/raw/{args.dataset}/{args.complex_dir}'
    )

    if args.dataset == 'db5':
        filenames = {
            'ligand': f'{gt_complex_dir}/{complex_id}_l_b.pdb',
            'receptor': f'{gt_complex_dir}/{complex_id}_r_b.pdb'
        }
    
    elif args.dataset == 'multimer':
        filenames = {
            complex_id: f'{gt_complex_dir}/{complex_id}-assembly1.cif'
        }

    return filenames


def parse_evaluation_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--complex_dir", type=str)
    parser.add_argument("--complex_list_file", type=str)

    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--score_fn_name", type=str, default=None)

    return parser.parse_args()


def evaluate(args):

    complex_list = read_complexes_from_txt(
        filename=f'{args.data_dir}/raw/{args.dataset}/{args.complex_list_file}'
    )
    print(f'Evaluating for num complexes: {len(complex_list)}', flush=True)

    experiment_dir = f'{args.results_dir}/{args.task_name}'

    metrics_all = {}
    prot_parser = ProteinParser()

    if args.score_fn_name is None:
        score_fn = None
    else:
        try:
            score_fn = get_score_fn(args.score_fn_name)
        except Exception as e:
            score_fn = None

    for complex_id in tqdm(complex_list):
        if complex_id == "1N2C":
            continue

        pdb_metrics = defaultdict(list)

        complex_save_dir = f'{experiment_dir}/{complex_id}'
        if not os.path.exists(f'{experiment_dir}/{complex_id}'):
            print(f'Skipping {complex_id} because directory not found.')

        gt_filenames = gather_filenames_gt(complex_id=complex_id, args=args)
        structures_true = prot_parser.parse(gt_filenames)
        structure_true = Protein.from_protein_list(structures=structures_true)

        pdb_files = os.listdir(complex_save_dir)
        pdb_files = [filename for filename in pdb_files 
                     if "init" not in filename and ".pdb" in filename]
        
        structures_all = []
        for pdb_file in tqdm(pdb_files):
            try:
                structures_pred = prot_parser.parse(
                    {complex_id: f'{experiment_dir}/{complex_id}/{pdb_file}'}
                )

                if len(structures_pred) == 1:
                    structure_pred = structures_pred[0]
                else:
                    # Join the structures into a single structure
                    structure_pred = Protein.from_protein_list(structures_pred)

                if score_fn is not None:
                    pyrosetta_score_pred = compute_pyrosetta_scores(
                        filenames={'joint': f'{experiment_dir}/{complex_id}/{pdb_file}'},
                        score_fn=score_fn,
                        agent_keys=['ligand', 'receptor']
                    )
                    pyrosetta_score_bound = compute_pyrosetta_scores(
                        filenames=gt_filenames,
                        score_fn=score_fn,
                        agent_keys=['ligand', 'receptor']
                    )

                    pdb_metrics['pyrosetta_score_pred'].append(pyrosetta_score_pred)
                    pdb_metrics['pyrosetta_score_bound'].append(pyrosetta_score_bound)
                
                structures_all.append(structure_pred)
            except Exception as e:
                print(f"Failed to process {pdb_file} due to {e}")
                pass

        for structure in structures_all:
            c_alpha_mask_pred = structure.atom_mask[:, 1].astype(bool)
            c_alpha_pred = structure.atom_positions[:, 1]
            c_alpha_pred = c_alpha_pred[c_alpha_mask_pred]

            c_alpha_mask_true = structure_true.atom_mask[:, 1].astype(bool)
            c_alpha_true = structure_true.atom_positions[:, 1]
            c_alpha_true = c_alpha_true[c_alpha_mask_true]

            try:
                complex_rmsd, _ = compute_complex_rmsd(
                    complex_pred=c_alpha_pred, complex_true=c_alpha_true
                )
                pdb_metrics['complex_rmsd'].append(complex_rmsd)
            except Exception as e:
                continue
            
            try:
                tm_score = compute_tm_score(
                    c_alpha_pred=c_alpha_pred, c_alpha_true=c_alpha_true,
                    restypes_pred=structure_pred.residue_types,
                    restypes_true=structure_true.residue_types
                )
                pdb_metrics['tm_score'].append(tm_score)
            except Exception as e:
                continue
        
        if 'pairwise_tm' in args and args.pairwise_tm:
            for idx in range(len(structures_all)):
                for inner_idx in range(idx + 1, len(structures_all)):
                    structure_idx = structures_all[idx]
                    structure_inner_idx = structures_all[inner_idx]

                    c_alpha_mask_pred = structure_idx.atom_mask[:, 1].astype(bool)
                    c_alpha_pred = structure_idx.atom_positions[:, 1]
                    c_alpha_pred = c_alpha_pred[c_alpha_mask_pred]

                    c_alpha_mask_true = structure_inner_idx.atom_mask[:, 1].astype(bool)
                    c_alpha_true = structure_inner_idx.atom_positions[:, 1]
                    c_alpha_true = c_alpha_true[c_alpha_mask_true]

                    try:
                        tm_score = compute_tm_score(
                            c_alpha_pred=c_alpha_pred, c_alpha_true=c_alpha_true,
                            restypes_pred=structure_idx.residue_types,
                            restypes_true=structure_inner_idx.residue_types
                        )
                        pdb_metrics['pairwise_tm_score'].append(tm_score)
                    except Exception as e:
                        continue

        metrics_all[f'{complex_id}'] = pdb_metrics


    with open(f"{experiment_dir}/results.json", 'w') as f:
        json.dump(metrics_all, f)


def main():

    args = parse_evaluation_args()
    evaluate(args=args)


if __name__ == "__main__":
    main()
