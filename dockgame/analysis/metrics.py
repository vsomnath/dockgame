import os
import numpy as np
import torch
from typing import Union, Callable

import dockgame.common.constants as constants
from dockgame.utils import geometry as geometry_ops
try:
    import pyrosetta
    pyrosetta.init(' '.join([
     '-mute', 'all',
     '-use_input_sc',
    '-ignore_unrecognized_res',
    '-ignore_zero_occupancy', 'false',
    '-load_PDB_components', 'false',
    '-relax:default_repeats', '2',
    '-no_fconfig'
    ]))
    from pyrosetta.rosetta.protocols.fold_from_loops.utils import append_pose_to_pose_keep_fold_tree
    from pyrosetta.rosetta.protocols.toolbox.pose_manipulation import remove_non_protein_residues
except Exception as e:
    pass

try:
    import tmtools
except Exception as e:
    pass


Tensor = torch.Tensor
Array = np.ndarray
ArrayOrTensor = Union[Tensor, Array]
RmsdOutput = tuple[ArrayOrTensor, dict[str, float]]

# ==============================================================================
# RMSD Metrics
# ==============================================================================


def rmsd(y_pred: ArrayOrTensor, y_true: ArrayOrTensor) -> ArrayOrTensor:
    se = (y_pred - y_true)**2
    try:
        mse = se.sum(dim=1).mean()
        return torch.sqrt(mse)
    except Exception:
        mse = se.sum(axis=1).mean()
        return np.sqrt(mse)


def compute_complex_rmsd_torch(
    complex_pred: Tensor, 
    complex_true: Tensor
) -> RmsdOutput:
    rot_mat, tr = geometry_ops.rigid_transform_kabsch_3D_torch(
        complex_pred.T, complex_true.T
    )
    complex_pred_aligned = ( (rot_mat @ complex_pred.T) + tr ).T

    complex_rmsd = rmsd(complex_pred_aligned, complex_true)

    return complex_rmsd, {
        "complex_rmsd": np.round(complex_rmsd.item(), 4),
    }


def compute_complex_rmsd(
    complex_pred: Array, 
    complex_true: Array
) -> RmsdOutput:
    rot_mat, tr = geometry_ops.rigid_transform_kabsch_3D(
        complex_pred.T, complex_true.T
    )
    complex_pred_aligned = ( (rot_mat @ complex_pred.T) + tr ).T

    complex_rmsd = rmsd(complex_pred_aligned, complex_true)

    return complex_rmsd, {
        "complex_rmsd": np.round(complex_rmsd.item(), 4),
    }


# ==============================================================================
# PyRosetta Specific Metrics (Energetics)
# ==============================================================================


def get_score_fn(score_fn_name: str) -> Callable:
    if score_fn_name == "dock_high_res":
        score_fn = pyrosetta.create_score_function("ref2015.wts", "docking")
    elif score_fn_name == "dock_low_res":
        score_fn = pyrosetta.create_score_function("interchain_cen", "docking")
    else:
        msg = f"Score function with name: {score_fn_name} is currently not supported."
        raise ValueError(msg)
    return score_fn


def compute_pyrosetta_scores(
    filenames: dict[str, str],
    score_fn: Callable,
    agent_keys: list[str]
) -> float:
    
    try:
        if len(filenames) == 1:
            pyrosetta_fn = compute_pyrosetta_scores_joint
        elif len(filenames) > 1:
            pyrosetta_fn = compute_pyrosetta_scores_individual
        else:
            raise ValueError(f"Pyrosetta computation is missing because no filenames.")
        return pyrosetta_fn(
            filenames=filenames, score_fn=score_fn, agent_keys=agent_keys
        )
    except Exception as e:
        return None


def compute_pyrosetta_scores_individual(
    filenames: dict[str, str], 
    score_fn: Callable,
    agent_keys: list[str]
) -> float:
    poses = {
        agent_key: pyrosetta.pose_from_file(filenames[agent_key])
        for agent_key in agent_keys
    }

    complex_pose = poses[agent_keys[0]].clone()
    remove_non_protein_residues(complex_pose)

    for agent_key in agent_keys[1:]:
        cloned_pose_agent = poses[agent_key].clone()
        remove_non_protein_residues(cloned_pose_agent)
        append_pose_to_pose_keep_fold_tree(
            complex_pose, cloned_pose_agent, new_chain=True
        )
    remove_non_protein_residues(complex_pose)
    return score_fn(complex_pose)


def compute_pyrosetta_scores_joint(
    filenames: dict[str, str], 
    score_fn: Callable, 
    agent_keys: list[str] = None
) -> float:
    filename = list(filenames.values())[0]
    complex_pose = pyrosetta.pose_from_file(os.path.abspath(filename))
    remove_non_protein_residues(complex_pose)
    return score_fn(complex_pose)


# ==============================================================================
# Radius of Gyration specific metrics
# ==============================================================================


def compute_radius_of_gyration(coords: Array, atoms: list[str]) -> float:
    MASS_DICT = {
    'C': 12.0107,
    "O": 15.9994,
    "N": 14.0067,
    "S": 32.065
}
    if isinstance(coords, list):
        coords = np.asarray(coords)

    masses = np.asarray([MASS_DICT[atom_type] for atom_type in atoms])[:, None]
    x_times_m = coords * masses
    total_mass = np.sum(masses)
    rr = np.sum(coords * x_times_m)

    mm = np.sum( (np.sum(x_times_m, axis=0) / total_mass)**2 )
    rg = np.sqrt(rr / total_mass - mm)

    return rg


# def compute_radius_of_gyration(protein_files):
#     atoms = []
#     coords = []
#     for key in ["ligand", "receptor"]:
#         protein_file = protein_files[key]
#         df = _parse_pdb_biopandas(protein_file)
#         coords.extend(df[['x_coord', 'y_coord', 'z_coord']].values)
#         atoms.extend([a[0] for a in df['atom_name'].to_list()])
#     assert len(coords) == len(atoms), "Number of atoms and atom types do not match"
#     return compute_radius_of_gyration(coords, atoms)


# ==============================================================================
# TM-Score
# ==============================================================================


def compute_tm_score(
    c_alpha_pred: Array,
    c_alpha_true: Array,
    restypes_pred: Array,
    restypes_true: Array
) -> float:
    pred_seq = "".join(
        constants.restype_3to1[str(elem)] for elem in restypes_pred.tolist()
    )
    true_seq = "".join(
        constants.restype_3to1[str(elem)] for elem in restypes_true.tolist()
    )

    tm_results = tmtools.tm_align(
        c_alpha_pred, c_alpha_true, pred_seq, true_seq
    )
    tm_score, _ = tm_results.tm_norm_chain1, tm_results.tm_norm_chain2
    return tm_score
