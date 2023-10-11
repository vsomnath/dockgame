import os
from typing import Callable
import string
import traceback

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
    from pyrosetta.bindings import pose as pose_utils
    from pyrosetta.toolbox.extract_coords_pose import pose_coords_as_rows
    from pyrosetta.distributed import packed_pose

    import numpy as np
    from scipy.spatial.transform import Rotation
    from dockgame.data.tools.decoys.base import DecoyGenerator
    # from dockgame.data.parser import _parse_pdb_biopandas
    from dockgame.analysis.metrics import compute_radius_of_gyration

except Exception as e:
    import numpy as np
    from scipy.spatial.transform import Rotation
    from dockgame.data.tools.decoys.base import DecoyGenerator
    # from dockgame.data.parser import _parse_pdb_biopandas
    from dockgame.analysis.metrics import compute_radius_of_gyration


def get_score_fn(score_fn_name: str) -> Callable:
    if score_fn_name == "dock_high_res":
        score_fn = pyrosetta.create_score_function("ref2015.wts", "docking")
    elif score_fn_name == "dock_low_res":
        score_fn = pyrosetta.create_score_function("interchain_cen", "docking")
    else:
        msg = f"Score function with name: {score_fn_name} is currently not supported."
        raise ValueError(msg)
    return score_fn


def _sample_rigid_transform(max_length: int = None):

    def sample_translation(max_tr: float) -> np.ndarray:
        tr = np.random.randn(1, 3)
        tr = tr / np.sqrt( np.sum(tr * tr))
        tr_len = np.random.uniform(low=0.0, high=max_tr)
        tr = tr * tr_len
        assert np.allclose(np.linalg.norm(tr), tr_len, atol=1e-5)
        return tr
    
    def sample_rotation():
        R = Rotation.random(num=1)
        return R.as_rotvec()

    tr_vec = sample_translation(max_tr=max_length)
    rot_vec = sample_rotation()
    return rot_vec, tr_vec


def _modify_pyrosetta_pose(pose, rot_mat, tr_vec):
    # We assume that pose is already centered
    # tr_vec contains the pose center to be added back
    new_pose = pose.clone()
    remove_non_protein_residues(new_pose)
    pose_utils.rotate(new_pose, rot_mat) # R.(x - x_com) 
    pose_utils.translate(new_pose, tr_vec) # R.(x - x_com) + x_com + tr
    return new_pose


def _individual_to_joint_pose(poses, keys):
    complex_pose = poses[keys[0]].clone()
    remove_non_protein_residues(complex_pose)

    for agent_key in keys[1:]:
        cloned_pose_agent = poses[agent_key].clone()
        remove_non_protein_residues(cloned_pose_agent)
        append_pose_to_pose_keep_fold_tree(
            complex_pose, cloned_pose_agent, new_chain=True
        )
    remove_non_protein_residues(complex_pose)
    return complex_pose


def _modify_and_join_poses(agent_keys, poses, centers, decoy_info):
    updated_poses = {}
    for key in agent_keys:
        rot_vec = decoy_info[key]['rot_vec']
        tr_vec = decoy_info[key]['tr_vec']
        rot_mat = Rotation.from_rotvec(rot_vec).as_matrix().squeeze()

        updated_pose = _modify_pyrosetta_pose(
            pose=poses[key], rot_mat=rot_mat,
            tr_vec=tr_vec.squeeze() + centers[key]
        )
        updated_poses[key] = updated_pose

    complex_pose = _individual_to_joint_pose(poses=updated_poses, keys=agent_keys)
    return complex_pose


class RosettaDecoys(DecoyGenerator):

    def __init__(self, max_tr: int = None, **kwargs):
        super().__init__(**kwargs)
        self.max_tr = max_tr if max_tr is not None else 10.0

    def sample_translation(self, max_length: int = None) -> np.ndarray:
        if max_length is None:
            max_length = self.max_tr
        if max_length < 1e-6:
            return np.zeros((1, 3))
        tr = np.random.randn(1, 3)
        tr = tr / np.sqrt( np.sum(tr * tr))
        tr_len = np.random.uniform(low=0.0, high=max_length)
        tr = tr * tr_len
        assert np.allclose(np.linalg.norm(tr), tr_len, atol=1e-5)
        return tr

    def sample_rotation(self) -> np.ndarray:
        R = Rotation.random(num=1)
        return R.as_rotvec()

    def sample_rigid_transform(self, max_length: int = None):
        R = self.sample_rotation()
        tr = self.sample_translation(max_length=max_length)
        return R, tr
    
    def modify_pose(self, pose, rot_mat, tr_vec):
        # We assume that pose is already centered
        # tr_vec contains the pose center to be added back
        new_pose = pose.clone()
        remove_non_protein_residues(new_pose)
        pose_utils.rotate(new_pose, rot_mat) # R.(x - x_com) 
        pose_utils.translate(new_pose, tr_vec) # R.(x - x_com) + x_com + tr
        return new_pose

    def individual_to_joint_pose(self, lig_pose, rec_pose):
        complex_pose = lig_pose.clone()
        rec_pose_clone = rec_pose.clone()

        remove_non_protein_residues(complex_pose)
        remove_non_protein_residues(rec_pose_clone)

        append_pose_to_pose_keep_fold_tree(complex_pose, rec_pose_clone, new_chain=True)
        remove_non_protein_residues(complex_pose)
        return complex_pose

    def get_coords_and_atoms(self, pdb_file):
        df = _parse_pdb_biopandas(pdb_file)
        coords = df[['x_coord', 'y_coord', 'z_coord']].values
        atoms = df['element_symbol'].to_list()
        return coords, atoms

    def apply_rigid_transform(self, pos, rot_mat, tr_vec):
        center = np.mean(pos, axis=0, keepdims=True)

        if len(tr_vec.shape) == 1:
            tr_vec = tr_vec[None, :]

        rigid_new_pos = (pos - center) @ rot_mat.T + center + tr_vec
        return rigid_new_pos
    
    def generate_decoys(self, inputs):
        pdb_id, filenames = inputs

        try:
            decoy_infos = []
            poses_base = {
                key: pyrosetta.pose_from_file(os.path.abspath(filename))
                for key, filename in filenames.items()
            }

            if self.agent_type == 'chain':
                poses = {}
                agent_keys = []

                for key, pose in poses_base.items():
                    for chain_id in range(1, pose.num_chains() + 1):
                        # Switching between 0 index and 1 index
                        chain_id_str = string.ascii_uppercase[chain_id-1]
                        poses[f"{key}_{chain_id_str}"] = pose.split_by_chain(chain_id).clone()
                        agent_keys.append(f"{key}_{chain_id_str}")
            else:
                poses = poses_base
                agent_keys = list(sorted(poses.keys()))

            centers = {}
            for key in agent_keys:
                pose = poses[key]
                pose_coords = pose_coords_as_rows(pose)
                pose_center = np.mean(pose_coords, axis=0)
                pose.center() # This is inplace

                centers[key] = pose_center

            print(f'Started decoy generation for {pdb_id}', flush=True)
            for decoy_idx in range(self.num_decoys):
                decoy_info = {'id': pdb_id}
                #print(f'Sampling rigid transforms..', flush=True)
                for key in agent_keys[:-1]:
                    if decoy_idx == 0:
                        rot_vec = np.zeros((1, 3))
                        tr_vec = np.zeros((1, 3))
                    else:
                        rot_vec, tr_vec \
                            = _sample_rigid_transform(max_length=self.max_tr)
                        
                    decoy_info_agent = {
                        'rot_vec': rot_vec,
                        'tr_vec': tr_vec
                    }
                    decoy_info[key] = decoy_info_agent
                
                last_key = agent_keys[-1]
                decoy_info[last_key] = {
                    'rot_vec': np.zeros((1, 3)),
                    'tr_vec': np.zeros((1, 3))
                }
                #print(f'Joining poses..', flush=True)
                complex_pose = _modify_and_join_poses(
                    agent_keys=agent_keys, poses=poses, centers=centers,
                    decoy_info=decoy_info
                )
                #print(f'To packed pose..', flush=True)
                decoy_info['pose'] = packed_pose.to_packed(complex_pose)
                decoy_info['agent_keys'] = agent_keys
                decoy_infos.append(decoy_info)
            print(f'End of decoy generation for {pdb_id}', flush=True)
            return decoy_infos

        except Exception as e:
            print(f"Decoy generation failed for {pdb_id} with {e}", flush=True)
            traceback.print_exc()
            return None

    def generate_decoys_old(self, inputs):
        pdb_id, lig_pdb_file, rec_pdb_file = inputs
        
        # TODO: Fix this
        coords_lig, atoms_lig = self.get_coords_and_atoms(lig_pdb_file)
        coords_rec, atoms_rec = self.get_coords_and_atoms(rec_pdb_file)
        atoms_complex  = atoms_lig.copy()
        atoms_complex.extend(atoms_rec)
        
        try:
            decoy_infos = []
            lig_pose = pyrosetta.pose_from_pdb(os.path.abspath(lig_pdb_file))
            rec_pose = pyrosetta.pose_from_pdb(os.path.abspath(rec_pdb_file))

            # Centering the ligand and saving ligand center
            # We apply the rotation about the center of the ligand
            lig_coords = pose_coords_as_rows(lig_pose) #TODO: check why this is different from coords_lig 
            lig_center = np.mean(lig_coords, axis=0)
            lig_pose.center()

            for decoy_idx in range(self.num_decoys):
                if decoy_idx == 0:
                    # Ground truth bound decoy
                    rot_vec = np.zeros((1, 3))
                    tr_vec = np.zeros((1, 3))
                else:
                    rot_vec, tr_vec = self.sample_rigid_transform(max_length=self.max_tr)
                
                rot_mat = Rotation.from_rotvec(rot_vec).as_matrix().squeeze()
                decoy_pose = self.modify_pose(
                        pose=lig_pose, rot_mat=rot_mat, 
                        tr_vec=tr_vec.squeeze() + lig_center
                    )
                
                coords_lig = self.apply_rigid_transform(
                pos=coords_lig, rot_mat=rot_mat, tr_vec=tr_vec)

                ligand_info = dict(
                    rot_vec=rot_vec,
                    tr_vec=tr_vec,
                    pose=lig_pose.clone(),
                    radius_of_gyration=compute_radius_of_gyration(coords_lig, atoms_lig)
                )

                rec_info = dict(
                    rot_vec=np.zeros((1, 3)),
                    tr_vec=np.zeros((1, 3)),
                    pose=rec_pose.clone(),
                    radius_of_gyration=compute_radius_of_gyration(coords_rec, atoms_rec)
                )


                # Create the new complex pose by joining ligand and receptor poses
                complex_pose = self.individual_to_joint_pose(lig_pose=decoy_pose,
                                                            rec_pose=rec_pose)
                
                coords_decoy = np.concatenate([coords_lig, coords_rec], axis=0)
                
                #score = self.score_fn(complex_pose)
                decoy_info = dict(
                    id=pdb_id,
                    ligand=ligand_info,
                    receptor=rec_info,
                    pose=packed_pose.to_packed(complex_pose), # Converts this into a serializable format
                    radius_of_gyration=compute_radius_of_gyration(coords_decoy, atoms_complex),
                )

                decoy_infos.append(decoy_info)

            return decoy_infos

        except Exception as e:
            print(f"Decoy generation failed for {pdb_id} with {e}")
            return None
