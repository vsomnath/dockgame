import copy
import dataclasses
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn

from dockgame.common.structure import Protein, Structure
from dockgame.utils import geometry as geometry_ops

Tensor = torch.Tensor
Array = np.ndarray
ActionDict = dict[str, dict[str, Tensor]]


@dataclasses.dataclass
class Agent:

    name: str
    x: torch.Tensor = None
    edge_attr: torch.Tensor = None
    pos: torch.Tensor = None
    is_player: bool = False

    def __post_init__(self):
        if self.is_player:
            self.pos, self.rot_init, self.tr_init = self._randomize_pos(pos=self.pos)
        self.num_nodes = len(self.x)
        self.pos_init = self.pos.clone().detach()
        self.pos_ref = self.pos.clone().detach()

    def add_structure(self, structure: Structure):
        # These are the raw positions from structure
        base_positions = copy.deepcopy(structure.atom_positions)
        n_res, n_atoms, _ = base_positions.shape

        self.residue_types = copy.deepcopy(structure.residue_types)
        self.chain_ids = copy.deepcopy(structure.chain_ids)
        self.atom_mask = copy.deepcopy(structure.atom_mask)
        self.residue_index = copy.deepcopy(structure.residue_index)
        
        if self.is_player:
            # Gather c_alpha positions from orig_positions 
            ca_positions = base_positions[:, 1]

            # Apply any relevant mask to remove residues with no c_alpha_atoms
            ca_positions = ca_positions[self.atom_mask[:, 1].astype(bool)]
            ca_center = np.mean(ca_positions, axis=0, keepdims=True)

            # Updates the starting positions based on rot_init and tr_init
            # Note this is different from base positions since the agent positions
            # were randomized
            init_positions = geometry_ops.apply_rigid_transform_np(
                pos=base_positions.reshape(-1, 3), rot_vec=self.rot_init, 
                tr_vec=self.tr_init, center=ca_center
            )
            
            # At the start both atom positions and init positions are same
            # The atom positions are updated based on init_positions, rot_vec, tr_vec
            self.atom_positions = init_positions.reshape(n_res, n_atoms, -1)
            self.init_positions = init_positions.reshape(n_res, n_atoms, -1)
        else:
            self.atom_positions = copy.deepcopy(base_positions)

    def get_structure(self) -> Protein:
        return Protein(
            name=self.name,
            chain_ids=self.chain_ids,
            residue_types=self.residue_types,
            atom_positions=self.atom_positions,
            atom_mask=self.atom_mask,
            residue_index=self.residue_index
        )


@dataclasses.dataclass
class ScoreGameAgent(Agent):
    
    # Maximum translation noise used
    # This is used for sampling the random translation at the start of gameplay
    tr_sigma_max: float = 19.0

    def update_pose(self, action_dict: ActionDict):
        if not self.is_player:
            return
        
        assert action_dict is not None
        rot_vec = action_dict['rot_vec']
        tr_vec = action_dict['tr_vec']

        self.pos = geometry_ops.apply_rigid_transform(
            pos=self.pos, rot_vec=rot_vec, tr_vec=tr_vec
        )

        if hasattr(self, 'atom_positions'):
            n_res, n_atoms, _ = self.atom_positions.shape
            ca_positions = self.atom_positions[:, 1]
            ca_positions = ca_positions[self.atom_mask[:, 1].astype(bool)]
            ca_center = np.mean(ca_positions, axis=0, keepdims=True)

            atom_positions = self.atom_positions.reshape(-1, 3)

            atom_positions = \
                geometry_ops.apply_rigid_transform_np(
                    pos=atom_positions, 
                    rot_vec=rot_vec, tr_vec=tr_vec, center=ca_center
                )
            self.atom_positions = atom_positions.reshape(n_res, n_atoms, 3) 

    def _randomize_pos(self, pos: Tensor) -> Sequence[Tensor]:
        rot_vec = Rotation.random(num=1).as_rotvec()
        rot_vec = pos.new_tensor(rot_vec, dtype=torch.float)

        tr_vec = torch.normal(
            mean=0, std=self.tr_sigma_max, size=(1, 3), device=pos.device)

        randomized_pos = geometry_ops.apply_rigid_transform(
            pos=pos, rot_vec=rot_vec, tr_vec=tr_vec
        )
        return randomized_pos, rot_vec, tr_vec
    

@dataclasses.dataclass
class RewardGameAgent(Agent):
    
    # Reference position 
    pos_ref: torch.Tensor = None

    # Learning rate for the rotation action
    rot_lr: float = 1

    # Learning rate for the translation action
    tr_lr: float = 1

    # Maximum magnitude of translation
    max_tr: float = 35.0

    def _randomize_pos(self, pos: Tensor) -> Sequence[Tensor]:
        rot_vec = Rotation.random(num=1).as_rotvec()
        rot_vec = pos.new_tensor(rot_vec, dtype=torch.float, device=pos.device)

        def sample_translation(max_length):
            tr = np.random.randn(1, 3)
            tr = tr / np.sqrt( np.sum(tr * tr))
            tr_len = np.random.uniform(low=0.0, high=max_length)
            tr = tr * tr_len
            assert np.allclose(np.linalg.norm(tr), tr_len, atol=1e-5)
            tr = tr.squeeze()
            return tr

        #tr_vec_np = sample_translation(max_length=self.max_tr)
        #tr_vec = pos.new_tensor(tr_vec_np)
        
        tr_vec = torch.normal(
            mean=0, std=self.max_tr, size=(1, 3), device=pos.device)
        
        randomized_pos = geometry_ops.apply_rigid_transform(
            pos=pos, rot_vec=rot_vec, tr_vec=tr_vec
        )

        # Initialize self.rot_vec and self.tr_vec
        self.rot_vec = rot_vec
        self.tr_vec = tr_vec
        return randomized_pos, rot_vec, tr_vec

    def update_pose(self, action_dict: ActionDict = None):
        if not self.is_player:
            return

        if action_dict is not None:
            grad_rot, grad_tr = action_dict['rot_vec'], action_dict['tr_vec']
            self.rot_vec.data = self.rot_vec.data - self.rot_lr * grad_rot
            self.tr_vec.data = self.tr_vec.data - self.tr_lr * grad_tr

        self.pos = geometry_ops.apply_rigid_transform(
            pos=self.pos_init,
            rot_vec=self.rot_vec,
            tr_vec=self.tr_vec
        )

        if hasattr(self, 'atom_positions'):
            ca_positions_init = self.init_positions[:, 1]
            ca_positions_init = ca_positions_init[self.atom_mask[:, 1].astype(bool)]
            ca_center = np.mean(ca_positions_init, axis=0, keepdims=True)

            n_res, n_atoms, _ = self.atom_positions.shape
            atom_positions = \
                  geometry_ops.apply_rigid_transform_np(
                pos=self.init_positions.reshape(-1, 3), 
                rot_vec=self.rot_vec, tr_vec=self.tr_vec, 
                center=ca_center
            )
            self.atom_positions = atom_positions.reshape(n_res, n_atoms, -1)


def get_agent_cls(cls_name: str):
    if cls_name == "score":
        return ScoreGameAgent
    elif cls_name == "reward":
        return RewardGameAgent
    else:
        raise ValueError(f"Agent cls of type {cls_name} is not supported")
