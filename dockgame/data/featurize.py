import argparse
import dataclasses
from typing import Sequence, Tuple

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from dockgame.common import constants, structure
from dockgame.utils import ops


ALLOWED_AGENT_TYPES = ["protein", "chain"]

RESIDUE_NAME_FEATS = 145
DISTANCE_FEATS = 120 #TODO: This is with rbf=20
DIHEDRAL_FEATS = 6
ANGLE_FEATS = 6

FEATURE_DIMENSIONS = {
    'base': RESIDUE_NAME_FEATS,
    'pifold': RESIDUE_NAME_FEATS + DISTANCE_FEATS + DIHEDRAL_FEATS + ANGLE_FEATS
}

# Type aliases
Tensor = torch.Tensor
Array = np.ndarray
Structure = structure.Structure


@dataclasses.dataclass
class ProteinFeaturizer:

    # Name of the featurizer
    name: str = 'base'

    # Type of agents used in gameplay - Possible choices include chain / protein
    agent_type: str = 'protein'

    # Granularity of protein structure utilized in features
    # c_alpha: uses C-\alpha atom coordinates of residues
    # backbone: uses backbone atom coordinates of residues
    # atom: uses all atom coordinates
    resolution: str = 'c_alpha'

    def __post_init__(self):
        if self.agent_type not in ALLOWED_AGENT_TYPES:
            raise ValueError(f'Provided agent type {self.agent_type} not in'
                             f'allowed list {ALLOWED_AGENT_TYPES}')

    def featurize(self, 
                  structures: Sequence[structure.Protein],
                  graph: HeteroData) -> HeteroData:
        
        agent_keys = []

        for structure in structures:
            if self.agent_type == 'chain':
                for chain in structure.get_chains():
                    agent_key = chain.protein_name + "_" + str(chain.chain_id).strip()
                    agent_pos, mask = self.get_coords(chain.atom_positions, chain.atom_mask)
                    x = self.featurize_residues(structure=chain, mask=mask)

                    graph[agent_key].x = x
                    graph[agent_key].pos = torch.tensor(agent_pos).float()
                    graph[agent_key].pos_bound = torch.tensor(agent_pos).float()
                    agent_keys.append(agent_key)

            else:
                agent_key = structure.name
                agent_pos, mask = self.get_coords(structure.atom_positions, structure.atom_mask)
                x = self.featurize_residues(structure=structure, mask=mask)

                graph[agent_key].x = x
                graph[agent_key].pos = torch.tensor(agent_pos).float()
                graph[agent_key].pos_bound = torch.tensor(agent_pos).float()
                agent_keys.append(agent_key)
          
        graph.agent_keys = agent_keys
        graph.protein_keys = ["ligand", "receptor"]
        return graph
    
    def get_coords(self, 
                   atom_positions: Array, 
                   atom_mask: Array, 
                   resolution: str = None) -> Tuple[Array, Array]:
        if resolution is None:
            resolution = self.resolution

        if resolution == 'c_alpha':
            ca_atom_mask = atom_mask[:, 1].astype(bool)
            ca_positions = atom_positions[:, 1]
            ca_positions_subset = ca_positions[ca_atom_mask]
            return ca_positions_subset, ca_atom_mask
        elif resolution == 'backbone':
            return atom_positions[:, :3], None
        elif resolution == 'atom':
            return atom_positions, None
        else:
            raise ValueError(f"Resolution {resolution} not supported")
        
    def featurize_residues(self, structure, mask=None) -> Tensor:
        if mask is not None:
            res_types = structure.residue_types[mask].tolist()
        else:
            res_types = structure.residue_types.tolist()
        return compute_residue_name_feats(res_types=res_types)


# ==============================================================================
# PIFOLD featurizer (Taken from https://arxiv.org/abs/2209.12643)
# ==============================================================================


@dataclasses.dataclass
class PiFoldFeaturizer(ProteinFeaturizer):
    """PiFold featurizer based on https://arxiv.org/abs/2209.12643."""

    name: str = "pifold"

    # Number of centers to use in RBF computation
    num_rbf: int = 20
    
    def featurize_residues(self, 
                           structure: Structure, 
                           mask: Array = None) -> Tensor:
        atom_pos = ops.make_float_tensor(structure.atom_positions)
        atom_mask = ops.make_float_tensor(structure.atom_mask)

        res_feats = compute_residue_name_feats(res_types=structure.residue_types)
        angle_feats = compute_angle_feats(
            atom_pos=atom_pos, atom_mask=atom_mask, eps=1e-7
        )
        distance_feats = compute_distance_feats(
            atom_pos=atom_pos, atom_mask=atom_mask, num_rbf=self.num_rbf
        )

        dihedral_feats = compute_dihedral_feats(
            atom_pos=atom_pos, atom_mask=atom_mask, debug=False, eps=1e-7
        )

        node_features = torch.cat([
            res_feats, angle_feats, dihedral_feats, distance_feats
        ], dim=1)

        return node_features
    

def compute_residue_name_feats(res_types: list[str]) -> Tensor:
    residue_feats = []
    for res_name in res_types:
        residue_feat = \
                ops.onek_encoding_unk(res_name, constants.amino_acid_types) + \
                [
                ops.index_with_unk(constants.kd_scale, res_name),
                ops.index_with_unk(constants.volumes, res_name),
                ops.index_with_unk(constants.charges, res_name),
                ops.index_with_unk(constants.polarity, res_name),
                ops.index_with_unk(constants.acceptor, res_name),
                ops.index_with_unk(constants.donor, res_name),
            ]
        residue_feats.append(residue_feat)

    n_aa_types = len(constants.amino_acid_types)

    residue_feats = torch.tensor(residue_feats)
    hydrophobicity = residue_feats[:, n_aa_types]
    volumes = residue_feats[:, n_aa_types + 1] / 100.0
    charges = residue_feats[:, n_aa_types + 2]
    polarity_hbonds = residue_feats[:, n_aa_types + 2 :]

    # Expand components into gaussian basis
    # Taken from https://github.com/wengong-jin/abdockgen
    residue_feats = torch.cat(
        [
            residue_feats[:, :n_aa_types],
            ops.rbf_basis(hydrophobicity, -4.5, 4.5, 0.1),
            ops.rbf_basis(volumes, 0, 2.2, 0.1),
            ops.rbf_basis(charges, -1.0, 1.0, 0.25),
            torch.sigmoid(polarity_hbonds * 6 - 3),
        ],
        dim=-1,
    )
    return residue_feats


def compute_angle_feats(
    atom_pos: Tensor, 
    atom_mask: Tensor, 
    eps=1e-7
) -> Tensor:
    """Computes bond angle features for a given chain/protein."""
    n_residues, _, _ = atom_pos.shape

    # For bond angles, we only use N, CA, C [0, 1, 2]
    bb_atom_pos = atom_pos[:, :3].reshape(n_residues * 3, 3)
    bb_mask = atom_mask[:, :3].reshape(n_residues * 3)
    # Gather difference vectors: CA_i-N_i, C_i-CA_i, N_{i+1}-C_i
    bond_vectors = bb_atom_pos[1:] - bb_atom_pos[:-1]
    # Bond is present only if both atoms are present
    bond_mask = bb_mask[1:] * bb_mask[:-1]
    bond_vectors = F.normalize(bond_vectors, dim=-1)

    # Angle can be computed only if both bonds are present
    angle_mask = bond_mask[:-2] * bond_mask[1:-1]
    cos_bond_angles = -(bond_vectors[:-2] * bond_vectors[1:-1]).sum(dim=-1)
    cos_bond_angles = torch.clamp(cos_bond_angles, -1 + eps, 1-eps) * angle_mask
    bond_angles = torch.acos(cos_bond_angles)
    bond_angles_padded = F.pad(bond_angles, (1, 2), 'constant', 0)
    bond_angles_final = bond_angles_padded.view(n_residues, 3)

    angle_feats = torch.cat(
        [torch.cos(bond_angles_final), torch.sin(bond_angles_final)], dim=1
    )

    return angle_feats


def compute_dihedral_feats(
    atom_pos: Tensor, 
    atom_mask: Tensor, 
    eps=1e-7, 
    debug: bool = False
) -> Tensor:
    n_residues, _, _ = atom_pos.shape

    # For dihedral angles, we only use N, CA, C [0, 1, 2]
    bb_atom_pos = atom_pos[:, :3].reshape(n_residues * 3, 3)
    bb_mask = atom_mask[:, :3].reshape(n_residues * 3)
    # Gather difference vectors: CA_i-N_i, C_i-CA_i, N_{i+1}-C_i
    bond_vectors = bb_atom_pos[1:] - bb_atom_pos[:-1]
    # Bond is present only if both atoms are present
    bond_mask = bb_mask[1:] * bb_mask[:-1]
    bond_vectors = F.normalize(bond_vectors, dim=-1)

    u_0, u_0_mask = bond_vectors[:-2], bond_mask[:-2]
    u_1, u_1_mask = bond_vectors[1:-1], bond_mask[1:-1]
    u_2, u_2_mask = bond_vectors[2:], bond_mask[2:]

    n_0 = F.normalize(torch.cross(u_0, u_1), dim=-1) * (u_0_mask * u_1_mask).unsqueeze(1)
    n_1 = F.normalize(torch.cross(u_1, u_2), dim=-1) * (u_1_mask * u_2_mask).unsqueeze(1)
    n_0_mask = u_0_mask * u_1_mask
    n_1_mask = u_1_mask * u_2_mask

    # TODO: Check masking
    cos_dihedrals = (n_0 * n_1).sum(dim=-1)
    cos_dihedrals = torch.clamp(cos_dihedrals, -1+eps, 1-eps)
    v = F.normalize(torch.cross(n_0, n_1), dim=-1)

    dihedrals = torch.sign((v * u_1).sum(dim=-1)) * torch.acos(cos_dihedrals)
    dihedrals = dihedrals * n_0_mask * n_1_mask

    dihedrals_padded = F.pad(dihedrals, (1, 2), 'constant', 0)
    dihedral_angles_final = dihedrals_padded.view(n_residues, 3)
    dihedral_feats = torch.cat(
        [torch.cos(dihedral_angles_final), 
            torch.sin(dihedral_angles_final)], dim=1
    )

    if debug:
        phi, psi, omega = torch.unbind(dihedral_angles_final, dim=-1)
        return dihedral_feats, (phi, psi, omega)

    return dihedral_feats


def compute_distance_feats(
    atom_pos: Tensor, 
    atom_mask: Tensor, 
    num_rbf: int = 20
) -> Tensor:
    """
    Computes distance features for a given chain/protein.
    
    These distances correspond to pairwise distances between backbone atoms
    N, CA, C, (maybe O) for each residue.
    """
    n_residues, _, _ = atom_pos.shape
    # We use N, CA, C, O for distances
    bb_atoms = atom_pos[:, :4]
    bb_mask = atom_mask[:, :4]
    distance_vec = bb_atoms[:, :, None] - bb_atoms[:, None, :]
    distance_mask = bb_mask[:, :, None] * bb_mask[:, None, :]
    distances = torch.sqrt(torch.sum(distance_vec**2, dim=-1)) * distance_mask
    
    # Gather distances from upper triangular matrix
    distances_flat = ops.triu_flatten(triu=distances)

    def _rbf(D, num_rbf):
        D_min, D_max, D_count = 0., 10., num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view([1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    distance_feats = _rbf(distances_flat, num_rbf=num_rbf).reshape(n_residues, -1)
    return distance_feats


def construct_featurizer(args: argparse.Namespace) -> ProteinFeaturizer:
    """Constructs the featurizer from command line arguments."""

    if 'featurizer' not in args or args.featurizer is None:
        return None

    if args.featurizer == "base":
        return ProteinFeaturizer(
            agent_type=args.agent_type,
            resolution=args.resolution
        )
    elif args.featurizer == "pifold":
        return PiFoldFeaturizer(
            agent_type=args.agent_type,
            resolution=args.resolution,
            num_rbf=args.num_pifold_rbf if 'num_pifold_rbf' in args else 20
        )
    else:
        raise ValueError(f"Featurizer {args.featurizer} is not supported.")
