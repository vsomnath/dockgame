import dataclasses
from typing import Union, Sequence

import numpy as np

from dockgame.common import constants

# ==============================================================================
# Adapted from AlphaFold 
# ==============================================================================

# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Residue:
    """Residue object that contains atom coordinates and atom mask."""
    atom_positions: np.ndarray # [num_atoms, 3]
    atom_mask: np.ndarray # [num_atoms]


@dataclasses.dataclass(frozen=True)
class Chain:
    """Structural representation of a chain."""

    # Name of the protein
    protein_name: str

    # Identifier for the chain
    chain_id: int

    name: str

    # Chain ids list
    chain_ids: np.ndarray

    # Atom coordinates (in angstroms)
    atom_positions: np.ndarray  # [num_res_in_chain, num_atom_type, 3]

    # Binary float mask to indicate the presence of a particular atom. 1.0 if 
    # atom is present and 0.0 if not. Used in feature computation
    atom_mask: np.ndarray  # [num_res_in_chain, num_atom_type]

    # Residue index as used in PDB. Not necessarily continuous or 0-indexed
    residue_index: np.ndarray  # [num_res]

    # Residue names
    residue_types: np.ndarray # [num_res_in_struct]


@dataclasses.dataclass(frozen=True)
class Protein:
    """Class representing a protein. 
    
    In DB5.5 dataset, the ligand or receptor protein can have multiple chains,
    which by the conventional definition makes it a complex. This class
    assumes that the simplest unit is a chain or a monomer, and the Structure
    object corresponding to a protein can be composed of multiple monomers.
    """
    
    # Name of the protein
    name: str

    # Chain identifiers for all residues in the protein
    chain_ids: np.ndarray # [num_res_in_struct]

    # Atom coordinates (in angstroms)
    atom_positions: np.ndarray # [num_res_in_struct, num_atom_type, 3]

    # Binary mask indicating presence (1.0) or absence of an atom (1.0)
    # Used in feature computation
    atom_mask: np.ndarray # [num_res_in_struct]

    # Residue index as used in PDB. Not necessarily continuous or 0-indexed
    residue_index: np.ndarray # [num_res_in_struct]

    # Residue names
    residue_types: np.ndarray # [num_res_in_struct]
 
    def get_chains(self):
        chain_ids = np.unique(self.chain_ids)

        for chain_id in chain_ids:
            chain_mask = (self.chain_ids == chain_id)
            yield Chain(
                protein_name=self.name,
                chain_id=chain_id,
                name=f"{self.name}_{chain_id}",
                chain_ids=self.chain_ids[chain_mask], 
                atom_positions=self.atom_positions[chain_mask],
                atom_mask=self.atom_mask[chain_mask],
                residue_types=self.residue_types[chain_mask],
                residue_index=self.residue_index[chain_mask]
            )

    def get_residues(self, chain_id: str = None):
        for res_idx in range(self.atom_positions.shape[0]):
            if chain_id is not None:
                chain_id_residue = self.chain_ids[res_idx]
                if chain_id_residue == chain_id:
                    yield Residue(
                        atom_positions=self.atom_positions[res_idx],
                        atom_mask=self.atom_mask[res_idx]
                    )
            
            else:
                yield Residue(
                            atom_positions=self.atom_positions[res_idx],
                            atom_mask=self.atom_mask[res_idx]
                        )
    
    @classmethod
    def from_chain_list(cls, chains, name='structure') -> 'Protein':
        atom_positions = np.concatenate(
           [chain.atom_positions for chain in chains], axis=0)
        atom_mask = np.concatenate([chain.atom_mask for chain in chains], axis=0)
        residue_types = np.concatenate([chain.residue_types for chain in chains], axis=0)
        
        # Chain index starting from 1
        chain_ids = []
        chain_id_counter = 1
        for chain in chains:
            chain_ids_arr = np.full(
                (chain.atom_positions.shape[0],), fill_value=chain_id_counter)
            chain_ids.append(chain_ids_arr)
            chain_id_counter += 1
        chain_ids = np.concatenate(chain_ids, axis=0)

        # No need to increment these since these are chain specific
        residue_index = np.concatenate(
            [chain.residue_index for chain in chains], axis=0)

        return cls(
           name=name, chain_ids=chain_ids, atom_positions=atom_positions,
           atom_mask=atom_mask, residue_types=residue_types, 
           residue_index=residue_index
        )
    
    @classmethod
    def from_protein_list(cls, structures, name='structure') -> 'Protein':
        atom_positions = np.concatenate(
           [structure.atom_positions for structure in structures], axis=0)
        atom_mask = np.concatenate([structure.atom_mask for structure in structures], axis=0)
        residue_types = np.concatenate([structure.residue_types for structure in structures], axis=0)

        # Chain index starting from 1
        chain_ids = []
        chain_id_counter = 0
        for structure in structures:
            chain_ids.append(structure.chain_ids + chain_id_counter)
            chain_id_counter += 1
        chain_ids = np.concatenate(chain_ids, axis=0)

        residue_index = np.concatenate(
            [structure.residue_index for structure in structures], axis=0)

        return cls(
           name=name, chain_ids=chain_ids, atom_positions=atom_positions,
           atom_mask=atom_mask, residue_types=residue_types, 
           residue_index=residue_index
        )
    

# ==============================================================================
# I/O utils for Structures (Taken from AlphaFold)
# ==============================================================================

Structure = Union[Chain, Protein]


def _chain_end(
    atom_index: int, end_resname: str, chain_name: str, residue_index: int
) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')


def to_pdb(structure: Protein) -> list[str]:
    """Converts a given structure to a PDB string.
   
    Args:
     structure: The protein/chain to convert to PDB.
    
    Returns:
     PDB string.
    """
    restypes = constants.restypes + ['X']
#   res_1to3 = lambda r: constants.restype_1to3.get(restypes[r], 'UNK')
    atom_types = constants.atom_types
    pdb_lines = []
   
    atom_mask = structure.atom_mask
    aatype = structure.residue_types.tolist()
    aatype_int = [constants.restype_order_3.get(elem, constants.restype_num) for elem in aatype]
    aatype_int = np.array(aatype_int).astype(np.int32)

    atom_positions = structure.atom_positions
    # (Old version) We don't store residue index so assumed a continuous order for now!
    # (Old version) residue_index = np.arange(len(aatype)).astype(np.int32) + 1
    residue_index = structure.residue_index.astype(np.int32)
    chain_index = structure.chain_ids.astype(np.int32)
    # Beta factors are not stored, so zeroed out.
    b_factors = np.zeros(structure.atom_mask.shape)

    if np.any(aatype_int > constants.restype_num):
        raise ValueError('Invalid aatypes.') 

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
              f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
        chain_ids[i] = PDB_CHAIN_IDS[i-1]

    pdb_lines.append('MODEL     1')
    atom_index = 1
    last_chain_index = chain_index[0]

    # Add all atom sites.
    for i in range(aatype_int.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(_chain_end(
                atom_index, aatype[i - 1], chain_ids[chain_index[i - 1]],
                residue_index[i - 1]))
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = aatype[i]
        for atom_name, pos, mask, b_factor in zip(
                atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue
       
            record_type = 'ATOM'
            name = atom_name if len(atom_name) == 4 else f' {atom_name}'
            alt_loc = ''
            insertion_code = ''
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ''
            # PDB is a columnar format, every space matters here!
            atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                        f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                        f'{residue_index[i]:>4}{insertion_code:>1}   '
                        f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                        f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                        f'{element:>2}{charge:>2}')
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(_chain_end(atom_index, aatype[-1],
                              chain_ids[chain_index[-1]], residue_index[-1]))
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'


def write_to_pdb(structures: Sequence[Structure], filename: str):
    if isinstance(structures, list):
        base_obj = structures[0]
        if isinstance(base_obj, Protein):
            structure = Protein.from_protein_list(structures)
        elif isinstance(base_obj, Chain):
            structure = Protein.from_chain_list(structures, name='structure')
        else:
            pass

        pdb_lines = to_pdb(structure=structure)
    else:
        pdb_lines = to_pdb(structure=structures)

    with open(filename, "w") as f:
        f.write(pdb_lines)
