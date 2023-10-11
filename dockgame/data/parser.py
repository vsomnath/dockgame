import os
import dataclasses
from typing import Sequence, Union, Mapping

import dill
import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif

from dockgame.common import structure
from dockgame.common import constants


EXT_TO_CHAIN_COLS = {
    '.pdb': 'chain_id',
    '.dill': 'chain',
    '.cif': 'chain_id'
}

EXT_TO_RES_COLS = {
    '.pdb': ['residue_name', 'residue_number'],
    '.dill': ['resname', 'residue'],
    '.cif': ['residue_name', 'residue_number']
}

EXT_TO_POS_COLS = {
    '.pdb': ['x_coord', 'y_coord', 'z_coord'],
    '.dill': ['x', 'y', 'z'],
    '.cif': ['x_coord', 'y_coord', 'z_coord']
}

ProteinList = Sequence[structure.Protein]


@dataclasses.dataclass
class ProteinParser:
    """Parser to generate a Protein object for different filetypes."""

    def parse(self, filenames: Mapping[str, str]) -> ProteinList:
        assert type(filenames) == dict
        ext = _extension_from_filename(list(filenames.values())[0])

        if ext == ".pdb":
            parse_fn = self.parse_from_pdb
        elif ext == ".cif":
            parse_fn = self.parse_from_mmcif
        elif ext == ".dill":
            parse_fn = self.parse_from_dill
        else:    
            raise ValueError(f'Extension wtih type {ext} is not supported.'
                             f'Possible options include .pdb, .cif, .dill')
        
        chain_col = EXT_TO_CHAIN_COLS.get(ext, None)
        residue_cols = EXT_TO_RES_COLS.get(ext, None)
        pos_cols = EXT_TO_POS_COLS.get(ext, None)

        return parse_fn(
            filenames=filenames, 
            chain_col=chain_col,
            residue_cols=residue_cols,
            pos_cols=pos_cols
        )    

    def parse_from_pdb(self, 
                       filenames: Mapping[str, str], 
                       chain_col: str,
                       residue_cols: Sequence[str],
                       pos_cols: Sequence[str]) -> ProteinList:
        structures = []
        for name, filename in filenames.items():
            df = _parse_pdb_biopandas(filename=filename)
            structure = _build_structure(
                name=name, df=df, chain_col=chain_col, 
                residue_cols=residue_cols, pos_cols=pos_cols
            )
            structures.append(structure)

        return structures
    
    def parse_from_mmcif(self, 
                         filenames: Mapping[str, str], 
                         chain_col: str,
                         residue_cols: Sequence[str],
                         pos_cols: Sequence[str]) -> ProteinList:
        structures = []
        for name, filename in filenames.items():
            df = _parse_mmcif_biopandas(filename=filename)
            structure = _build_structure(
                name=name, df=df, chain_col=chain_col, 
                residue_cols=residue_cols, pos_cols=pos_cols
            )
            structures.append(structure)
        return structures

    def parse_from_dill(self, 
                        filenames: Mapping[str, str], 
                        chain_col: str,
                        residue_cols: Sequence[str],
                        pos_cols: Sequence[str]) -> ProteinList:
        df = _parse_dill(filename=list(filenames.values())[0])
        struct_ligand = _build_structure(
                name="ligand",  df=df.df0, chain_col=chain_col, 
                residue_cols=residue_cols, pos_cols=pos_cols
            )
        struct_receptor = _build_structure(
                name="receptor", df=df.df1, chain_col=chain_col, 
                residue_cols=residue_cols, pos_cols=pos_cols
            )
        structures = [struct_ligand, struct_receptor]
        return structures


def _extension_from_filename(filename: str) -> str:
    """Get extension from filename."""
    ext = os.path.splitext(filename)[-1]
    return ext


def _parse_pdb_biopandas(filename: str) -> pd.DataFrame:
    """Parses files with .pdb extension using biopandas."""
    pdb_file = os.path.abspath(filename)
    ppdb = PandasPdb().read_pdb(pdb_file)
    df = ppdb.df['ATOM']
    df['residue_number'] = df['residue_number'].astype(int)
    return df


def _parse_mmcif_biopandas(filename: str) -> pd.DataFrame:
    """Parses files with .cif extension using biopandas."""
    ppdb = PandasMmcif().read_mmcif(path=filename)
    df = ppdb._df['ATOM']
    df.rename(columns={
        'auth_atom_id': 'atom_name',
        'auth_comp_id': 'residue_name',
        'auth_seq_id': 'residue_number',
        'auth_asym_id': 'chain_id',
        'Cartn_x': 'x_coord',
        'Cartn_y': 'y_coord',
        'Cartn_z': 'z_coord'
        }, inplace=True)
    
    df['residue_number'] = df['residue_number'].astype(int)
    return df


def _parse_dill(filename: str) -> pd.DataFrame:
    """Parses files with .dill extension."""
    with open(filename, "rb") as f:
        df = dill.load(f)
    return df


def _build_structure(
    name: str,
    df: pd.DataFrame,
    chain_col: str,
    residue_cols: Sequence[str],
    pos_cols: Sequence[str] = ['x', 'y', 'z']
) -> structure.Protein:
    """
    Builds a `Structure` object from a pd.DataFrame representaion of protein.
    
    Args:
        df: A pandas DataFrame object containing the atom records of the protein.
            The atom records contain information like the atom name, residue it 
            is part of, the position of the residue, and the chain membership.
        chain_col: Field in the DataFrame that indicates the chain membership.
        residue_cols: Fields in the DataFrame that indicate the residue name and
            its position in the protein sequence
        pos_cols: Fields in the DataFrame that indicate the x, y, z coords of the
            participating atoms

    Returns:
        structure.Structure object
    """
    residue_atom_pos = []
    residue_atom_mask = []
    residue_types = []

    residue_ids = []
    chain_ids = []
    residue_index = []
    
    residue_name_col, residue_num_col = residue_cols

    for idx in range(len(df)):
        atom_record = df.iloc[idx]

        atom_name = atom_record['atom_name']
        chain_id = atom_record[chain_col]
        res_name = atom_record[residue_name_col]
        residue_loc_in_chain = int(atom_record[residue_num_col])

        if atom_name not in constants.atom_types:
            continue

        residue_id = (chain_id, residue_loc_in_chain, res_name)

        if residue_id not in residue_ids:
            residue_atom_pos.append(np.zeros((constants.atom_type_num, 3)))
            residue_atom_mask.append(np.zeros((constants.atom_type_num,)))

            residue_ids.append(residue_id)
            residue_types.append(res_name)
            residue_index.append(residue_loc_in_chain)
            chain_ids.append(atom_record[chain_col])

        # Gather index of residue in the residue_id_list to update atom mask and
        # atom position
        residue_idx_in_struct = residue_ids.index(residue_id)
        atom_pos = residue_atom_pos[residue_idx_in_struct]
        atom_mask = residue_atom_mask[residue_idx_in_struct]

        # Gather atom coords and update atom_pos and atom_mask
        atom_coords = atom_record[pos_cols]
        atom_idx_in_residue = constants.atom_order[atom_name]
        atom_pos[atom_idx_in_residue] = atom_coords
        atom_mask[atom_idx_in_residue] = 1.0
        
        # Update residue_atom_pos and residue atom mask
        residue_atom_pos[residue_idx_in_struct] = atom_pos
        residue_atom_mask[residue_idx_in_struct] = atom_mask

    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n+1 for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return structure.Protein(
        name=name,
        chain_ids=chain_index,
        atom_positions=np.array(residue_atom_pos),
        atom_mask=np.array(residue_atom_mask),
        residue_types=np.array(residue_types),
        residue_index=np.array(residue_index)
    )
