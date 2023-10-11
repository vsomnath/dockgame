import os
from itertools import product
from multiprocessing import Pool
import random
import pickle
import traceback
from typing import Union

import numpy as np
import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform

from dockgame.data import parser, featurize
from dockgame.utils import geometry as geometry_ops

# Type aliases
Array = np.ndarray
DecoyInfo = dict[str, dict[str, Array]]


def pretty_print_pyg(data: HeteroData, key: str) -> str:
    attr_str = f"x={data[key].x.shape},"
    attr_str += f" pos={data[key].pos.shape},"
    attr_str += f" pos_bound={data[key].pos_bound.shape}"

    if 'edge_index' in data[key]:
        edge_index = data[key].edge_index
        if edge_index is not None:
            attr_str += f", edge_index={data[key].edge_index.shape}"
    return attr_str


def _construct_filenames(complex_id: str, dataset: str = 'db5') -> dict[str, str]:
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
            complex_id: f"{complex_id}-assembly1.cif"
        }
    
    return filenames


# ==============================================================================
# Base Dataset class
# ==============================================================================


class BaseDataset(Dataset):

    def __init__(
        self,
        root: str,
        parser: parser.ProteinParser,
        featurizer: featurize.ProteinFeaturizer,
        complex_list_file: str,
        transform: BaseTransform = None,
        dataset: str = "db5",
        complex_dir: str = 'complexes',
        mode: str = "train",
        resolution: str = "c_alpha",
        agent_type: str = "protein",
        num_workers: int = 1,
        progress_every: int = None,
        esm_embeddings_path: str = None,
        center_complex: bool = False,
        size_sorted: bool = False,
    ):
        super().__init__(root=root, transform=transform)

        # Base directory where splits and other metadata are stored
        self.raw_data_dir = os.path.join(self.raw_dir, dataset)

        if complex_list_file is not None:
            self.complex_list_file = os.path.join(
                self.raw_data_dir, complex_list_file)
        else:
            self.complex_list_file = None

        # Directory where complexes files (.pdb, .mmcif etc) are stored
        self.complex_dir = os.path.join(self.raw_data_dir, complex_dir)

        # Directory where processed files are stored
        self.processed_data_dir = os.path.join(self.processed_dir, dataset)

        self.dataset = dataset
        self.parser = parser
        self.featurizer = featurizer

        self.mode = mode
        self.resolution = resolution
        self.agent_type = agent_type

        self.num_workers = num_workers
        self.progress_every = progress_every
        self.center_complex = center_complex
        self.size_sorted = size_sorted
        
        proceseed_arg_str = f"agent={agent_type}_resolution={resolution}"
        if featurizer is not None:
            proceseed_arg_str += f"_featurizer={featurizer.name}"
        if esm_embeddings_path is not None:
            proceseed_arg_str += f"_esmEmbeddings"
        if center_complex:
            proceseed_arg_str += f"_centered"
        if size_sorted:
            proceseed_arg_str += f"_size-sorted"

        self.full_processed_dir = os.path.join(
            self.processed_data_dir, proceseed_arg_str
        )

        # Loading all complex ids
        with open(f"{self.complex_list_file}", "r") as f:
            complex_ids_all = f.readlines()
            complex_ids_all = [complex_id.strip() for complex_id in complex_ids_all]

        self.complexes_split = [
            complex for complex in complex_ids_all
            if os.path.exists(f"{self.full_processed_dir}/{complex}.pt")
        ]

    def load_ids(self):
        raise NotImplementedError("Subclasses must implement for themselves")

    def len(self) -> int:
        return len(self.ids)
    
    def get(self):
        raise NotImplementedError("Subclasses must implement for themselves")

    def preprocess_complexes(self):
        os.makedirs(self.full_processed_dir, exist_ok=True)

        # Loading all complex ids
        with open(f"{self.complex_list_file}", "r") as f:
            complex_ids_all = f.readlines()
            complex_ids_all = [complex_id.strip() for complex_id in complex_ids_all]

        print(f"Preprocessing {len(complex_ids_all)} complexes.", flush=True)
        print(f"Loading from: {self.complex_dir}", flush=True)
        print(f"Saving to: {self.full_processed_dir}", flush=True)
        print(flush=True)

        failures = []

        if self.num_workers > 1:
            for i in range(len(complex_ids_all) // self.progress_every + 1):
                complex_ids_batch = complex_ids_all[
                    self.progress_every * i : self.progress_every * (i + 1)
                ]

                p = Pool(self.num_workers, maxtasksperchild=1)
                map_fn = p.imap_unordered
                for (complex, complex_id) in map_fn(
                    self.preprocess_complex, complex_ids_batch):
                    if complex is not None:
                        complex_file = f"{self.full_processed_dir}/{complex_id}.pt"

                        #pdb_ids have a / in their name which can get confused
                        if self.dataset.lower() == "dips": 
                            dirname = os.path.dirname(complex_file)
                            os.makedirs(dirname, exist_ok=True)

                        print(f"Saving {complex_id} to {complex_file}", flush=True)
                        torch.save(complex, f"{complex_file}")
                        print(flush=True)
                    else:
                        failures.append(complex_id)
                        print(flush=True)
                p.__exit__(None, None, None)

        else:
            for (complex, complex_id) in map(
                self.preprocess_complex, complex_ids_all):
                if complex is not None:
                    complex_file = f"{self.full_processed_dir}/{complex_id}.pt"

                    #pdb_ids have a / in their name which can get confusing
                    if self.dataset.lower() == "dips": 
                        dirname = os.path.dirname(complex_file)
                        os.makedirs(dirname, exist_ok=True)

                    print(f"Saving {complex_id} to {complex_file}", flush=True)

                    torch.save(complex, f"{complex_file}")
                    print(flush=True)
                else:
                    failures.append(complex_id)
                    print(flush=True)

        print("Finished preprocessing complexes", flush=True)
        print(f"Failures: {failures}", flush=True)

    def preprocess_complex(self, complex_id: str) -> Union[HeteroData, str]:
        filenames = _construct_filenames(
            complex_id=complex_id, 
            dataset=self.dataset
        )
        for key, filename in filenames.items():
            filenames[key] = f"{self.complex_dir}/{filename}"

        try:
            structures = self.parser.parse(filenames=filenames)
            base_complex = HeteroData()

            if self.featurizer:
                base_complex = self.featurizer.featurize(
                    structures=structures, 
                    graph=base_complex
                )
            
            if self.size_sorted:
                sizes = [base_complex[agent].x.size(0) for agent in base_complex.agent_keys]
                sorted_idxs = sorted(range(len(sizes)), key=lambda x: sizes[x])
                agent_keys = [base_complex.agent_keys[idx] for idx in sorted_idxs]
                base_complex.agent_keys = agent_keys
  
            if self.center_complex:
                base_complex = BaseDataset._center_complex(complex_data=base_complex)
        
            for agent in base_complex.agent_keys:
                attr_str_agent = pretty_print_pyg(base_complex, agent)
                print(f"{complex_id}: Prepared {agent} graph - {attr_str_agent}", flush=True)

            print(base_complex.agent_keys, flush=True)
            
            return base_complex, complex_id
        except Exception as e:
            print(f"Failed to process {complex_id} because of {e}", flush=True)
            traceback.print_exc()
            return None, complex_id
    
    @staticmethod
    def _center_complex(complex_data: HeteroData) -> HeteroData:
        stationary_agent = complex_data.agent_keys[-1]
        agent_center = complex_data[stationary_agent].pos.mean(dim=0, keepdims=True)

        for agent in complex_data.agent_keys:
            complex_data[agent].pos -= agent_center
            complex_data[agent].pos_bound -= agent_center
            print(agent, complex_data[agent].pos.mean(dim=0))

        assert torch.allclose(
            complex_data[stationary_agent].pos.mean(dim=0, keepdims=True),
            torch.zeros(1, 3), atol=1e-3
        )
        return complex_data

# ==============================================================================
# Reward Dataset
# ==============================================================================


class DockRewardDataset(BaseDataset):

    def __init__(
        self,
        n_decoys: int = 10,
        norm_method: str = "minmax",
        ref_choice: str = "bound",
        max_radius: float = None,
        max_neighbors: int = None,
        cross_max_radius: float = None, 
        cross_max_neighbors: int = None,
        max_tr_decoys: float = 10.0,
        weight_radius_gyration: float = 0.0,
        score_fn_decoys: str = "dock_low_res",
        model_name: str = "dock_score",
        **kwargs,
    ):  
        super().__init__(**kwargs)
        self.n_decoys = n_decoys
        self.norm_method = norm_method
        self.ref_choice = ref_choice
        self.model_name = model_name

        self.max_tr_decoys = max_tr_decoys
        self.weight_radius_gyration = weight_radius_gyration
        self.score_fn_decoys = score_fn_decoys

        self.max_radius = max_radius
        self.max_neighbors = max_neighbors
        self.cross_max_radius = cross_max_radius
        self.cross_max_neighbors = cross_max_neighbors

        self.ids_without_decoys = []
        if self.dataset == "db5":
            self.ids_without_decoys = [
                "2CFH", "4CPA", "3R9A", "1ZM4", "1TMQ", "3H11", "1DFJ"
            ]

        self.load_ids()

    def _construct_decoy_file(self, complex_id, mode: str = None):
        decoy_file = f"{complex_id}-agent={self.agent_type}-decoys={self.n_decoys}"
        decoy_file += f"-score={self.score_fn_decoys}"
        decoy_file += f"-max_tr={self.max_tr_decoys}"
        decoy_file += f"-{mode}.decoy"
        full_decoy_file = f'{self.raw_data_dir}/decoys/{decoy_file}'
        return full_decoy_file

    def load_ids(self):
        complex_ids = self.complexes_split            

        # First we remove all ids for which no *train* decoy files 
        # could be generated
        if self.norm_method is not None:
            if "minmax" in self.norm_method or self.norm_method == "std":
                self.decoy_stats = {}
                for complex_id in complex_ids:

                    decoy_file = self._construct_decoy_file(
                        complex_id=complex_id, mode='train'
                    )

                    if os.path.exists(decoy_file):
                        with open(decoy_file, "rb") as f:
                            decoy_info = pickle.load(f)

                        scores = [info["score"] for info in decoy_info]
                        if "minmax" in self.norm_method:    
                            self.decoy_stats[complex_id] \
                                = (np.min(scores), np.max(scores))
                        elif "std" in self.norm_method:
                            self.decoy_stats[complex_id] \
                                = (np.mean(scores), np.std(scores))
                    else:
                        self.ids_without_decoys.append(complex_id)

        # Then we move complex_ids from the current mode
        for complex_id in complex_ids:
            decoy_file = self._construct_decoy_file(
                complex_id=complex_id, mode=self.mode
            )

            if not os.path.exists(decoy_file):
                self.ids_without_decoys.append(complex_id)

        self.ids_without_decoys = list(set(self.ids_without_decoys))
        complex_ids = [complex_id for complex_id in complex_ids 
                        if complex_id not in self.ids_without_decoys]

        decoy_ids = list(range(self.n_decoys))
        self.ids = list(product(complex_ids, decoy_ids))

        if self.mode == "train":
            random.shuffle(self.ids)

        self.ids = list(product(complex_ids, decoy_ids))
        print(f"Number of {self.mode} found complexes: {len(complex_ids)}", flush=True)
        print(flush=True)

    def _adjust_agent_names(self, decoy_info):
        keys = decoy_info['agent_keys'].copy()
        for i, agent_key in enumerate(keys):
            if agent_key[-2] == '_' and agent_key[-1].isalpha():
                new_key = agent_key[:-1] + str(ord(agent_key[-1]) - ord('A') + 1)
                decoy_info[new_key] = decoy_info.pop(agent_key)
                keys[i] = new_key
        decoy_info.update({'agent_keys': keys})
        return
            
    def get(self, idx: int) -> HeteroData:
        complex_id, decoy_id = self.ids[idx]

        complex_file = f"{self.full_processed_dir}/{complex_id}.pt"

        decoy_file = self._construct_decoy_file(
            complex_id=complex_id, mode=self.mode
        )

        if not os.path.exists(complex_file) or not os.path.exists(decoy_file):
            return None
        
        # Load the complex and decoy information
        complex = torch.load(complex_file, map_location='cpu')
        complex_decoy = complex.clone()
        complex_ref = complex.clone()

        agent_keys = complex.agent_keys

        with open(decoy_file, "rb") as f:
            decoy_info = pickle.load(f)

        if self.n_decoys > 1 and self.ref_choice == "random":
            p = np.ones((self.n_decoys,))
            p[decoy_id] = 0 # zeroing out the decoy prob
            p = p / p.sum()
            ref_id = np.random.choice(np.arange(self.n_decoys), p=p)
            assert ref_id != decoy_id, "reference chosen cannot be same as decoy id in random sampling"

        elif self.n_decoys == 1 or self.ref_choice == "bound":
            ref_id = 0

        info = decoy_info[decoy_id]
        info_ref = decoy_info[ref_id]
        # Adjust name of info.agent_keys in case they contain 'A', 'B' instead of '1', '2', etc.
        # This is done to match the naming convention of the preprocessed complex
        self._adjust_agent_names(info)
        self._adjust_agent_names(info_ref)

        self._modify_complex(complex=complex_decoy, info=info)
        self._modify_complex(complex=complex_ref, info=info_ref)

        for agent in agent_keys:
            complex_decoy[agent].pos_ref = complex_ref[agent].pos.clone()

        score, score_ref = info["score"], info_ref["score"]
        #rg , rg_ref = info["radius_of_gyration"], info_ref["radius_of_gyration"]
        #score += self.weight_radius_gyration * rg
        #score_ref += self.weight_radius_gyration * rg_ref
        
        y = self._normalize_score(score, complex_id=complex_id)
        y_ref = self._normalize_score(score_ref, complex_id=complex_id)
        complex_decoy.y = y
        complex_decoy.y_ref = y_ref

        return complex_decoy
    
    def _normalize_score(self, score: float, complex_id: str, **kwargs) -> float:
        if self.norm_method is None:
            return score
        
        y = score
        if "minmax" in self.norm_method and self.n_decoys > 1:
            decoy_min, decoy_max = self.decoy_stats[complex_id]
            y = (score - decoy_min) / (decoy_max - decoy_min)

            if "tanh" in self.norm_method:
                y = 2 * y - 1

        elif self.norm_method == "sqrt":
            y = np.sign(score) * np.sqrt(np.abs(score))
        
        return y

    def _modify_complex(self, complex: HeteroData, info: DecoyInfo):
        agent_keys = complex.agent_keys

        if agent_keys[0] not in info:
            for agent_key in agent_keys:
                # Chain agents are written as "{protein_name}_{chain_name}"
                protein_key = agent_key.split("_")[0]

                if protein_key not in info:
                    continue

                rot_vec = info[protein_key]["rot_vec"]
                tr_vec = info[protein_key]["tr_vec"]
                rot_vec = torch.tensor(rot_vec).float()
                tr_vec = torch.tensor(tr_vec).float()

                new_pos_agent = geometry_ops.apply_rigid_transform(
                    pos=complex[agent_key].pos,
                    rot_vec=rot_vec, tr_vec=tr_vec
                )

                complex[agent_key].pos = new_pos_agent

        else:
            for agent_key in agent_keys:
                if agent_key not in info:
                    continue

                agent_dict = info[agent_key]
                if not agent_dict: # empty dict
                    continue

                rot_vec, tr_vec = agent_dict["rot_vec"], agent_dict["tr_vec"]
                rot_vec = torch.tensor(rot_vec).float()
                tr_vec = torch.tensor(tr_vec).float()

                new_pos_agent = geometry_ops.apply_rigid_transform(
                    pos=complex[agent_key].pos,
                    rot_vec=rot_vec, tr_vec=tr_vec
                )

                complex[agent_key].pos = new_pos_agent


# ==============================================================================
# Score Matching Dataset
# ==============================================================================


class DockScoreDataset(BaseDataset):

    def __init__(
        self,
        timepoints_per_complex: int = 1,
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.timepoints_per_complex = timepoints_per_complex
        self.load_ids()

    def load_ids(self):
        if self.timepoints_per_complex is not None:
            self.sample_ids = list(range(self.timepoints_per_complex))
            self.ids = list(product(self.complexes_split, self.sample_ids))

        else:
            self.ids = self.complexes_split
        print(f"Number of {self.dataset} {self.mode} complexes: {len(self.ids)}", flush=True)
        random.shuffle(self.ids)

    def get(self, idx: int) -> HeteroData:
        complex_id, _ = self.ids[idx]
        complex_file = f"{self.full_processed_dir}/{complex_id}.pt"
        if not os.path.exists(complex_file):
            return None

        complex_base = torch.load(complex_file, map_location="cpu")
        return complex_base.clone()
