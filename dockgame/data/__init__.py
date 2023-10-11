from torch_geometric.loader import DataLoader, DataListLoader

from dockgame.data.featurize import construct_featurizer
from dockgame.data.dataset import (
    BaseDataset, DockScoreDataset, DockRewardDataset
)


def get_mode_specific_base_params(args, mode='train'):
    base_params = {}
    if mode == 'train':
        complex_list_file = args.train_complex_list_file
        complex_dir = args.train_complex_dir
        dataset = args.train_dataset
        if args.model != "score":
            n_decoys = args.num_decoys
        size_sorted = args.train_size_sorted if 'train_size_sorted' in args else False
    
    elif mode == 'val':
        complex_list_file = args.val_complex_list_file \
            if args.val_complex_list_file is not None else args.train_complex_list_file
        complex_dir = args.val_complex_dir \
            if args.val_complex_dir is not None else args.train_complex_dir
        dataset = args.val_dataset \
            if args.val_dataset is not None else args.train_dataset
        if args.model != "score":
            n_decoys = args.val_num_decoys \
                if args.val_num_decoys is not None else args.num_decoys

        size_sorted = False            
        if 'val_size_sorted' in args:
            size_sorted = args.val_size_sorted
        else:
            if 'train_size_sorted' in args:
                size_sorted = args.train_size_sorted

    base_params = {
        'complex_list_file': complex_list_file,
        'complex_dir': complex_dir,
        'dataset': dataset,
        'size_sorted': size_sorted
    }

    if args.model != "score":
        base_params['n_decoys'] = n_decoys

    return base_params


def build_reward_dataset(args, mode: str = "train"):

    from dockgame.data.transforms import construct_reward_transform

    transform = construct_reward_transform(args=args)
    featurizer = construct_featurizer(args=args)

    base_params = get_mode_specific_base_params(args=args, mode=mode)

    params = {
        "root": args.data_dir,
        "parser": None,
        "featurizer": featurizer,
        "transform": transform,
        "mode": mode,
        "resolution": args.resolution,
        "agent_type": args.agent_type,
        "center_complex": args.center_complex,
        "esm_embeddings_path": None,
        "n_decoys": args.num_decoys,
        "norm_method": args.norm_method,
        "ref_choice": args.ref_choice,
        "score_fn_decoys": args.score_fn_decoys,
        "max_tr_decoys": args.max_tr_decoys,
        "model_name": args.model,
    }

    params.update(base_params)
    dataset = DockRewardDataset(**params)
    return dataset


def build_score_dataset(args, mode: str = "train"):

    from dockgame.data.transforms import construct_score_transform

    base_params = get_mode_specific_base_params(args=args, mode=mode)

    if mode in ["val", "test"]:
        timepoints_per_complex = 1
    else:
        timepoints_per_complex = args.timepoints_per_complex

    featurizer = construct_featurizer(args=args)
    transform = construct_score_transform(args, mode='train')

    params = {
        "root": args.data_dir,
        "parser": None,
        "featurizer": featurizer,
        "transform": transform,
        "mode": mode,
        "resolution": args.resolution,
        "agent_type": args.agent_type,
        "center_complex": args.center_complex,
        "esm_embeddings_path": None,
        "timepoints_per_complex": timepoints_per_complex
    }

    params.update(base_params)
    dataset = DockScoreDataset(**params)
    return dataset
    

def build_data_loader(args, mode: str = "train"):
    if args.model in [
        "dock_score", "dock_score_hetero", "dock_reward", "dock_reward_hetero"
    ]:
        dataset = build_reward_dataset(args=args, mode=mode)
    
    elif args.model == "score":
        dataset = build_score_dataset(args=args, mode=mode)

    if mode == "train":
        batch_size = args.train_bs
    elif mode == "val":
        batch_size = args.val_bs
    
    loader_cls = DataListLoader if args.n_gpus > 1 else DataLoader
    loader = loader_cls(
        dataset=dataset, batch_size=batch_size, shuffle=(mode=="train")
    )
    return loader
