import os
import argparse
import yaml
from functools import partial

import torch
from torch_geometric.nn.data_parallel import DataParallel

from dockgame.models.dock_reward import RewardModel
from dockgame.models.dock_reward_hetero import RewardModelHetero
from dockgame.models.score_model import ScoreModel
from dockgame.utils.diffusion import get_timestep_embedding, t_to_sigma
from dockgame.common.constants import DEVICE

from dockgame.data.featurize import FEATURE_DIMENSIONS


MODELS = {
    "dock_score": RewardModel,
    "dock_score_hetero": RewardModelHetero,
    "score": ScoreModel
}

ALLOWED_MODELS = list(MODELS.keys())


def build_reward_model_from_args(args):

    model_cls = MODELS.get(args.model, None)
    if model_cls is None:
        raise ValueError(f"{args.model} is not a valid model name. Allowed models: {list(MODELS.keys())}")
    
    if 'featurizer' not in args or args.featurizer is None:
        args.featurizer = "base"
    
    node_fdim = FEATURE_DIMENSIONS[args.featurizer]

    params = {
        "node_fdim": node_fdim,
        "edge_fdim": args.edge_fdim,
        "sh_lmax": args.sh_lmax,
        "n_s": args.n_s, "n_v": args.n_v,
        "n_conv_layers": args.n_conv_layers,
        "max_radius": args.max_radius,
        "distance_emb_dim": args.distance_emb_dim,
        "dropout_p": args.dropout_p,
        "activation": args.activation,
        "n_deviations": args.n_deviations,
        "deviation_eps": args.deviation_eps,
    }

    # Just a fix for some backward compatability
    try:
        params["enforce_stability"] = args.enforce_stability
    except Exception:
        params["enforce_stability"] = args.compute_grad_norm

    if args.model == "dock_score":
        # Used in building the graph inside the model
        params["max_neighbors"] =  args.max_neighbors
    elif args.model == "dock_score_hetero":
        # Used for distance expansion
        params["cross_max_radius"] = args.cross_max_radius
        params["cross_dist_emb_dim"] = args.cross_dist_emb_dim if "cross_dist_emb_dim" in args else None
    
    model = model_cls(**params)
    return model


def build_score_model_from_args(args):

    if 'featurizer' not in args or args.featurizer is None:
        args.featurizer = "base"

    if 'node_encoder_type' not in args:
        args.node_encoder_type = "base"
    
    node_fdim = FEATURE_DIMENSIONS[args.featurizer]

    model_cls = MODELS.get(args.model, None)
    if model_cls is None:
        raise ValueError(
            f"{args.model} is not a valid model name." +
            f"Allowed models: {list(MODELS.keys())}"
        )
    
    timestep_emb_fn = get_timestep_embedding(
        embedding_type=args.time_embed_type,
        embedding_dim=args.sigma_emb_dim,
        embedding_scale=args.time_embed_scale
    )
    
    t_to_sigma_fn = partial(
        t_to_sigma,
        tr_sigma_min=args.tr_sigma_min,
        tr_sigma_max=args.tr_sigma_max,
        rot_sigma_min=args.rot_sigma_min,
        rot_sigma_max=args.rot_sigma_max
    )
    
    params = {
        "node_fdim": node_fdim,
        "node_encoder_type": args.node_encoder_type,
        "edge_fdim": args.edge_fdim,
        "sh_lmax": args.sh_lmax,
        "n_s": args.n_s, "n_v": args.n_v,
        "n_conv_layers": args.n_conv_layers,
        "max_radius": args.max_radius,
        "cross_max_radius": args.cross_max_radius,
        "center_max_radius": args.center_max_radius,
        "distance_emb_dim": args.distance_emb_dim,
        "cross_dist_emb_dim": args.cross_dist_emb_dim,
        "center_dist_emb_dim": args.center_dist_emb_dim,
        "timestep_emb_fn": timestep_emb_fn,
        "sigma_emb_dim": args.sigma_emb_dim,
        "dropout_p": args.dropout_p,
        "activation": args.activation,
        "scale_by_sigma": args.scale_by_sigma if "scale_by_sigma" in args else False,
        "t_to_sigma_fn": t_to_sigma_fn
    }

    model = model_cls(**params)
    return model


def build_model_from_args(args, mode: str = "train") -> torch.nn.Module:
    if args.model == "score":
        print("Building Score Model from args:")
        model = build_score_model_from_args(args=args)

    else:
        print("Building reward model from args:")
        model = build_reward_model_from_args(args=args)
    
    if args.n_gpus > 1 and mode != "test":
        print("Using DataParallel for multiple gpus", flush=True)
        model = DataParallel(model)

    model.to(DEVICE)
    return model


def load_model_from_args(
        args, mode: str = "train", 
        return_model_args: bool = False) -> dict[str, torch.Tensor]:
    model_ckpt = _fetch_pretrained_model_ckpt(args=args)
    if model_ckpt is None:
        raise ValueError("Unable to fetch model ckpt.")
    
    print(f'Model ckpt found at {model_ckpt}')
    ckpt_info = _load_pretrained_checkpoint(
        model_ckpt_file=model_ckpt, return_model_args=return_model_args
    )

    if 'n_gpus' in args and args.n_gpus > 1:
        model_state_dict = {}
        for key, value in ckpt_info['model_state'].items():
            model_state_dict[f'module.{key}'] = value
    else:
        model_state_dict = ckpt_info['model_state']

    if 'model_args' in ckpt_info:
        model_args = ckpt_info['model_args']

        print(f"Found model args in restored checkpoint.", flush=True)
        print(f"Restored checkpoint args: {model_args}", flush=True)
        print(flush=True)

        model = build_model_from_args(model_args, mode=mode)
        
        if 'n_gpus' in args and args.n_gpus > 1:
            model_state_dict = {}
            for key, value in ckpt_info['model_state'].items():
                model_state_dict[f'module.{key}'] = value
        else:
            model_state_dict = ckpt_info['model_state']

        model.load_state_dict(model_state_dict)
        return model, model_args
    
    return model_state_dict


def _fetch_pretrained_model_ckpt(args):
    if 'restore_from' in args:
        print("Found restore_from in args", flush=True)
        return args.restore_from

    if 'model_dir' in args:
        print("Found model_dir in args", flush=True)
        if 'model_name' not in args:
            print("No model name found in args. Using best_ema_model.pt")
            model_name = "best_ema_model.pt"
        else:
            model_name = args.model_name
        model_ckpt = f"{args.model_dir}/{model_name}"
        return model_ckpt

    if 'log_dir' in args and 'exp_name' in args:
        print("Found log_dir and exp_name in args", flush=True)
        model_ckpt = f"{args.log_dir}/{args.exp_name}/{args.model_name}"
        return model_ckpt
    
    return None


def _load_pretrained_checkpoint(
    model_ckpt_file: str,
    return_model_args: bool = False
) -> dict[str, dict[str, torch.Tensor]]:
    model_dict = torch.load(model_ckpt_file, map_location='cpu')
    model_dir = os.path.dirname(model_ckpt_file)

    if 'model' in model_dict.keys():
        model_dict = model_dict['model']

    with open(f'{model_dir}/config_train.yml') as f:
        model_args = argparse.Namespace(**yaml.full_load(f))

    ckpt_info = {}

    check_key = list(model_dict.keys())[0]
    if 'module.' in check_key: # Potential legacy models on multiple gpus
        model_dict_single_gpu = {}
        for key, value in model_dict.items():
            new_key = ".".join(key.split(".")[1:])
            model_dict_single_gpu[new_key] = value
    
        ckpt_info['model_state'] = model_dict_single_gpu
    else:
        ckpt_info['model_state'] = model_dict

    
    if return_model_args:
        ckpt_info['model_args'] = model_args

    return ckpt_info
