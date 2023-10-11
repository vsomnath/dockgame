import argparse
from argparse import FileType
import os
from typing import Callable

import torch
import wandb
import yaml

from dockgame.common.constants import (
    WANDB_ENTITY, CLUSTER_EXP_DIR, IS_CLUSTER, EXP_DIR
)
from dockgame.data import build_data_loader
from dockgame.models import build_model_from_args, load_model_from_args
from dockgame.training.updaters import get_ema, get_optimizer, get_scheduler


# Type aliases
Args = argparse.Namespace
ArgParser = argparse.ArgumentParser
Model = torch.nn.Module


def construct_log_dir(args: Args) -> str:
    if args.log_dir is not None:
        experiment_str = f"train_ds={args.train_dataset}"
        if args.val_dataset is not None:
            experiment_str += f"-val_ds={args.val_dataset}"        
        experiment_str += f"-model={args.model}-agent={args.agent_type}-feat={args.featurizer}"

        base_dir = f"{args.log_dir}/{experiment_str}"
        os.makedirs(base_dir, exist_ok=True)
        log_dir = f"{base_dir}/{args.run_name}"
        os.makedirs(log_dir, exist_ok=True)

        print(f"Logging experiments at directory: {log_dir}", flush=True)
        print(f"Experiment Name: {experiment_str}-{args.run_name}", flush=True)
        return log_dir
    return None


def wandb_setup(args: Args):
    DIR = CLUSTER_EXP_DIR if IS_CLUSTER else EXP_DIR
    print(f"Supplied experiment directory: {DIR}", flush=True)

    if not os.path.exists(DIR):
        os.makedirs(DIR)

    run_id = wandb.util.generate_id()

    if args.run_name is None:
        if args.group_name is not None:
            args.run_name = args.group_name + f"-{run_id}"
        else:
            args.run_name = run_id
    else:
        args.run_name = args.run_name + f"-{run_id}"
    
    print("Setting up wandb...", flush=True)
    wandb.init(
        id=run_id,
        project=args.project_name if 'project_name' in args else 'dockgame',
        entity=args.wandb_entity,
        group=args.group_name,
        name=args.run_name,
        config=vars(args),
        dir=DIR,
        mode=args.wandb_mode,
        notes=args.notes,
        job_type=args.job_type
    )


def setup_parser() -> ArgParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Directory to save data to.")
    parser.add_argument("--log_dir", default=None, 
                        help="Directory to save local logs to.")
    parser.add_argument("--config", type=FileType(mode='r'), 
                        help="Config file to load args from. args will be overwritten")
    parser.add_argument("--train_complex_dir", type=str, default='complexes',
                        help="Directory where train complexes are located (e.g complexes)")
    parser.add_argument("--val_complex_dir", type=str, default=None,
                        help="Directory where val complexes are located")
    parser.add_argument("--train_complex_list_file", type=str, default='complexes.txt',
                        help="List of complexes train")
    parser.add_argument("--val_complex_list_file", type=str, default=None,
                        help="List of complexes train")
    
    parser.add_argument("--restore_from", default=None, 
                        help="Where to restore pretrained model from.")
    
    parser.add_argument("--n_gpus", default=1, type=int)

    # wandb
    parser.add_argument("--project_name", default="dockgame")
    parser.add_argument("--wandb_entity", default=WANDB_ENTITY)
    parser.add_argument("--group_name", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--wandb_mode", default="disabled")
    parser.add_argument("--job_type", default=None)
    parser.add_argument("--notes", default=None)

    # Data
    parser.add_argument("--train_dataset", default="db5", type=str)
    parser.add_argument("--val_dataset", default=None, type=str)
    parser.add_argument("--resolution", default="c_alpha", type=str)
    parser.add_argument("--agent_type", default="protein", type=str)
    parser.add_argument("--center_complex", action='store_true')
    parser.add_argument("--featurizer", default=None, 
                        choices=["base", "pifold", None])
    parser.add_argument("--train_size_sorted", action='store_true')
    parser.add_argument("--val_size_sorted", action='store_true')
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--train_bs", default=8, type=int)
    parser.add_argument("--val_bs", default=2, type=int)
    parser.add_argument("--max_radius", type=float, default=10.0)
    parser.add_argument("--max_neighbors", type=int, default=32)
    parser.add_argument("--cross_max_radius", type=float, default=40.0)
    parser.add_argument("--cross_max_neighbors", type=float, default=50)

    # Model
    parser.add_argument("--node_fdim", default=145, type=int)
    parser.add_argument("--edge_fdim", default=0, type=int)
    parser.add_argument("--n_s", type=int, default=20)
    parser.add_argument("--n_v", type=int, default=10)
    parser.add_argument("--activation", default='relu', type=str)
    parser.add_argument("--dropout_p", default=0.1, type=float)
    parser.add_argument("--n_conv_layers", default=1, type=int)
    parser.add_argument("--sh_lmax", type=int, default=2)
    parser.add_argument("--distance_emb_dim", type=int, default=32)

    # Stability
    parser.add_argument("--n_deviations", type=int, default=0)
    parser.add_argument("--deviation_eps", type=float, default=0.1)
    parser.add_argument("--enforce_stability", action='store_true')

    # Training
    parser.add_argument("--n_epochs", default=10, type=int)

    # Optimizer & Scheduler
    parser.add_argument("--optim_name", default='adam', type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--grad_clip_value", default=10.0, type=float, 
                        help="Gradient clipping max value")
    parser.add_argument("--scheduler", type=str, default="plateau")
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--scheduler_mode", type=str, default='min')  
    parser.add_argument("--ema_decay_rate", type=float, default=0.999)

    # Logging
    parser.add_argument("--log_every", default=1000, type=int, 
                        help="Logging frequency")
    parser.add_argument("--eval_every", default=10000, type=int, 
                        help="Evaluation frequency during training.")
    parser.add_argument("--step_every", default=1, type=int,
                        help="How quickly to accumulate gradient steps")

    # Val scheduler based stepping
    parser.add_argument("--lr_sched_metric", default="val_loss", type=str)
    parser.add_argument("--lr_sched_metric_goal", default="min", type=str)

    return parser


def parse_energy_model_args() -> ArgParser:
    parser = setup_parser()

    parser.add_argument("--score_fn_decoys", default="dock_low_res")
    parser.add_argument("--max_tr_decoys", type=float, default=10.0)
    parser.add_argument("--norm_method", type=str, default=None)
    parser.add_argument("--ref_choice", type=str, default="random")

    parser.add_argument("--model", type=str, default="dock_score_hetero")
    parser.add_argument("--mode", type=str, default="base")

    parser.add_argument("--w_energy", default=0.0, type=float)
    parser.add_argument("--w_energy_ref", default=0.0, type=float)
    parser.add_argument("--w_energy_diff", default=1.0, type=float)

    parser.add_argument("--enable_plotting", default=True, action='store_true')

    args = parser.parse_args()
    return args


def parse_score_model_args() -> ArgParser:
    parser = setup_parser()

    parser.add_argument("--transform", type=str, default="ma_noise")
    parser.add_argument("--pert_strategy", type=str, default="all-but-one")
    parser.add_argument("--same_t_for_agent", action='store_true')
    parser.add_argument("--dynamic_max_cross", action='store_true', default=True)
    parser.add_argument("--timepoints_per_complex", type=int, default=1)

    # Model
    parser.add_argument("--model", type=str, default="score")
    parser.add_argument("--cross_cutoff_threshold", type=float, default=40.0)
    parser.add_argument("--center_max_radius", type=float, default=30.0)
    parser.add_argument("--cross_dist_emb_dim", type=int, default=32)
    parser.add_argument("--center_dist_emb_dim", type=int, default=32)
    parser.add_argument("--time_embed_type", type=str, default="sinusoidal")
    parser.add_argument("--time_embed_scale", type=int, default=10000)
    parser.add_argument("--sigma_emb_dim", type=int, default=32)

    parser.add_argument("--tr_sigma_min", type=float, default=0.1)
    parser.add_argument("--tr_sigma_max", type=float, default=19.0)
    parser.add_argument("--rot_sigma_min", type=float, default=0.03)
    parser.add_argument("--rot_sigma_max", type=float, default=1.55)

    parser.add_argument("--scale_by_sigma", action='store_true', default=True)

    parser.add_argument("--w_tr", default=1.0)
    parser.add_argument("--w_rot", default=1.0)

    args = parser.parse_args()
    return args


def update_args_from_config(args: Args) -> Args:
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        args_dict = args.__dict__

        for key, value in config_dict.items():
            if isinstance(value, list):
                if key not in args_dict:
                    args_dict[key] = []

                for v in value:
                    args_dict[key].append(v)
            else:
                args_dict[key] = value

        args.config = args.config.name
    
    return args


def parse_train_args(mode: str = 'energy') -> Args:

    if mode == "energy":
        parse_fn = parse_energy_model_args
    elif mode == "score_matching":
        parse_fn = parse_score_model_args
    else:
        raise ValueError()

    args = parse_fn()

    if args.log_dir is None:
        args.log_dir = CLUSTER_EXP_DIR if IS_CLUSTER else EXP_DIR

    args = update_args_from_config(args)
    return args


def parse_game_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--config", type=FileType(mode='r'), 
                        help="Config file to load args from. args will be overwritten")
    parser.add_argument("--out_dir", type=str, default="game_outputs/")

    parser.add_argument("--complex_dir", type=str, default='complexes')
    parser.add_argument("--complex_list_file", type=str, default='complexes.txt',
                        help="List of complexes to play game on")
    
    #parser.add_argument("--model_dir", type=str, required=True)
    #parser.add_argument("--model_name", type=str, required=True)

    # wandb
    parser.add_argument("--wandb_entity", default=WANDB_ENTITY)
    parser.add_argument("--project_name", default="dockgame-inf-debug")
    parser.add_argument("--group_name", default="gameplay")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--wandb_mode", default="disabled")
    parser.add_argument("--job_type", default=None)
    parser.add_argument("--notes", default=None)

    parser.add_argument("--debug", action='store_true')

    # Common args for game
    parser.add_argument("--dataset", type=str, default="db5")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_rounds", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_vis", action='store_true')
    parser.add_argument("--save_trajectory", action='store_true')
    parser.add_argument("--agent_type", default="protein", 
                        choices=["protein", "chain"])
    parser.add_argument("--score_fn_name", default="dock_low_res", 
                        choices=["dock_low_res", "dock_high_res"])
    parser.add_argument("--n_equilibria", default=10, type=int)
    parser.add_argument("--strategy", default="langevin", 
                        choices=["langevin", "reward_grad"])

    # Langevin specific args
    parser.add_argument("--use_ode", action='store_true')

    # Reward_grad specific args
    parser.add_argument("--perturbation_mag", type=float, default=0.0)
    parser.add_argument("--num_protein_copies", type=int, default=0)
    parser.add_argument("--distance_penalty", type=float, default=0.0)

    args = parser.parse_args()
    args = update_args_from_config(args)
    return args


def count_parameters(model: Model, log_to_wandb: bool = False) -> int:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if log_to_wandb:
        wandb.log({'n_params': n_params})
    return n_params


def read_complexes_from_txt(filename: str) -> list[str]:
    with open(filename, "r") as f:
        print(f"Loading complex ids from {filename}", flush=True)
        pdb_ids = f.readlines()
        pdb_ids = [pdb_id.strip() for pdb_id in pdb_ids]
    return pdb_ids


def launch_experiment(train_fn: Callable, mode: str = 'score_matching') -> Model:
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=4)
    torch.manual_seed(1331323)

    # Load args from command line and replace values with those from config
    print(flush=True)
    args = parse_train_args(mode=mode)

    # Wandb setup
    wandb_setup(args)
    args.wandb_dir = os.path.dirname(wandb.run.dir)

    print(f"Args supplied for the experiment", flush=True)
    print(f"{args}", flush=True)
    print(flush=True)

    log_dir = construct_log_dir(args=args)
    print(f"Run Name: {args.run_name}", flush=True)
    print(flush=True)

    config_file = os.path.join(log_dir, "config_train.yml")
    yaml_dump = yaml.dump(args.__dict__)
    with open(config_file, "w") as f:
        f.write(yaml_dump)

    print(f"Saved model config to {config_file}", flush=True)
    print(flush=True)

    train_loader = build_data_loader(args=args, mode="train")
    val_loader = build_data_loader(args=args, mode="val")

    # Model
    model = build_model_from_args(args)

    if 'restore_from' in args and args.restore_from is not None:
        print(f"Loading pretrained model from {args.restore_from}", flush=True)

        model_dict = load_model_from_args(args=args, return_model_args=False)
        model.load_state_dict(model_dict)

    n_params = count_parameters(model=model, log_to_wandb=False and args.online)
    print(f"Model with {n_params / (10**6)}M parameters", flush=True)
    print(flush=True)

    # Optimizers
    optimizer = get_optimizer(model=model, optim_name=args.optim_name,
                              lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer=optimizer, scheduler_name=args.scheduler,
                              scheduler_mode=args.scheduler_mode, factor=0.7,
                              patience=args.scheduler_patience, min_lr=args.lr / 100)
    ema = get_ema(model=model, decay_rate=args.ema_decay_rate)


    train_fn(args=args, train_loader=train_loader, 
          val_loader=val_loader, model=model, optimizer=optimizer,
          scheduler=scheduler, ema_weights=ema, log_dir=log_dir)

    return model
