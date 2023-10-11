import argparse
from collections import defaultdict
import copy
import traceback
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
from tqdm import tqdm

from dockgame.data import BaseDataset
from dockgame.utils.ops import to_numpy
from dockgame.common.constants import DEVICE
from dockgame.models import ALLOWED_MODELS

from dockgame.game import DockingEngine, get_strategy_from_args
from dockgame.analysis.metrics import (
    compute_complex_rmsd_torch,
)

# Type aliases
Tensor = torch.Tensor
Loader = Union[DataLoader, DataListLoader]
StepFnOutputs = tuple[Tensor, dict[str, Tensor], Optional[dict[str, list]]]
Args = argparse.Namespace


class ProgressMonitor:

    def __init__(self, metric_names: Optional[list[str]] = None):
        if metric_names is not None:
            self.metric_names = metric_names
            self.metrics = {metric: 0.0 for metric in self.metric_names}
        self.count = 0

    def add(self, metric_dict: dict[str, float], batch_size: int = None):
        if not hasattr(self, 'metric_names'):
            self.metric_names = list(metric_dict.keys())
            self.metrics = {metric: 0.0 for metric in self.metric_names}

        self.count += (1 if batch_size is None else batch_size)

        for metric_name, metric_value in metric_dict.items():
            if metric_name not in metric_dict:
                self.metrics[metric_name] = 0.0
                self.metric_names.append(metric_name)
            
            self.metrics[metric_name] += metric_value * (1 if batch_size is None else batch_size)

    def summarize(self) -> dict[str,float]:
        return {k: np.round(v / self.count, 4) for k, v in self.metrics.items()}


def prepare_gt_outputs_energy(data: Data, device: str = 'cpu') -> dict[str, Tensor]:
    energy_true = torch.stack([d.y for d in data], dim=0).unsqueeze(1) \
            if isinstance(data, list) else data.y.unsqueeze(1)
    
    energy_true_ref = torch.stack([d.y_ref for d in data], dim=0).unsqueeze(1) \
            if isinstance(data, list) else data.y_ref.unsqueeze(1)
    
    energy_true_diff = torch.stack([d.y_diff for d in data], dim=0).unsqueeze(1) \
            if isinstance(data, list) else data.y_diff.unsqueeze(1)
    
    rmsd_true = torch.stack([d.rmsd for d in data], dim=0).unsqueeze(1) \
            if isinstance(data, list) else data.rmsd.unsqueeze(1)
    
    rmsd_true_ref = torch.stack([d.rmsd for d in data], dim=0).unsqueeze(1) \
            if isinstance(data, list) else data.rmsd_ref.unsqueeze(1)
    
    rmsd_true_diff = torch.stack([d.rmsd_diff for d in data], dim=0).unsqueeze(1) \
            if isinstance(data, list) else data.rmsd_diff.unsqueeze(1)
    
    energy_true = energy_true.to(device=device)
    energy_true_ref = energy_true_ref.to(device=device)
    energy_true_diff = energy_true_diff.to(device=device)
    rmsd_true = rmsd_true.to(device=device)
    rmsd_true_ref = rmsd_true_ref.to(device=device)
    rmsd_true_diff = rmsd_true_diff.to(device=device)

    gt_outputs = {
        'energy_true': energy_true,
        'energy_true_ref': energy_true_ref,
        'energy_true_diff': energy_true_diff,
        'rmsd_true': rmsd_true,
        'rmsd_true_ref': rmsd_true_ref,
        'rmsd_true_diff': rmsd_true_diff
    }
    return gt_outputs


# TODO: This should be removed
def prepare_outputs_classifier(data, device: str = 'cpu'):
    y_true = torch.stack([d.y for d in data], dim=0).unsqueeze(1) \
            if isinstance(data, list) else data.y.unsqueeze(1)
    y_true = y_true.to(device=device)
    return y_true


def prepare_gt_outputs_score_model(
    data: Data, 
    t_to_sigma: Callable, 
    device: str = 'cpu'
) -> dict[str, tuple[Tensor, Tensor]]:
    tr_score = torch.cat(
        [d.tr_score for d in data], dim=0) \
            if isinstance(data, list) else data.tr_score
    rot_score = torch.cat([d.rot_score for d in data], dim=0) \
          if isinstance(data, list) else data.rot_score

    # tr_score = tr_score.to(device=device)
    # rot_score = rot_score.to(device=device)

    if isinstance(data, list):
        tr_sigma_list, rot_sigma_list = zip(*[t_to_sigma(d.t_tr, d.t_rot) 
                                              for d in data])
        tr_sigma = torch.cat(tr_sigma_list, dim=0)
        rot_sigma = torch.cat(rot_sigma_list, dim=0)
    else:
        tr_sigma, rot_sigma = t_to_sigma(data.t_tr, data.t_rot)
    
    # tr_sigma = tr_sigma.to(device=device)
    # rot_sigma = rot_sigma.to(device=device)

    scores_true = (tr_score, rot_score)
    sigmas = (tr_sigma, rot_sigma)
    gt_outputs = {
        'scores_true': scores_true,
        'sigmas': sigmas
    }
    return gt_outputs


# ==============================================================================
# Custom step_fns for different models 
# ==============================================================================

def step_fn_energy(
    model: torch.nn.Module,
    data: Data, 
    loss_fn: Callable,
    outputs: Optional[dict[str, list]] = None,
    **kwargs
) -> StepFnOutputs:
    """Step function for the energy model."""

    model_name = kwargs['model_name']

    if model_name in ALLOWED_MODELS:
        energies_pred, stability_pred = model(data)
        gt_outputs = prepare_gt_outputs_energy(
            data, device=energies_pred[0].device)
        energy_deviations, energy_pred_bound = stability_pred

        energy_pred, energy_pred_ref, energy_pred_diff = energies_pred
        energies_true = (
            gt_outputs['energy_true'],
            gt_outputs['energy_true_ref'],
            gt_outputs['energy_true_diff']
        )

        if outputs is not None:
            # true RMSD and RMSD diff
            outputs["rmsd_true"].extend(to_numpy(gt_outputs['rmsd_true']))
            outputs["diff_rmsd_true"].extend(
                to_numpy(gt_outputs['rmsd_true_diff']))

            # Energy differences pred vs true
            outputs["diff_pred"].extend(to_numpy(energy_pred_diff))
            outputs["diff_true"].extend(
                to_numpy(gt_outputs['energy_true_diff']))

            if energy_pred is not None:
                # Energy pred vs true
                outputs["pred"].extend(to_numpy(energy_pred))
                outputs["true"].extend(to_numpy(gt_outputs['energy_true']))

        loss, loss_dict = loss_fn(
            energies_pred=energies_pred,
            energies_true=energies_true,
            energy_deviations=energy_deviations,
            energy_pred_bound=energy_pred_bound,
            rmsd_true=gt_outputs['rmsd_true'],
            rmsd_true_ref=gt_outputs['rmsd_true_ref']
        )

    else:
        raise ValueError(f"Model with name {model_name} is not supported.")
    
    return loss, loss_dict, outputs


def step_fn_score(
    model: torch.nn.Module,
    data: Data, 
    loss_fn: Callable,
    outputs=None,
    **kwargs
) -> StepFnOutputs:  
    """Step function for the score model."""
    t_to_sigma_fn = kwargs['t_to_sigma_fn']
    tr_pred, rot_pred = model(data)
    gt_outputs = \
        prepare_gt_outputs_score_model(
            data, t_to_sigma=t_to_sigma_fn, device=tr_pred.device)

    loss, loss_dict = loss_fn(
        scores_pred=(tr_pred, rot_pred),
        scores_true=gt_outputs['scores_true'],
        sigmas=gt_outputs['sigmas']
    )

    return loss, loss_dict, outputs


# ==============================================================================
# Train, validation and inference epoch fns
# ==============================================================================


def train_epoch(
    model: torch.nn.Module,
    loader: Loader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    ema_weights=None,
    model_name: str = 'dock_reward',
    grad_clip_value: float = 10.0,
    step_every: int = 1,
    **kwargs) -> dict[str, float]:
    """Training epoch fn. Involves repeated calls to step_fn defined by model."""

    model.train()
    monitor = ProgressMonitor()
    grad_monitor = ProgressMonitor()

    optimizer.zero_grad()

    kwargs['model_name'] = model_name
    train_outputs = None

    for idx, data in enumerate(tqdm(loader, total=len(loader))):
        if not isinstance(data, list):
            data = data.to(DEVICE)
        
        if model_name == "score":
            train_step_fn = step_fn_score
        else:
            train_step_fn = step_fn_energy

        try:
            loss, loss_dict, train_outputs = train_step_fn(
                model=model, data=data, loss_fn=loss_fn,
                outputs=train_outputs,
                **kwargs
            )

            if step_every > 0:
                loss /= step_every
                loss.backward()

            if idx % step_every == 0:
                if grad_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                
                grads = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grads.append(param.grad.norm().item())
                grad_mean = torch.tensor(grads).mean()
                grad_monitor.add({'grads': grad_mean.item()})

                optimizer.step()
                optimizer.zero_grad()

            monitor.add(loss_dict)
            
            if ema_weights is not None:
                ema_weights.update(model.parameters())
            
        except Exception as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                print(e)
                traceback.print_exc()
                continue
    
    train_losses = monitor.summarize()
    grads_summary = grad_monitor.summarize()
    train_losses['grads'] = grads_summary['grads']
    return train_losses


def validation_epoch(
    model: torch.nn.Module,
    loader: Loader, 
    loss_fn: Callable,
    make_outputs: bool = False,
    model_name: str = 'dock_reward',
    **kwargs,
) -> dict[str, float]:
    """Validation epoch. Involves repeated calls to step_fn defined by model."""

    model.eval()
    monitor = ProgressMonitor()

    val_outputs = None
    if make_outputs and model_name in ALLOWED_MODELS:
        val_outputs = defaultdict(list)
    
    kwargs["model_name"] = model_name

    for idx, data in enumerate(tqdm(loader, total=len(loader))):
        
        try:
            with torch.no_grad():
                if not isinstance(data, list):
                    data = data.to(DEVICE)

                if model_name == 'score':
                    val_step_fn = step_fn_score
                else:
                    val_step_fn = step_fn_energy

                _, loss_dict, val_outputs = val_step_fn(
                    model=model, data=data, outputs=val_outputs,
                    loss_fn=loss_fn, **kwargs
                )

                monitor.add(loss_dict)

        except Exception as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                print(e)
                traceback.print_exc()
                continue

    val_losses = monitor.summarize()
    
    if make_outputs and val_outputs is not None:
        for key, output in val_outputs.items():
            val_outputs[key] = np.asarray(output)

        # Sort by true scores
        if model_name in ALLOWED_MODELS:
            sorted_indices = np.argsort(val_outputs["diff_true"], axis=None)

            for key, output in val_outputs.items():
                val_outputs[key] = output[sorted_indices]
    
        return val_losses, val_outputs
    return val_losses


def inference_epoch(
    model: torch.nn.Module, 
    dataset_orig: BaseDataset, 
    args: Args
) -> dict[str, float]:
    """Inference epoch. Calls the DockingEngine to play the game."""

    id_list = dataset_orig.complexes_split
    if 'num_inference_complexes' in args:
        id_list = id_list[:args.num_inference_complexes]
    
    if 'inference_multiplicity' in args:
        id_list = id_list * args.inference_multiplicity
    else:
        id_list = id_list * 10

    # Setup strategy and docking engine
    strategy = get_strategy_from_args(
        strategy_type="langevin" if args.model == "score" else "reward_grad",
        model=model.module if DEVICE == 'cuda' and args.n_gpus > 1 else model, 
        model_args=args,
        n_rounds=args.inference_steps, 
        ode=args.use_ode if 'ode' in args else False,
        distance_penalty=args.distance_penalty \
            if 'distance_penalty' in args else 0.0,
        device=DEVICE
    )

    engine = DockingEngine(
        strategy=strategy,
        n_rounds=args.inference_steps, 
        agent_type=args.agent_type, 
        debug=False, log_every=None
    )

    rmsd_pdbs = defaultdict(list)
    rmsds = []

    for pdb_id in tqdm(id_list, total=len(id_list)):
        try:
            data = torch.load(f"{dataset_orig.full_processed_dir}/{pdb_id}.pt")
            data.pdb_id = pdb_id
            complex_graph = copy.deepcopy(data)
            complex_graph = complex_graph.to(DEVICE)

            protein_dict, _ = engine.play(
                data=complex_graph, agent_params=None, 
                player_keys=data.agent_keys[:-1],
                monitor=None
            )

            pos_bound_list = [
                data[key].pos_bound.to(DEVICE) for key in data.agent_keys
            ]
            pos_bound_pred_list = [protein_dict[key].pos for key in data.agent_keys]
        
            pos_bound = torch.cat(pos_bound_list, dim=0)
            pos_bound_pred = torch.cat(pos_bound_pred_list, dim=0)

            complex_rmsd, _ = compute_complex_rmsd_torch(
                    complex_pred=pos_bound_pred,
                    complex_true=pos_bound
                )
        
            rmsd_pdbs[pdb_id].append(complex_rmsd.item())
            rmsds.append(complex_rmsd.item())

            del complex_graph
        except Exception as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                print(e)
                traceback.print_exc()
                continue

    rmsds = np.asarray(rmsds)
    min_rmsds = np.asarray([min(rmsd_pdbs[pdb]) for pdb in rmsd_pdbs])

    metrics = {}
    
    for threshold in [2.0, 5.0, 10.0, 15.0, 20.0]:
        metrics[f'rmsd_lt{int(threshold)}'] \
            = (100 * (rmsds < threshold).sum() / len(rmsds))
    
    for threshold in [2.0, 5.0, 10.0, 15.0, 20.0]:
        metrics[f'min_rmsd_lt{int(threshold)}'] \
            = (100 * (min_rmsds < threshold).sum() / len(min_rmsds))
    
    metrics['mean_rmsd'] = np.mean(rmsds)
    metrics['std_rmsd'] = np.std(rmsds)
    metrics['median_rmsd'] = np.median(rmsds)

    return metrics
