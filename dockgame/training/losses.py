from functools import partial
from typing import Tuple, Optional, Sequence, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import dockgame.utils.so3 as so3

# Type aliases
Tensor = torch.Tensor


def loss_fn_energy(
    energies_pred: Sequence[Tensor],
    energies_true: Sequence[Tensor],
    energy_deviations: Optional[list[Tensor]],
    energy_pred_bound: Optional[Tensor],
    rmsd_true: Optional[Tensor],
    rmsd_true_ref: Optional[Tensor],
    w_energy_diff: float = 1.0,
    w_energy: float = 1.0,
    w_energy_ref: float = 1.0,
    w_stability: float = 1.0,
    loss_type: str = "l2",
):
    """
    Loss function for energy differences (and optionally the true energies)
    Additional options to regularize the norm of gradients at the bound complex.
    """

    if loss_type == "l2":
        energy_criterion = nn.MSELoss()
    elif loss_type == "l1":
        energy_criterion = nn.L1Loss()
    
    energy_pred, energy_pred_ref, energy_pred_diff = energies_pred
    energy_true, energy_true_ref, energy_true_diff = energies_true

    if energy_pred_diff is not None:
        energy_loss_diff = energy_criterion(energy_pred_diff, energy_true_diff)
        loss = w_energy_diff * energy_loss_diff
    
        loss_dict = {"loss": 0.0, "energy_loss_diff": energy_loss_diff.item()}

    if energy_pred_ref is not None:
        energy_loss_ref = energy_criterion(energy_pred_ref, energy_true_ref)
        energy_loss = energy_criterion(energy_pred, energy_true)

        loss = loss + w_energy * energy_loss + \
                w_energy_ref * energy_loss_ref

        loss_dict["energy_loss_ref"] = energy_loss_ref.item()
        loss_dict["energy_loss"] = energy_loss.item()
    
    if w_stability > 0:
        if energy_deviations is not None and energy_pred_bound is not None: 

            if 0: #old stability loss
                stability_loss = torch.stack(
                    [nn.MSELoss()(F.relu(-(energy_pred_dev - energy_pred_bound - 1e-1)), torch.zeros_like(energy_pred_bound))
                    for energy_pred_dev in energy_deviations], dim=0
                )
            elif 0:# negative log-likelihood loss w.r.t bound
                stability_loss = torch.stack(
                    [- F.logsigmoid(energy_pred_dev - energy_pred_bound)
                    for energy_pred_dev in energy_deviations ], dim=0
                )
        if 0: # permute batch (does not work across different proteins)
            idx_permuted = torch.randperm(len(energy_true))
            energy_true_permuted = energy_true[idx_permuted]
            energy_pred_permuted = energy_pred[idx_permuted]
            rmds_true_permuted = rmsd_true[idx_permuted]
            #signs = torch.sign(energy_true - energy_true_permuted)
            signs = torch.sign(rmsd_true - rmds_true_permuted)
            stability_loss = - F.logsigmoid( signs * (energy_pred - energy_pred_permuted))
        else:
            signs = torch.sign(energy_true - energy_true_ref)
            #signs = torch.sign(rmsd_true - rmsd_true_ref)
            stability_loss = - F.logsigmoid( signs * (energy_pred - energy_pred_ref))

        stability_loss = stability_loss.mean()

        #stability_bound_loss = - F.logsigmoid( energy_pred - energy_pred_bound)
        #stability_bound_loss = stability_bound_loss.mean()

        loss = loss + w_stability * stability_loss #(stability_loss + stability_bound_loss)
        loss_dict["stability_loss"] = stability_loss.item()
        #loss_dict["stability_bound_loss"] = stability_bound_loss.item()

    loss_dict["loss"] = loss.item()
    return loss, loss_dict


def score_matching_loss(
    scores_pred: Sequence[Tensor],
    scores_true: Sequence[Tensor],
    sigmas: Sequence[Tensor],
    w_tr: float = 1.0,
    w_rot: float = 1.0,
    apply_mean: bool = True,
) -> Tuple[Tensor, dict[str, Tensor]]:  
    mean_dims = (0, 1) if apply_mean else 1

    tr_score_pred, rot_score_pred = scores_pred
    tr_score_true, rot_score_true = scores_true
    tr_sigma, rot_sigma = sigmas

    # Translation Losses
    tr_sigma = tr_sigma.unsqueeze(-1).cpu()
    tr_loss = (((tr_score_pred.cpu() - tr_score_true) * tr_sigma) ** 2).mean(dim=mean_dims)
    tr_base_loss = ((tr_score_true * tr_sigma) ** 2).mean(dim=mean_dims).detach()

    # Rotation losses
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1)
    rot_loss = (((rot_score_pred.cpu() - rot_score_true) / (rot_score_norm + 1e-5)) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score_true / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    loss = tr_loss * w_tr + rot_loss * w_rot

    loss_dict = {
        "loss": loss.item(),
        "tr_loss": tr_loss.item(),
        "rot_loss": rot_loss.item(),
        "tr_base_loss": tr_base_loss.item(),
        "rot_base_loss": rot_base_loss.item()
    }

    return loss, loss_dict


def loss_fn_from_args(args) -> Callable:

    if args.model in ["dock_score", "dock_score_hetero"]:
        loss_fn = partial(
            loss_fn_energy,
            w_energy_diff=args.w_energy_diff,
            w_energy=args.w_energy,
            w_energy_ref=args.w_energy_ref if 'w_energy_ref' in args else args.w_energy,
            w_stability=args.w_stability,
            loss_type=args.loss_type
        )

        return loss_fn
    
    elif args.model == "score":
        loss_fn = partial(
            score_matching_loss,
            w_tr=args.w_tr,
            w_rot=args.w_rot,
        )

        return loss_fn
