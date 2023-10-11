import os
import yaml
import torch
import wandb
import numpy as np
import copy
import math
import matplotlib.pyplot as plt

from dockgame.models import ALLOWED_MODELS
from dockgame.training.epoch_fns import train_epoch, validation_epoch
from dockgame.training.losses import loss_fn_from_args
from dockgame.utils.setup import launch_experiment
from dockgame.common.constants import DEVICE
from dockgame.analysis.plotting import (
    plot_histogram, plot_scores_vs_ids, plot_diff_scores_sign
)


def train(args, train_loader, val_loader, model, optimizer, scheduler, ema_weights=None, log_dir=None):
    best_val_loss = math.inf
    best_epoch = 0

    # Construct loss function with prescribed args
    loss_fn = loss_fn_from_args(args)

    # Computing a bunch of logs before training
    print("==============================================", flush=True)
    print("Before training", flush=True)
    print("==============================================", flush=True)
    print(flush=True)

    val_losses = validation_epoch(
            model=model, loader=val_loader, loss_fn=loss_fn,
            model_name=args.model,
            make_outputs=False
        )

    # Write logs to wandb
    if args.wandb_mode == "online":
        # Logging metrics and losses
        log_init = {}
        log_init.update({'val_' + k: v for k, v in val_losses.items()})
        log_init['current_lr'] = optimizer.param_groups[0]['lr']
        log_init["step"] = 0
        wandb.log(log_init)
        
    for epoch in range(args.n_epochs):
        log_dict = {}
        
        # Run an epoch of training and get predictions and metrics
        train_losses = train_epoch(
                model=model, loader=train_loader, 
                optimizer=optimizer, loss_fn=loss_fn, 
                ema_weights=ema_weights,
                grad_clip_value=args.grad_clip_value,
                step_every=args.step_every,
                model_name=args.model,
            )

        # Print training metrics
        print_msg = f"Epoch {epoch+1}: "
        for item, value in train_losses.items():
            if item == "loss":
                print_msg += f"Train Loss: {np.round(value, 4)} "
            else:
                print_msg += f"{item}: {np.round(value, 4)} "
        print(print_msg, flush=True)

        # Load ema parameters into model
        if ema_weights is not None:
            ema_weights.store(model.parameters())
            ema_weights.copy_to(model.parameters())

        _, train_outputs = validation_epoch(
            model=model, loader=train_loader, loss_fn=loss_fn,
            model_name=args.model,
            make_outputs=True
        )
            
        val_losses, val_outputs = validation_epoch(
            model=model, loader=val_loader, loss_fn=loss_fn,
            model_name=args.model,
            make_outputs=True
        )

        # Print validation metrics
        print_msg = f"Epoch {epoch+1}: "
        for item, value in val_losses.items():
            if item == "loss":
                print_msg += f"Valid Loss: {np.round(value, 4)} "
            else:
                print_msg += f"{item}: {np.round(value, 4)} "
        print(print_msg, flush=True)

        if ema_weights is not None:
            if args.n_gpus > 1 and DEVICE == 'cuda':
                ema_model_state_dict = copy.deepcopy(model.module.state_dict())
            else:
                ema_model_state_dict = copy.deepcopy(model.state_dict())
            # Restore model back to original parameters after validation
            ema_weights.restore(model.parameters())

        if args.n_gpus > 1 and DEVICE == 'cuda':
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()
        
        if args.enable_plotting:
            
            if args.model in ALLOWED_MODELS:

                # Histograms for train and validation
                hist_train_diff = plot_histogram(
                    y_pred=train_outputs["diff_pred"], y_true=train_outputs["diff_true"],
                    bins=50, show=False
                )

                hist_val_diff = plot_histogram(
                        y_pred=val_outputs["diff_pred"], y_true=val_outputs["diff_true"],
                        bins=50, show=False
                )

                # Scatter plots of diff scores for train and valid
                scatter_plot_train_diff = plot_scores_vs_ids(
                    y_pred=train_outputs["diff_pred"],
                    y_true=train_outputs["diff_true"],
                    show=False
                )

                scatter_plot_val_diff = plot_scores_vs_ids(
                    y_pred=val_outputs["diff_pred"],
                    y_true=val_outputs["diff_true"],
                    show=False
                )

                # Scatter plots of diff scores for train and valid
                idx_sort_rmds = np.argsort(train_outputs["diff_rmsd_true"], axis=None)
                scatter_plot_train_rmsd_diff = plot_scores_vs_ids(
                    y_pred=train_outputs["diff_pred"][idx_sort_rmds],
                    y_true=train_outputs["diff_rmsd_true"][idx_sort_rmds], 
                    y_true_unord=train_outputs["diff_rmsd_true"],
                    show=False
                )
                idx_sort_rmds = np.argsort(val_outputs["diff_rmsd_true"], axis=None)
                scatter_plot_val_rmds_diff = plot_scores_vs_ids(
                    y_pred=val_outputs["diff_pred"][idx_sort_rmds],
                    y_true=val_outputs["diff_rmsd_true"][idx_sort_rmds],
                    y_true_unord=val_outputs["diff_rmsd_true"],
                    show=False
                )

                # Scatter plots of signs of diff scores
                scatter_plot_train_diff_sign = plot_diff_scores_sign(
                    y_pred=np.sign(train_outputs["diff_pred"]),
                    y_true=np.sign(train_outputs["diff_true"]),
                    x_values=train_outputs["diff_true"], 
                    show=False
                )

                scatter_plot_val_diff_sign = plot_diff_scores_sign(
                    y_pred=np.sign(val_outputs["diff_pred"]),
                    y_true=np.sign(val_outputs["diff_true"]),
                    x_values=val_outputs["diff_true"],  
                    show=False
                )

                if "pred" in train_outputs:
                    hist_train = plot_histogram(
                        y_pred=train_outputs["pred"], y_true=train_outputs["true"],
                        bins=50, show=False
                    )

                    scatter_plot_train = plot_scores_vs_ids(
                        y_pred=train_outputs["pred"],
                        y_true=train_outputs["true"],
                        show=False
                    )

                if "pred" in val_outputs:
                    hist_val = plot_histogram(
                        y_pred=val_outputs["pred"], y_true=val_outputs["true"],
                        bins=50, show=False
                    )

                    scatter_plot_val = plot_scores_vs_ids(
                        y_pred=val_outputs["pred"],
                        y_true=val_outputs["true"], 
                        show=False
                    )

        # Write logs to wandb
        if args.wandb_mode == "online":
            # Logging metrics and losses
            log_dict.update({'train_' + k: v for k, v in train_losses.items()})
            log_dict.update({'val_' + k: v for k, v in val_losses.items()})
            log_dict['current_lr'] = optimizer.param_groups[0]['lr']
            
            if args.enable_plotting:
                # Histograms of differences & absolute scores for train

                if args.model in ALLOWED_MODELS:

                    log_dict["train_score_diff_vs_ids"] = wandb.Image(scatter_plot_train_diff, caption="Train Diff vs IDs")
                    log_dict["val_score_diff_vs_ids"] = wandb.Image(scatter_plot_val_diff, caption="Val Diff vs IDs")
                    log_dict["train_score_rmsd_diff_vs_ids"] = wandb.Image(scatter_plot_train_rmsd_diff, caption="Train RMSD Diff vs IDs")
                    log_dict["val_score_rmsd_diff_vs_ids"] = wandb.Image(scatter_plot_val_rmds_diff, caption="Val RMSD Diff vs IDs")

                    log_dict["train_scores_hist_diff"] = wandb.Image(hist_train_diff, caption="Train Hist Diff")
                    log_dict["val_scores_hist_diff"] = wandb.Image(hist_val_diff, caption="Val Hist Diff")

                    log_dict["train_score_diff_sign_vs_true_diff"] = wandb.Image(scatter_plot_train_diff_sign, 
                                                                                caption="Sign of Train Diff vs true diff")
                    log_dict["val_score_diff_sign_vs_true_diff"] = wandb.Image(scatter_plot_val_diff_sign, 
                                                                            caption="Sign of Val Diff vs true diff")

                    if "pred" in train_outputs:
                        log_dict["train_scores_hist"] = wandb.Image(hist_train, caption="Train Hist")
                        log_dict["val_scores_hist"] = wandb.Image(hist_val, caption="Val Hist")

                        log_dict["train_score_vs_ids"] = wandb.Image(scatter_plot_train, caption="Train Score vs IDs")
                        log_dict["val_score_vs_ids"] = wandb.Image(scatter_plot_val, caption="Val Score vs IDs")

            log_dict["step"] = epoch + 1
            wandb.log(log_dict)

            plt.close('all')

        else:
            if args.enable_plotting:

                if args.model in ALLOWED_MODELS:
                    hist_train_diff.savefig(f"{log_dir}/hist_train_diff_{epoch+1}.png")
                    hist_val_diff.savefig(f"{log_dir}/hist_val_diff.png")

                    scatter_plot_train_diff.savefig(f"{log_dir}/scatter_train_diff_{epoch+1}.png")
                    scatter_plot_val_diff.savefig(f"{log_dir}/scatter_val_diff_{epoch+1}.png")
                    scatter_plot_train_rmsd_diff.savefig(f"{log_dir}/scatter_train_rmsd_diff_{epoch+1}.png")
                    scatter_plot_val_rmds_diff.savefig(f"{log_dir}/scatter_val_rmsd_diff_{epoch+1}.png")

                    scatter_plot_train_diff_sign.savefig(f"{log_dir}/scatter_train_diff_sign_{epoch+1}.png")
                    scatter_plot_val_diff_sign.savefig(f"{log_dir}/scatter_val_diff_sign_{epoch+1}.png")
                    
                    if "pred" in train_outputs:
                        hist_train.savefig(f"{log_dir}/hist_train_{epoch+1}.png")
                        hist_val.savefig(f"{log_dir}/hist_val.png")

                        scatter_plot_train.savefig(f"{log_dir}/scatter_train_{epoch+1}.png")
                        scatter_plot_val.savefig(f"{log_dir}/scatter_val_{epoch+1}.png")

                plt.close('all')

        model_dict = model.state_dict()

        # Updating best validation loss and saving models
        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch

            if log_dir is not None:
                model_file = os.path.join(log_dir, "best_model.pt")
                print(f"After best validation, saving model to {model_file}", flush=True)
                torch.save(model_dict, model_file)
                
                if ema_weights is not None:
                    ema_file = os.path.join(log_dir, "best_ema_model.pt")
                    print(f"After best validation, saving ema to {ema_file}", flush=True)
                    torch.save(ema_model_state_dict, ema_file)
    
            print(flush=True)

        if scheduler is not None:
            if args.lr_sched_metric in log_dict:
                scheduler.step(log_dict[args.lr_sched_metric])

        if log_dir is not None:
            print(f"Saving last model to {log_dir}/last_model.pt", flush=True)
            save_dict = {
                'epoch': epoch,
                'model': model_dict,
                'optimizer': optimizer.state_dict(),
            }
            if ema_weights is not None:
                save_dict['ema_weights'] = ema_weights.state_dict()

            torch.save(save_dict, os.path.join(log_dir, 'last_model.pt'))
            print(flush=True)

    print(f"Best Validation Loss {best_val_loss} on Epoch {best_epoch+1}", flush=True)


def main():
    trained_model = launch_experiment(train_fn=train, mode='energy')
    return trained_model


if __name__ == "__main__":
    main()
