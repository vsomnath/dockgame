import os
import yaml
import torch
import wandb
import numpy as np
import copy
import math

from dockgame.training.epoch_fns import (
    train_epoch, validation_epoch, inference_epoch
)
from dockgame.training.losses import loss_fn_from_args
from dockgame.utils.setup import launch_experiment
from dockgame.common.constants import DEVICE


def train(args, train_loader, val_loader, model, optimizer, scheduler, ema_weights=None, log_dir=None):
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_goal == 'min' else 0
    best_epoch = 0
    logs = {'val_loss': math.inf, "val_inference_rmsd": best_val_inference_value}

    loss_fn = loss_fn_from_args(args)

    # Computing a bunch of logs before training
    print("==============================================", flush=True)
    print("Before training", flush=True)
    print("==============================================", flush=True)
    print(flush=True)

    kwargs = {'t_to_sigma_fn': train_loader.dataset.transform.t_to_sigma}

    val_losses = validation_epoch(
        model=model, loader=val_loader, loss_fn=loss_fn, 
        make_outputs=False, model_name=args.model,
        **kwargs
    )

    print_msg = ""
    for item, value in val_losses.items():
        if item == "loss":
            print_msg += f"Valid Loss: {np.round(value, 4)} "
        else:
            print_msg += f"{item}: {np.round(value, 4)} "
    print(print_msg, flush=True)
    

    if args.inference_every is not None:
        inference_metrics = inference_epoch(
            model=model, dataset_orig=val_loader.dataset, args=args
        )
    
        print_msg = f"Inference Metrics: "
        for item, value in inference_metrics.items():
            print_msg += f"{item}: {np.round(value, 4)} "                    
        print(print_msg, flush=True)
        print(flush=True)


    # Write logs to wandb
    if args.wandb_mode == "online":
        # Logging metrics and losses
        log_init = {}
        log_init.update({'val_' + k: v for k, v in val_losses.items()})
        if args.inference_every is not None and args.inference_every > 0:
            log_init.update({'val_inference_' + k: v for k, v in inference_metrics.items()})
        log_init['current_lr'] = optimizer.param_groups[0]['lr']
        log_init["step"] = 0
        wandb.log(log_init)


    print("==============================================", flush=True)
    print(f"Starting training for {args.n_epochs} epochs.", flush=True)
    print("==============================================", flush=True)
    print(flush=True)


    for epoch in range(args.n_epochs):
        log_dict = {}

        kwargs = {'t_to_sigma_fn': train_loader.dataset.transform.t_to_sigma}
        
        # Run an epoch of training and get predictions and metrics
        train_losses = train_epoch(
                model=model, loader=train_loader, 
                optimizer=optimizer, loss_fn=loss_fn, 
                ema_weights=ema_weights,
                grad_clip_value=args.grad_clip_value,
                model_name=args.model,
                step_every=args.step_every,
                **kwargs
            )

        # Print training metrics
        print_msg = f"Epoch {epoch+1}: "
        for item, value in train_losses.items():
            if item == "loss":
                print_msg += f"Train Loss: {np.round(value, 4)} "
            else:
                print_msg += f"{item}: {np.round(value, 4)} "
        logs.update({'train_' + k: v for k, v in train_losses.items()})
        print(print_msg, flush=True)

        # Load ema parameters into model during validation and inference
        if ema_weights is not None:
            # Gather model parameters and store them
            ema_weights.store(model.parameters())
            # copy ema-averaged parameters for validation and inference
            ema_weights.copy_to(model.parameters())
            
        val_losses = validation_epoch(
            model=model, loader=val_loader, loss_fn=loss_fn, 
            make_outputs=False, model_name=args.model,
            **kwargs
        )

        # Print validation metrics
        print_msg = f"Epoch {epoch+1}: "
        for item, value in val_losses.items():
            if item == "loss":
                print_msg += f"Valid Loss: {np.round(value, 4)} "
            else:
                print_msg += f"{item}: {np.round(value, 4)} "
        logs.update({'val_' + k: v for k, v in val_losses.items()})
        print(print_msg, flush=True)

        # Inference on validation set
        if args.inference_every is not None:
            if (epoch + 1) % args.inference_every == 0:
                inference_metrics = inference_epoch(
                    model=model, dataset_orig=val_loader.dataset, args=args
                )
            
                print_msg = f"Epoch {epoch+1}: Inference "
                for item, value in inference_metrics.items():
                    print_msg += f"{item}: {np.round(value, 4)} "                    
                print(print_msg, flush=True)
                logs.update({'val_inference_' + k: v for k, v in inference_metrics.items()})
            print(flush=True)

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

        if args.inference_every is not None:
            if args.inference_metric in logs.keys() and \
                    (args.inference_goal == 'min' and logs[args.inference_metric] < best_val_inference_value or
                    args.inference_goal == 'max' and logs[args.inference_metric] > best_val_inference_value):
                best_val_inference_value = logs[args.inference_metric]
                best_val_inference_epoch = epoch

                if log_dir is not None:
                    model_file = os.path.join(log_dir, 'best_inference_epoch_model.pt')
                    print(f"After best inference, saving model to {model_file}", flush=True)
                    torch.save(model_dict, model_file)

                    if ema_weights is not None:
                        ema_file = os.path.join(log_dir, 'best_ema_inference_epoch_model.pt')
                        print(f"After best inference, saving ema to {ema_file}", flush=True)
                        torch.save(ema_model_state_dict, ema_file)

        # Write logs to wandb
        if args.wandb_mode == "online":
            # Logging metrics and losses
            log_dict.update({'train_' + k: v for k, v in train_losses.items()})
            log_dict.update({'val_' + k: v for k, v in val_losses.items()})
            if args.inference_every is not None: 
                if args.inference_every > 0 and (epoch + 1) % args.inference_every == 0:
                    log_dict.update({'val_inference_' + k: v for k, v in inference_metrics.items()})
            log_dict['current_lr'] = optimizer.param_groups[0]['lr']
            log_dict["step"] = epoch + 1
            wandb.log(log_dict)

        # Updating best validation loss and saving models
        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch

            if log_dir is not None:
                model_file = os.path.join(log_dir, "best_model.pt")
                torch.save(model_dict, model_file)
                print(f"After best validation, saving model to {model_file}", flush=True)
                
                if ema_weights is not None:
                    ema_file = os.path.join(log_dir, "best_ema_model.pt")
                    torch.save(ema_model_state_dict, ema_file)
                    print(f"After best validation, saving ema to {ema_file}", flush=True)
            print(flush=True)

        if scheduler is not None:
            if args.lr_sched_metric in logs:
                scheduler.step(logs[args.lr_sched_metric])
            else:
                scheduler.step(logs["val_loss"])

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
    if args.inference_every is not None and args.inference_every > 0:
        print(f"Best Inference Metric {best_val_inference_value} on Epoch {best_val_inference_epoch}", flush=True)


def main():
    trained_model = launch_experiment(train_fn=train, mode='score_matching')
    return trained_model


if __name__ == "__main__":
    main()
