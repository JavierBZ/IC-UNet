import json
import os
import time
import random
import argparse

import numpy as np
import torch
import yaml
import wandb

from monai.data import CacheDataset, ThreadDataLoader, DataLoader
from torch.optim.lr_scheduler import LinearLR

from src.dataloader import transformations
from src.losses import loss_funtions
from src.optimizers import optimizers
from src.metrics import evaluation_metrics
from src.models import neural_network

from train import train_one_epoch, validate
from test import validation_IC


torch.manual_seed(10)
random.seed(10)
np.random.seed(10)


def calculate_loss_weights(epoch, beta, gamma):
    """
    Returns (lambda_dice, lambda_ce) weights that shift from Dice-heavy to
    CE-heavy as training progresses from epoch *beta* to epoch *gamma*.
    """
    if epoch < beta:
        return 1.0, 0.0
    if epoch >= gamma:
        return 0.1, 0.9
    alpha = 0.8 * (1 - (epoch - beta) / (gamma - beta)) + 0.1
    return alpha, 1 - alpha


def run(config=None):
    wandb.init(
        settings=wandb.Settings(start_method="thread", console="off"),
        config=config,
        mode=in_config["wandb"]["mode"],
        project=in_config["wandb"]["project_name"],
        name=f"{in_config['wandb']['name']}-fold_{in_config['fold']}",
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model            = neural_network(in_config).to(device)
    evaluation_metric = evaluation_metrics(in_config)
    loss_function, beta, gamma, _ = loss_funtions(
        in_config, in_config["beta"], in_config["gamma"], in_config["epochs"]
    )
    optimizer = optimizers(model, config)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=3000)

    train_files = [
        {"image": images[key], "label": labels[key]}
        for key in sets["train"]
    ]
    valid_files = [
        {"image": images[key], "label": labels[key]}
        for key in sets["test"]
    ]

    train_transforms, val_transforms, minimal_transforms, coor_transforms = transformations(in_config)

    train_dataset  = CacheDataset(data=train_files, transform=train_transforms,  cache_rate=1.0, num_workers=16, copy_cache=False)
    val_dataset    = CacheDataset(data=valid_files, transform=val_transforms,    cache_rate=1.0, num_workers=5,  copy_cache=False)
    val_min_dataset = CacheDataset(data=valid_files, transform=minimal_transforms, cache_rate=1.0, num_workers=5,  copy_cache=False)

    train_loader = ThreadDataLoader(train_dataset, batch_size=config["batch_size"],      num_workers=16, shuffle=True)
    val_loader   = ThreadDataLoader(val_dataset,   batch_size=config["batch_size_test"], num_workers=5,  shuffle=False)
    val_min_loader = DataLoader(val_min_dataset,   batch_size=config["batch_size_test"], num_workers=5,  shuffle=False)

    average_loss_test = 0.0
    metric_train      = 0.0
    metric_test       = 0.0

    for epoch in range(config["epochs"]):
        start = time.time()

        log_metrics = (epoch % 25 == 0)

        if log_metrics:
            metric_train, average_loss_train, _ = train_one_epoch(
                model, train_loader, optimizer, loss_function, evaluation_metric,
                epoch, log_metrics=True, device=device, scheduler=scheduler
            )
        else:
            average_loss_train = train_one_epoch(
                model, train_loader, optimizer, loss_function, evaluation_metric,
                epoch, log_metrics=False, device=device, scheduler=scheduler
            )

        if epoch % 25 == 0:
            metric_test, average_loss_test = validate(
                model, val_loader, loss_function, evaluation_metric, epoch, device=device
            )

        elapsed = time.time() - start

        if log_metrics:
            print(
                f"Epoch [{epoch}/{config['epochs']}]  Time: {elapsed:.2f}s  "
                f"Loss Train/Val: [{average_loss_train:.4f} / {average_loss_test:.4f}]  "
                f"Dice Train/Val: [{metric_train:.4f} / {metric_test:.4f}]"
            )
        else:
            print(
                f"Epoch [{epoch}/{config['epochs']}]  Time: {elapsed:.2f}s  "
                f"Loss Train: {average_loss_train:.4f}"
            )

        if epoch % 1000 == 0:
            checkpoint_path = f"fold_{in_config['fold']}_{epoch}_epoch_model.pt"
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, checkpoint_path))

    # Save final model
    model_path = f"fold_{in_config['fold']}_last_model.pt"
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, model_path))

    # Reload and evaluate
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else "cpu"
    full_model_path = os.path.join(wandb.run.dir, model_path)

    if os.path.exists(full_model_path):
        last_model = neural_network(in_config)
        last_model.load_state_dict(torch.load(full_model_path, map_location=map_location))

        if in_config["classes"] == 11:
            validation_IC(
                in_config["model_evaluation_output"], last_model,
                val_loader, loss_function, evaluation_metric, epoch,
                log=True, fold=in_config["fold"], device=device,
                valid_files=valid_files, val_transforms=val_transforms,
                dataloader_min=val_min_loader, minimal_transforms=minimal_transforms,
                coor_transforms=coor_transforms, config=in_config,
            )

        artifact = wandb.Artifact(
            "last-model", type="last-model",
            description="last trained model",
            metadata=dict(config),
        )
        artifact.add_file(full_model_path)
        wandb.run.log_artifact(artifact)

    wandb.finish()


def evaluate(config=None, ensemble=False):
    """Load saved model weights and run evaluation on the test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluation_metric = evaluation_metrics(in_config)
    loss_function, _, _, _ = loss_funtions(
        in_config, in_config["beta"], in_config["gamma"], in_config["epochs"]
    )

    valid_files = [
        {"image": images[key], "label": labels[key]}
        for key in sets["test"]
    ]

    _, val_transforms, minimal_transforms, coor_transforms = transformations(in_config)

    val_dataset     = CacheDataset(data=valid_files, transform=val_transforms,    cache_rate=1.0, num_workers=0, copy_cache=False)
    val_min_dataset = CacheDataset(data=valid_files, transform=minimal_transforms, cache_rate=1.0, num_workers=0, copy_cache=False)
    val_loader      = DataLoader(val_dataset,     batch_size=config["batch_size_test"], num_workers=0, shuffle=False)
    val_min_loader  = DataLoader(val_min_dataset, batch_size=config["batch_size_test"], num_workers=0, shuffle=False)

    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else "cpu"

    if ensemble:
        model = [neural_network(in_config) for _ in in_config["model_evaluation_path"]]
        for m, path in zip(model, in_config["model_evaluation_path"]):
            m.load_state_dict(torch.load(path, map_location=map_location))
    else:
        model = neural_network(in_config)
        model.load_state_dict(torch.load(in_config["model_evaluation_path"], map_location=map_location))

    if in_config["classes"] == 11:
        validation_IC(
            in_config["model_evaluation_output"], model,
            val_loader, loss_function, evaluation_metric, 0,
            log=False, fold=in_config["fold"], device=device,
            valid_files=valid_files, val_transforms=val_transforms,
            dataloader_min=val_min_loader, minimal_transforms=minimal_transforms,
            coor_transforms=coor_transforms, config=in_config, ensemble=ensemble,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cerebral artery segmentation – training / evaluation")
    parser.add_argument("fold", type=int, help="Cross-validation fold index (0–4)")
    parser.add_argument("--config-route", type=str, default="config/config_unet.yml",
                        help="Path to the YAML config file")
    parser.add_argument("--eval", action="store_true", help="Run evaluation instead of training")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use all five fold models as an ensemble during evaluation")
    args = parser.parse_args()

    if not (0 <= args.fold <= 4):
        parser.error("fold must be between 0 and 4 inclusive")

    with open(args.config_route, "r") as f:
        config = yaml.safe_load(f)

    route_paths = config["dataset"]["route_paths"]

    with open(f"{route_paths}_fold_{args.fold}/ids_data.json")    as f: sets   = json.load(f)
    with open(f"{route_paths}_fold_{args.fold}/labels.json")      as f: labels = json.load(f)
    with open(f"{route_paths}_fold_{args.fold}/image_paths.json") as f: images = json.load(f)

    in_config          = config["base"]
    in_config["fold"]  = args.fold

    if args.eval:
        # run_ids maps each fold to the specific wandb run that holds its saved model.
        # Set these in the config under base.wandb.run_ids (list of 5 IDs, fold 0 -> 4)
        # so they can be updated without modifying this script.
        run_ids    = in_config["wandb"]["run_ids"]   # e.g. ["275563b0", "0989k3jy", ...]
        folds      = list(range(len(run_ids)))        # [0, 1, 2, 3, 4]
        wandb_dirs = [d for d in os.listdir("wandb/") if os.path.isdir(os.path.join("wandb/", d))]

        in_config["model_evaluation_path"] = [] if args.ensemble else None

        for run_id, fold_idx in zip(run_ids, folds):
            model_filename = f"fold_{fold_idx}_last_model.pt"
            matched = False
            for directory in wandb_dirs:
                if run_id not in directory:
                    continue
                candidate = os.path.join("wandb", directory, "files", model_filename)
                if not os.path.exists(candidate):
                    raise FileNotFoundError(
                        f"Run '{run_id}' found but '{model_filename}' is missing. "
                        f"Ensure fold {fold_idx} finished training."
                    )
                if args.ensemble:
                    in_config["model_evaluation_path"].append(candidate)
                elif fold_idx == args.fold:
                    in_config["model_evaluation_path"] = candidate
                matched = True
                break

            if not matched:
                raise FileNotFoundError(
                    f"No wandb directory found for run ID '{run_id}' (fold {fold_idx}). "
                    f"Check that base.wandb.run_ids in the config is correct."
                )

        evaluate(in_config, ensemble=args.ensemble)

    elif config["base"]["do_sweep"]:
        sweep_config = config["sweep"]
        sweep_id = wandb.sweep(sweep=sweep_config, project=in_config["wandb"]["project_name"])
        wandb.agent(sweep_id, function=run, count=in_config["wandb"]["sweep_count"])

    else:
        run(in_config)
