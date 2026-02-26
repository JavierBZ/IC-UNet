import numpy as np
import torch
import wandb
import time

from monai.transforms import Compose, AsDiscrete
from monai.data import decollate_batch
from monai.metrics import DiceMetric, CumulativeAverage
from monai.inferers import sliding_window_inference


def train_one_epoch(model, dataloader, optimizer, loss_function, evaluation_metric,
                    epoch, log_metrics=False, device="cuda", scheduler=None):
    """
    Runs one training epoch over the given dataloader.

    Args:
        model:              PyTorch model.
        dataloader:         Training dataloader.
        optimizer:          Optimizer.
        loss_function:      Loss function.
        evaluation_metric:  Metric object (unused directly; kept for API consistency).
        epoch:              Current epoch index.
        log_metrics:        If True, compute and log per-class Dice scores to wandb.
        device:             Torch device string.
        scheduler:          LR scheduler (stepped after epoch 1000).

    Returns:
        If log_metrics: (mean_dice, average_loss, per_class_dice_array)
        Otherwise:      average_loss
    """
    post_pred  = Compose([AsDiscrete(argmax=True, threshold=0.5, to_onehot=11)])
    post_label = Compose([AsDiscrete(to_onehot=11)])

    model.train()
    losses = CumulativeAverage()
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    start = time.time()

    for step, batch_data in enumerate(dataloader):
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )

        # Collapse one-hot label channels into a single integer-label volume
        labels = sum(labels[:, ch, :, :, :] * (ch + 1) for ch in range(10))
        labels = labels.unsqueeze(1)

        optimizer.zero_grad()
        # logit_map = sliding_window_inference(
        #     inputs, predictor=model, roi_size=(256, 256, 128), sw_batch_size=1, overlap=0.5
        # )
        logit_map = model(inputs)
        loss = loss_function(logit_map, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss)

        if log_metrics:
            outputs     = [post_pred(i)  for i in decollate_batch(logit_map)]
            true_labels = [post_label(i) for i in decollate_batch(labels)]
            dice_metric_batch(y_pred=outputs, y=true_labels)
        else:
            elapsed = time.time() - start
            print(f"Step [{step}/{len(dataloader)}] Time: {elapsed:.3f}s  Loss: {loss.item():.4f}")

        start = time.time()

    if epoch > 1000:
        scheduler.step()

    average_loss = losses.aggregate().item()
    wandb.log({"loss": average_loss, "lr": scheduler.get_last_lr()[0]}, step=epoch)
    losses.reset()

    if log_metrics:
        metric_batch = dice_metric_batch.aggregate()
        vessel_names = ["BA", "LACA", "LICA", "LMCA", "RACA", "RICA", "RMCA", "RPCA", "LPCA", "NA"]
        per_class = np.array([metric_batch[i + 1].item() for i in range(10)])

        for name, value in zip(vessel_names, per_class):
            wandb.log({f"dice_{name.lower()}": value}, step=epoch)
        mean_dice = float(np.nanmean(per_class))
        wandb.log({"dice_mean": mean_dice}, step=epoch)

        dice_metric_batch.reset()
        evaluation_metric.reset()
        return mean_dice, average_loss, per_class

    return average_loss


def validate(model, dataloader, loss_function, evaluation_metric, epoch,
             device="cuda"):
    """
    Runs validation over the given dataloader and logs metrics to wandb.

    Args:
        model:              PyTorch model.
        dataloader:         Validation dataloader.
        loss_function:      Loss function.
        evaluation_metric:  Metric object (unused directly; kept for API consistency).
        epoch:              Current epoch index.
        device:             Torch device string.

    Returns:
        (mean_dice, average_loss)
    """
    post_pred  = Compose([AsDiscrete(argmax=True, threshold=0.5, to_onehot=11)])
    post_label = Compose([AsDiscrete()])

    model.eval()
    losses = CumulativeAverage()
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    with torch.no_grad():
        for batch_data in dataloader:
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            labels = sum(labels[:, ch, :, :, :] * (ch + 1) for ch in range(10))
            labels = labels.unsqueeze(1)

            logit_map = sliding_window_inference(
                inputs, predictor=model, roi_size=(256, 256, 128), sw_batch_size=1, overlap=0.5
            )
            val_outputs = [post_pred(i) for i in decollate_batch(logit_map)]
            val_labels  = [post_label(i) for i in decollate_batch(labels)]

            loss = loss_function(logit_map, labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)
            losses.append(loss)

    metric_batch = dice_metric_batch.aggregate()
    vessel_names = ["BA", "LACA", "LICA", "LMCA", "RACA", "RICA", "RMCA", "RPCA", "LPCA", "NA"]
    per_class = np.array([metric_batch[i + 1].item() for i in range(10)])

    for name, value in zip(vessel_names, per_class):
        wandb.log({f"val_dice_{name.lower()}": value}, step=epoch)
    mean_dice = float(np.nanmean(per_class))
    wandb.log({"val_dice_mean": mean_dice}, step=epoch)

    average_loss = losses.aggregate().item()
    wandb.log({"val_loss": average_loss}, step=epoch)

    dice_metric_batch.reset()
    return mean_dice, average_loss
