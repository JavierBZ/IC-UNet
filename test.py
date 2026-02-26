import os
import time
import pickle

import numpy as np
import torch
import nrrd
import pandas as pd
import wandb

from tqdm import tqdm
from scipy.spatial import cKDTree

from monai.data import decollate_batch, MetaTensor
from monai.metrics import DiceMetric, CumulativeAverage
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    AsDiscrete,
    CropForegroundd,
    Resized,
    ResizeWithPadOrCropd,
    Spacingd,
    Affined,
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
)

from src.dataloader import transformations, ConvertToMultiChannelBasedOnICClassesd


# ---------------------------------------------------------------------------
# Coordinate mapping utilities
# ---------------------------------------------------------------------------

def find_k_closest_points(original, transformed, k=1, chunk_size=10_000):
    """
    For each point in *original*, find the *k* nearest neighbours in *transformed*
    using a cKDTree, processing in chunks to limit memory usage.

    Args:
        original    (np.ndarray): Shape [N, 3].
        transformed (np.ndarray): Shape [M, 3].
        k           (int):        Number of nearest neighbours.
        chunk_size  (int):        Number of query points per chunk.

    Returns:
        k_closest_indices   (np.ndarray): Shape [N, k].
        k_closest_distances (np.ndarray): Shape [N, k].
    """
    num_orig = original.shape[0]
    k_closest_indices   = np.zeros((num_orig, k), dtype=np.int64)
    k_closest_distances = np.full((num_orig, k), float("inf"))

    tree = cKDTree(transformed)

    for i in tqdm(range(0, num_orig, chunk_size), desc="KDTree lookup"):
        chunk = original[i : i + chunk_size]
        distances, indices = tree.query(chunk, k=k)
        if k == 1:
            k_closest_indices[i : i + chunk_size]   = np.expand_dims(indices,   axis=-1)
            k_closest_distances[i : i + chunk_size] = np.expand_dims(distances, axis=-1)
        else:
            k_closest_indices[i : i + chunk_size]   = indices
            k_closest_distances[i : i + chunk_size] = distances

    return k_closest_indices, k_closest_distances


def compute_final_labels(batch_orig, transform_label, transforms, device, use_GT=True, tta=False):
    """
    Map predicted labels from the transformed space back to the original image space
    using nearest-neighbour coordinate lookup.

    Args:
        batch_orig      (dict):         Original (untransformed) sample dict with 'image' and 'label'.
        transform_label (torch.Tensor): Predicted multi-channel label in the transformed space, shape [C, H, W, D].
        transforms      (Compose):      Coordinate transform pipeline.
        device          (torch.device): Target device for output tensor.
        use_GT          (bool):         If True, restrict mapping to foreground voxels only.
        tta             (bool):         If True, return a compact [C, N_fg] tensor instead of a full volume.

    Returns:
        closest_labels (torch.Tensor): Mapped label tensor.
    """
    dimensions = batch_orig["image"].shape[1:]
    x = np.linspace(0, dimensions[0] - 1, dimensions[0])
    y = np.linspace(0, dimensions[1] - 1, dimensions[1])
    z = np.linspace(0, dimensions[2] - 1, dimensions[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    label_meta = batch_orig["label"].meta

    def make_coord_dict(coord_array):
        return {
            "image": batch_orig["image"].type(torch.float32),
            "label": MetaTensor(torch.tensor(coord_array).unsqueeze(0).type(torch.float32), meta=label_meta),
        }

    X_trans_b = transforms(make_coord_dict(X))
    Y_trans_b = transforms(make_coord_dict(Y))
    Z_trans_b = transforms(make_coord_dict(Z))

    X_trans = X_trans_b["label"].flatten()
    Y_trans = Y_trans_b["label"].flatten()
    Z_trans = Z_trans_b["label"].flatten()

    SEG_orig  = batch_orig["image"].flatten()
    SEG_trans = X_trans_b["image"].flatten()

    X = torch.tensor(X).float().flatten()
    Y = torch.tensor(Y).float().flatten()
    Z = torch.tensor(Z).float().flatten()

    if use_GT:
        X       = X[SEG_orig > 0].unsqueeze(1)
        Y       = Y[SEG_orig > 0].unsqueeze(1)
        Z       = Z[SEG_orig > 0].unsqueeze(1)
        X_trans = X_trans[SEG_trans > 0].unsqueeze(1)
        Y_trans = Y_trans[SEG_trans > 0].unsqueeze(1)
        Z_trans = Z_trans[SEG_trans > 0].unsqueeze(1)
    else:
        X, Y, Z             = X.unsqueeze(1), Y.unsqueeze(1), Z.unsqueeze(1)
        X_trans, Y_trans, Z_trans = X_trans.unsqueeze(1), Y_trans.unsqueeze(1), Z_trans.unsqueeze(1)

    original_coords    = torch.cat((X, Y, Z), dim=1).cpu().numpy()
    transformed_coords = torch.cat((X_trans, Y_trans, Z_trans), dim=1).cpu().numpy()

    rounded_transformed_coords = np.round(transformed_coords, decimals=0)
    original_coords             = np.round(original_coords,    decimals=0)

    closest_indices, _ = find_k_closest_points(original_coords, rounded_transformed_coords, k=1)

    num_channels = transform_label.shape[0]
    if tta:
        closest_labels = torch.zeros(
            [num_channels, int(torch.sum(SEG_orig).item())], dtype=torch.float32, device=device
        )
    else:
        closest_labels = torch.zeros(
            [num_channels, *batch_orig["label"].shape[1:]], dtype=torch.float32, device=device
        )

    transform_label = transform_label.to(device)

    for channel in range(num_channels):
        if use_GT:
            label_data = transform_label[channel, ...].flatten()[SEG_trans.flatten() > 0]
        else:
            label_data = transform_label[channel, ...].flatten()

        closest_label    = label_data[closest_indices[:, 0]]
        closest_labels_aux = closest_labels[channel, ...].flatten()

        if use_GT:
            if tta:
                closest_labels_aux = closest_label
            else:
                closest_labels_aux[SEG_orig.flatten() > 0] = closest_label
        else:
            closest_labels_aux = closest_label

        if tta:
            closest_labels[channel, ...] = closest_labels_aux
        else:
            closest_labels[channel, ...] = closest_labels_aux.reshape(batch_orig["label"].shape[1:])

    return closest_labels


# ---------------------------------------------------------------------------
# Test-time augmentation
# ---------------------------------------------------------------------------

def custom_tta(config, file_dict, model, post_pred, num_examples=5,
               device="cpu", old_tta=False, rand_params=None, ensemble=False):
    """
    Apply test-time augmentation by running the model on *num_examples* randomly
    transformed versions of the input and aggregating predictions via mode voting.

    Args:
        config       (dict):  Model / transform configuration.
        file_dict    (dict):  {'image': path, 'label': path}.
        model:                Trained model (or list of models for ensemble).
        post_pred:            Post-processing transform for predictions.
        num_examples (int):   Number of TTA samples.
        device:               Torch device.
        old_tta      (bool):  Use inverse-transform aggregation instead of coordinate mapping.
        rand_params  (list):  Pre-generated augmentation parameter dicts (optional).
        ensemble     (bool):  Average predictions from a list of models.

    Returns:
        (MODE, MEAN, STD, VVC) if old_tta else (MODE, MEAN, STD, VVC, params_list)
    """
    outs        = []
    params_list = []

    for i in range(num_examples):
        if rand_params is not None:
            params = rand_params[i]
        else:
            params = {
                "rotate_params":    (np.random.uniform(-np.pi / 10, np.pi / 10),) * 3,
                "translate_params": (int(np.random.randint(-3, 3)),) * 3,
                "scale_params":     (1.0, 1.0),
                "shear_params":     (0, 0),
            }
            params_list.append(params)

        _, val_transforms, minimal_transforms, coor_transforms = transformations(config, params=params)
        batch_data = val_transforms(file_dict)
        batch_orig = minimal_transforms(file_dict)

        target_spacing = (
            batch_data["label"].meta.get("affine")[0, 0],
            batch_data["label"].meta.get("affine")[1, 1],
            batch_data["label"].meta.get("affine")[2, 2],
        )
        _, val_transforms, minimal_transforms, coor_transforms = transformations(
            config, params=None, target_spacing=target_spacing
        )
        batch_data = val_transforms(file_dict)

        inputs = batch_data["image"].unsqueeze(0).to(device)

        if ensemble:
            logit_map = sum(m(inputs).clone() for m in model) / len(model)
        else:
            logit_map = model(inputs)

        if old_tta:
            from monai.transforms.utils import allow_missing_keys_mode
            logit_map_post = [post_pred(i) for i in decollate_batch(logit_map)]
            pred           = logit_map_post[0][1:, ...]
            pred.applied_operations = batch_data["label"].applied_operations
            pred.meta               = batch_data["label"].meta
            seg_dict = {"label": pred}
            with allow_missing_keys_mode(val_transforms):
                closest_labels = val_transforms.inverse(seg_dict)
            closest_labels = closest_labels["label"]
        else:
            labels_pred    = [post_pred(i) for i in decollate_batch(logit_map)]
            labels_pred    = labels_pred[0][1:, ...]
            closest_labels = compute_final_labels(batch_orig, labels_pred, coor_transforms, device, tta=True)

        outs.append(closest_labels)

    output = torch.stack(outs, 0)
    MODE   = torch.mode(output, dim=0)[0]
    MEAN   = output.mean(0)
    STD    = output.std(0)
    VVC    = (output.std() / output.mean()).item()

    if old_tta:
        return MODE, MEAN, STD, VVC

    # Map compact [C, N_fg] tensors back to full volume
    num_channels = 10
    label_shape  = batch_orig["label"].shape[1:]
    MODE_vol = torch.zeros([num_channels, *label_shape], dtype=torch.float32, device=device)
    MEAN_vol = torch.zeros([num_channels, *label_shape], dtype=torch.float32, device=device)
    STD_vol  = torch.zeros([num_channels, *label_shape], dtype=torch.float32, device=device)

    fg_mask = batch_orig["image"].flatten() > 0

    for ch in range(num_channels):
        MODE_vol[ch, ...].flatten()[fg_mask] = MODE[ch, :].clone()
        MEAN_vol[ch, ...].flatten()[fg_mask] = MEAN[ch, :].clone()
        STD_vol[ch,  ...].flatten()[fg_mask] = STD[ch,  :].clone()

    weights = torch.arange(1, num_channels + 1, dtype=torch.float32, device=device).view(-1, 1, 1, 1)
    MODE_scalar = (MODE_vol * weights).sum(0)
    MEAN_scalar = (MEAN_vol * weights).sum(0)
    STD_scalar  = STD_vol.sum(0)

    return MODE_scalar, MEAN_scalar, STD_scalar, VVC, params_list


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def validation_IC(directory, model, dataloader, loss_function, evaluation_metric,
                  epoch, log=False, device="cuda", fold=0, valid_files=None,
                  val_transforms=None, dataloader_min=None, minimal_transforms=None,
                  coor_transforms=None, config=None, ensemble=False):
    """
    Evaluate the model on the test split, optionally applying test-time augmentation.
    Saves per-sample prediction, ground truth, error, and uncertainty nrrd files and
    a CSV with per-class Dice scores.

    Args:
        directory           (str):  Output directory for saved nrrd files and CSV.
        model:                      Trained model or list of models (ensemble).
        dataloader:                 Validation dataloader (used for TTA-free forward pass).
        loss_function:              Loss function (unused in evaluation; kept for API consistency).
        evaluation_metric:          Metric object (unused directly).
        epoch               (int):  Current epoch (used for wandb logging step).
        log                 (bool): If True, log metrics to wandb.
        device              (str):  Torch device.
        fold                (int):  Current fold index.
        valid_files         (list): List of {'image': path, 'label': path} dicts.
        val_transforms:             Validation transform pipeline.
        dataloader_min:             Dataloader with minimal transforms (no spatial augmentation).
        minimal_transforms:         Minimal transform pipeline.
        coor_transforms:            Coordinate-only transform pipeline (for label inversion).
        config              (dict): Model configuration dict.
        ensemble            (bool): If True, *model* is treated as a list.

    Returns:
        (predictions, true_labels, inputs) lists collected across the test set.
    """
    os.makedirs(directory, exist_ok=True)

    post_pred  = Compose([AsDiscrete(argmax=True, to_onehot=11, threshold=0.5)])
    post_label = Compose([AsDiscrete(to_onehot=11)])

    predictions = []
    true_labels = []
    inputs_list = []

    if ensemble:
        for m in model:
            m.eval()
    else:
        model.eval()

    vessel_names = ["BA", "LACA", "LICA", "LMCA", "RACA", "RICA", "RMCA", "RPCA", "LPCA", "NA"]
    save_dict = {name: [] for name in vessel_names}
    save_dict["ID"]   = []
    save_dict["fold"] = []

    dice_metric_batch     = DiceMetric(include_background=True, reduction="mean_batch")
    dice_metric_batch_tr  = DiceMetric(include_background=True, reduction="mean_batch")
    dice_metric_batch_tta = DiceMetric(include_background=True, reduction="mean_batch")
    dice_metric_batch_old_tta = DiceMetric(include_background=True, reduction="mean_batch")

    do_tta = True

    with torch.no_grad():
        for step, file_dict in enumerate(valid_files):
            batch_orig = minimal_transforms(file_dict)

            _, val_transforms, minimal_transforms, coor_transforms = transformations(
                config, params=None, target_spacing=None
            )
            batch_data = val_transforms(file_dict)

            inputs, labels = (
                batch_data["image"].unsqueeze(0).to(device),
                batch_data["label"].unsqueeze(0).to(device),
            )

            labels_combined = sum(labels[:, ch, :, :, :] * (ch + 1) for ch in range(10))
            labels_combined = labels_combined.unsqueeze(1)

            inputs_orig, labels_orig = (
                batch_orig["image"].unsqueeze(0).to(device),
                batch_orig["label"].unsqueeze(0).to(device),
            )
            labels_orig_combined = sum(labels_orig[:, ch, :, :, :] * (ch + 1) for ch in range(10))
            labels_orig_combined = labels_orig_combined.unsqueeze(1)

            # Forward pass
            if ensemble:
                logit_map = sum(m(inputs).clone() for m in model) / len(model)
            else:
                logit_map = model(inputs)

            labels_pred = [post_pred(i) for i in decollate_batch(logit_map)]
            labels_pred = labels_pred[0][1:, ...]

            # Invert spatial transform to map predictions back to original space
            closest_labels = compute_final_labels(batch_orig, labels_pred, coor_transforms, device)
            weights = torch.arange(1, 11, dtype=torch.float32, device=device).view(-1, 1, 1, 1)
            labels_pred_combined = (closest_labels * weights).sum(0).unsqueeze(0).unsqueeze(0)

            # TTA
            MODE_tta, MEAN_tta, STD_tta, VVC_tta, rand_params = custom_tta(
                config, file_dict, model, post_pred, num_examples=5, device=device, ensemble=ensemble
            )
            Old_mode_tta, _, Old_std_tta, _ = custom_tta(
                config, file_dict, model, post_pred, num_examples=5,
                device=device, old_tta=True, rand_params=rand_params, ensemble=ensemble
            )
            Old_std_tta  = Old_std_tta.squeeze()
            MODE_tta     = MODE_tta.unsqueeze(0).unsqueeze(0)
            Old_mode_tta = Old_mode_tta.unsqueeze(0).unsqueeze(0)

            val_outputs          = [post_pred(i)  for i in decollate_batch(logit_map)]
            val_outputs_inverted = [post_label(i) for i in decollate_batch(labels_pred_combined)]
            val_outputs_tta      = [post_label(i) for i in decollate_batch(MODE_tta)]
            val_outputs_old_tta  = [post_label(i) for i in decollate_batch(Old_mode_tta)]
            labels_orig_list     = [post_label(i) for i in decollate_batch(labels_orig_combined)]

            dice_metric_batch_tr(y_pred=val_outputs,          y=labels_combined)
            dice_metric_batch(y_pred=val_outputs_inverted,    y=labels_orig_list)
            dice_metric_batch_tta(y_pred=val_outputs_tta,     y=labels_orig_list)
            dice_metric_batch_old_tta(y_pred=val_outputs_old_tta, y=labels_orig_list)

            # Parse patient ID from file path
            ID = int(file_dict["image"].split("/")[-2].split("_")[1])

            # Save nrrd outputs for this sample
            true_data  = np.squeeze(labels_orig_list[0].detach().cpu().numpy())
            pred_data  = np.squeeze(val_outputs_inverted[0].detach().cpu().numpy())
            pred_mode  = np.squeeze(val_outputs_tta[0].detach().cpu().numpy())
            pred_old   = np.squeeze(val_outputs_old_tta[0].detach().cpu().numpy())

            def label_to_scalar(arr):
                return sum(arr[ch, ...] * (ch + 1) for ch in range(1, arr.shape[0]))

            true_scalar  = label_to_scalar(true_data)
            pred_scalar  = label_to_scalar(pred_data)
            mode_scalar  = label_to_scalar(pred_mode)
            old_scalar   = label_to_scalar(pred_old)
            error_scalar = ((np.abs(pred_scalar - true_scalar)) > 0).astype(np.int32)
            error_scalar += np.squeeze(inputs_orig.detach().cpu().numpy().astype(np.int32))

            nrrd.write(f"{directory}/{ID}-input.nrrd",    np.squeeze(inputs.detach().cpu().numpy()).astype(np.uint8))
            nrrd.write(f"{directory}/{ID}-true.nrrd",     true_scalar.astype(np.uint8))
            nrrd.write(f"{directory}/{ID}-pred.nrrd",     pred_scalar.astype(np.uint8))
            nrrd.write(f"{directory}/{ID}-mode.nrrd",     mode_scalar.astype(np.uint8))
            nrrd.write(f"{directory}/{ID}-old_mode.nrrd", old_scalar.astype(np.uint8))
            nrrd.write(f"{directory}/{ID}-error.nrrd",    error_scalar.astype(np.uint8))
            nrrd.write(f"{directory}/{ID}-std.nrrd",      np.squeeze(STD_tta.detach().cpu().numpy()))
            nrrd.write(f"{directory}/{ID}-old_std.nrrd",  np.squeeze(Old_std_tta.detach().cpu().numpy()))

            # Per-sample Dice (using tr metric as representative)
            metric_batch = dice_metric_batch_tr.aggregate()
            per_class    = [metric_batch[i + 1].item() for i in range(10)]
            mean_dice    = float(np.nanmean(per_class))

            print(
                "  ".join(f"{n}: {v:.3f}" for n, v in zip(vessel_names, per_class))
                + f"  mean: {mean_dice:.3f}"
            )

            for name, value in zip(vessel_names, per_class):
                save_dict[name].append(value)
            save_dict["ID"].append(ID)
            save_dict["fold"].append(fold)

            dice_metric_batch_tr.reset()
            dice_metric_batch.reset()
            dice_metric_batch_tta.reset()
            dice_metric_batch_old_tta.reset()

    pd.DataFrame(save_dict).to_csv(f"{directory}/fold_{fold}_metrics.csv", index=False)

    # Log final aggregated metrics to wandb
    final_metrics = {name: float(np.nanmean(save_dict[name])) for name in vessel_names}
    mean_dice     = float(np.nanmean(list(final_metrics.values())))

    if log:
        wandb.log({"test_dice_mean": mean_dice, "epoch": epoch})
        for name, value in final_metrics.items():
            wandb.log({f"test_dice_{name.lower()}": value, "epoch": epoch})
    else:
        print(f"Mean Dice: {mean_dice:.4f}")
        for name, value in final_metrics.items():
            print(f"  {name}: {value:.4f}")

    return predictions, true_labels, inputs_list
