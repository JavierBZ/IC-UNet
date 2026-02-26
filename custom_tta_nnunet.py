"""
custom_tta_nnunet.py
====================
Post-processing script for nnUNet predictions with test-time augmentation (TTA).

For each fold and TTA version, this script loads nnUNet predictions together
with the random affine parameters used during data preparation, inverts the
spatial transformation to map predictions back to the original image space,
and aggregates them via mode voting.

Two aggregation modes are supported:
  - "new": coordinate-mapping via KDTree nearest-neighbour lookup.
  - "old": MONAI inverse-transform.

Usage
-----
Configure the paths at the top of `__main__` and run:

    python custom_tta_nnunet.py
"""

import os
import pickle
import time

import numpy as np
import nrrd
import torch
from tqdm import tqdm
from scipy.spatial import cKDTree

from monai.data import decollate_batch, MetaTensor
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Affined,
    SpatialPadd,
    AsDiscrete,
    MapTransform,
)
from monai.transforms.utils import allow_missing_keys_mode


# ---------------------------------------------------------------------------
# Label conversion
# ---------------------------------------------------------------------------

class ConvertToMultiChannelBasedOnICClassesd(MapTransform):
    """
    Convert an integer-label volume into a 10-channel one-hot tensor.

    Label mapping:
        1 → BA    2 → LACA  3 → LICA  4 → LMCA  5 → RACA
        6 → RICA  7 → RMCA  8 → RPCA  9 → LPCA  10 → NA
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = torch.stack([(d[key] == i).float() for i in range(1, 11)], dim=0)
        return d

    def inverse(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = torch.argmax(d[key], dim=0)
        return d


# ---------------------------------------------------------------------------
# Coordinate mapping utilities
# ---------------------------------------------------------------------------

def find_k_closest_points(original, transformed, k=1, chunk_size=10_000):
    """
    For each point in *original*, find the *k* nearest neighbours in *transformed*
    using a cKDTree, processing in chunks to limit memory usage.

    Args:
        original    (np.ndarray): Query points, shape [N, 3].
        transformed (np.ndarray): Reference points, shape [M, 3].
        k           (int):        Number of nearest neighbours.
        chunk_size  (int):        Query chunk size.

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
    Map predicted labels from the transformed space back to the original image
    space using nearest-neighbour coordinate lookup.

    Args:
        batch_orig      (dict):         Original sample dict with 'image' and 'label'.
        transform_label (torch.Tensor): Predicted multi-channel label [C, H, W, D].
        transforms      (Compose):      Coordinate transform pipeline.
        device:                         Torch device.
        use_GT          (bool):         Restrict mapping to foreground voxels.
        tta             (bool):         Return compact [C, N_fg] tensor if True.

    Returns:
        closest_labels (torch.Tensor)
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
            "label": MetaTensor(
                torch.tensor(coord_array).unsqueeze(0).type(torch.float32), meta=label_meta
            ),
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
        X, Y, Z             = (t[SEG_orig > 0].unsqueeze(1) for t in (X, Y, Z))
        X_trans, Y_trans, Z_trans = (
            t[SEG_trans > 0].unsqueeze(1) for t in (X_trans, Y_trans, Z_trans)
        )
    else:
        X, Y, Z             = (t.unsqueeze(1) for t in (X, Y, Z))
        X_trans, Y_trans, Z_trans = (t.unsqueeze(1) for t in (X_trans, Y_trans, Z_trans))

    original_coords    = torch.cat((X, Y, Z), dim=1).cpu().numpy()
    transformed_coords = torch.cat((X_trans, Y_trans, Z_trans), dim=1).cpu().numpy()

    original_coords    = np.round(original_coords,    decimals=0)
    transformed_coords = np.round(transformed_coords, decimals=0)

    closest_indices, _ = find_k_closest_points(original_coords, transformed_coords, k=1)

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

        if closest_indices.shape[1] > 1:
            closest_label = torch.mode(label_data[closest_indices], dim=1)[0]
        else:
            closest_label = label_data[closest_indices[:, 0]]

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Configure paths here
    # -----------------------------------------------------------------------
    SAVE_DIRECTORY = "Predictions/nnunet/"
    NNUNET_RAW     = "nnUNet_raw/Dataset100"
    NNUNET_RESULTS = "nnUNet_results/Dataset100/nnUNetResEnc"
    NUM_FOLDS      = 5
    NUM_TTA        = 7
    # -----------------------------------------------------------------------

    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    post_label = Compose([AsDiscrete(to_onehot=11)])
    np.random.seed(1234)

    for tta_mode in ["new", "old"]:
        for fold in range(NUM_FOLDS):
            crossval_folder = os.path.join(NNUNET_RAW, f"imagesTs_fold_{fold}_test_tta0")
            all_files = os.listdir(crossval_folder)
            files     = [f for f in all_files if f.endswith(".nrrd") and "ICAD" in f]
            IDs       = [int(f.split("_")[1]) for f in files]

            start_time = time.time()

            for ID in IDs:
                outs = []

                for tta_idx in range(NUM_TTA):
                    random_params_folder = os.path.join(NNUNET_RAW, f"imagesTs_fold_{fold}_test_tta{tta_idx}")
                    original_file        = os.path.join(NNUNET_RAW, "imagesTr", f"ICAD_{ID:04d}_0000.nrrd")
                    original_label       = os.path.join(NNUNET_RAW, "labelsTr", f"ICAD_{ID:04d}.nrrd")
                    tta_folder           = os.path.join(NNUNET_RESULTS, f"tta{tta_idx}_f{fold}")

                    tta_files = [f for f in os.listdir(tta_folder) if f"ICAD_{ID:04d}" in f and f.endswith(".nrrd")]
                    if not tta_files:
                        print(f"Warning: no prediction file found for ID {ID:04d} in {tta_folder}")
                        continue

                    orig_dic = {"image": original_file, "label": original_label}
                    transformed_dic = {
                        "image": os.path.join(random_params_folder, f"ICAD_{ID:04d}_0000.nrrd"),
                        "label": os.path.join(tta_folder, f"ICAD_{ID:04d}.nrrd"),
                    }

                    with open(os.path.join(random_params_folder, "random_params", f"ICAD_{ID:04d}_random_params.pkl"), "rb") as f:
                        random_params = pickle.load(f)

                    keys = ["image", "label"]
                    target_padding = tuple(
                        nrrd.read_header(transformed_dic["image"])["sizes"]
                    )

                    minimal_transforms = Compose([
                        LoadImaged(keys=keys),
                        EnsureChannelFirstd(keys="image"),
                        ConvertToMultiChannelBasedOnICClassesd(keys="label"),
                    ])

                    affine_transform = Affined(
                        keys=keys,
                        translate_params=random_params["translate_params"],
                        rotate_params=random_params["rotate_params"],
                        scale_params=random_params["scale_params"],
                        padding_mode="zeros",
                        mode=["nearest", "nearest"],
                    )

                    full_transform = Compose([
                        LoadImaged(keys=keys),
                        EnsureChannelFirstd(keys="image"),
                        ConvertToMultiChannelBasedOnICClassesd(keys="label"),
                        SpatialPadd(keys=keys, spatial_size=target_padding),
                        affine_transform,
                    ])

                    coor_transforms = Compose([
                        SpatialPadd(keys=keys, spatial_size=target_padding),
                        affine_transform,
                    ])

                    batch_orig     = minimal_transforms(orig_dic)
                    batch_predict  = minimal_transforms(transformed_dic)
                    batch_transform = full_transform(orig_dic)

                    predictions = batch_predict["label"].to(device)

                    if tta_mode == "old":
                        predictions.applied_operations = batch_transform["label"].applied_operations
                        predictions.meta               = batch_transform["label"].meta
                        seg_dict = {"label": batch_transform["label"]}
                        with allow_missing_keys_mode(full_transform):
                            closest_labels = full_transform.inverse(seg_dict)
                        closest_labels = closest_labels["label"]
                    else:
                        closest_labels = compute_final_labels(
                            batch_orig, predictions, coor_transforms, device, tta=True
                        )

                    outs.append(closest_labels)

                if not outs:
                    continue

                output = torch.stack(outs, 0)
                MODE   = torch.mode(output, dim=0)[0]
                MEAN   = output.mean(0)
                STD    = output.std(0)

                if tta_mode == "old":
                    weights = torch.arange(1, 11, dtype=torch.float32, device=device).view(-1, 1, 1, 1)
                    MODE = (MODE * weights).sum(0)
                    MEAN = (MEAN * weights).sum(0)
                    STD  = STD.squeeze()
                else:
                    num_channels = 10
                    label_shape  = batch_orig["label"].shape[1:]
                    fg_mask = batch_orig["image"].flatten() > 0

                    MODE_vol = torch.zeros([num_channels, *label_shape], dtype=torch.float32, device=device)
                    MEAN_vol = torch.zeros([num_channels, *label_shape], dtype=torch.float32, device=device)
                    STD_vol  = torch.zeros([num_channels, *label_shape], dtype=torch.float32, device=device)

                    for ch in range(num_channels):
                        MODE_vol[ch, ...].flatten()[fg_mask] = MODE[ch, :].clone()
                        MEAN_vol[ch, ...].flatten()[fg_mask] = MEAN[ch, :].clone()
                        STD_vol[ch,  ...].flatten()[fg_mask] = STD[ch,  :].clone()

                    weights = torch.arange(1, num_channels + 1, dtype=torch.float32, device=device).view(-1, 1, 1, 1)
                    MODE = (MODE_vol * weights).sum(0)
                    MEAN = (MEAN_vol * weights).sum(0)
                    STD  = STD_vol.sum(0)

                MODE = MODE.unsqueeze(0).unsqueeze(0)
                val_outputs_tta = [post_label(i) for i in decollate_batch(MODE)]

                pred_arr = np.squeeze(val_outputs_tta[0].detach().cpu().numpy())
                pred_scalar = sum(pred_arr[ch, ...] * (ch + 1) for ch in range(1, pred_arr.shape[0]))

                suffix = "old" if tta_mode == "old" else ""
                mode_key = f"{suffix}_mode" if suffix else "mode"
                std_key  = f"{suffix}_std"  if suffix else "std"

                nrrd.write(f"{SAVE_DIRECTORY}/{ID}-{mode_key}.nrrd", pred_scalar.astype(np.uint8))
                nrrd.write(f"{SAVE_DIRECTORY}/{ID}-{std_key}.nrrd",  np.squeeze(STD.detach().cpu().numpy()))

                elapsed = time.time() - start_time
                print(f"Saved ID {ID:04d}  mode={tta_mode}  fold={fold}  elapsed={elapsed:.1f}s")
