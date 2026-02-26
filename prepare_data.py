"""
prepare_data.py
===============
Single-script data preparation pipeline.  Given a dataset of nrrd files,
this script performs **all** of the following in one run:

  1. 5-fold cross-validation split (same seed as nnUNet).
  2. Copy nrrd files into per-fold train/test directory trees.
  3. Save JSON route files (ids_data, image_paths, labels) for UNet / CSNet.
  4. Copy ALL data into nnUNet imagesTr / labelsTr (nnUNet manages its own CV).
  5. Generate TTA test sets for each fold (random affine × NUM_TTAS) and save
     them into separate nnUNet imagesTs / labelsTs folders together with the
     pickled random parameters used for each transformation.

Input layout
------------
    datasets/data/masks/{patient}/{file}.nrrd
    datasets/data/labels/{patient}/{file}.nrrd

Output layout (examples for fold 0, 2 TTA versions)
----------------------------------------------------
    # Per-fold nrrd trees (used by UNet / CSNet dataloader)
    datasets/data/fold_0/train_data/masks/{patient}/{file}.nrrd
    datasets/data/fold_0/train_data/labels/{patient}/{file}.nrrd
    datasets/data/fold_0/test_data/masks/{patient}/{file}.nrrd
    datasets/data/fold_0/test_data/labels/{patient}/{file}.nrrd

    # JSON route dictionaries (one set per fold)
    dataset_paths_fold_0/ids_data.json
    dataset_paths_fold_0/image_paths.json
    dataset_paths_fold_0/labels.json

    # nnUNet training data
    nnUNet_raw/Dataset100/imagesTr/IC_{ID:04d}_0000.nrrd
    nnUNet_raw/Dataset100/labelsTr/IC_{ID:04d}.nrrd

    # nnUNet TTA test data (per fold, per TTA index)
    nnUNet_raw/Dataset100/imagesTs_fold_0_test_tta0/IC_{ID:04d}_0000.nrrd
    nnUNet_raw/Dataset100/labelsTs_fold_0_test_tta0/IC_{ID:04d}.nrrd
    nnUNet_raw/Dataset100/imagesTs_fold_0_test_tta0/random_params/IC_{ID:04d}_random_params.pkl
"""

import json
import logging
import os
import pickle
import shutil

import nrrd
import numpy as np
from monai.data import MetaTensor
from monai.transforms import Affine, Compose, Pad
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Configuration – edit these paths / constants as needed
# ---------------------------------------------------------------------------

# Root of the flat input dataset
MASKS_DIR        = "datasets/data/masks"
LABELS_DIR       = "datasets/data/labels"

# Where the per-fold nrrd trees will be written
FOLD_DATA_ROOT   = "datasets/data"

# Where the JSON route files will be written (one sub-folder per fold)
ROUTES_ROOT      = "."   # dataset_paths_fold_X will be created here

# nnUNet output root
NNUNET_ROOT      = os.path.join("nnUNet_raw", "Dataset100")

# Cross-validation settings  (must match nnUNet's own split seed)
NUM_FOLDS        = 5
CV_RANDOM_STATE  = 12345

# Number of TTA versions to generate per test sample
NUM_TTAS         = 7

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ===========================================================================
# Helper utilities
# ===========================================================================

def save_json(obj, filename, folder):
    """Serialise *obj* to JSON at *folder*/*filename*.json."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{filename}.json")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    log.info(f"Saved {path}")


def collect_samples(masks_dir, labels_dir):
    """
    Walk *masks_dir* (layout: {patient}/{file}.nrrd) and pair every mask file
    with its corresponding label file.

    Each case is assigned a sequential integer ID (0, 1, 2, …) based on the
    sorted order of patient folders.  No sub-type grouping is assumed.

    Returns
    -------
    samples : list of dicts with keys
        - "mask_path"  : absolute path to the mask nrrd
        - "label_path" : absolute path to the label nrrd
        - "patient"    : patient folder name
        - "filename"   : bare filename (e.g. "data.nrrd")
        - "patient_id" : sequential int ID
    """
    samples = []
    idx = 0
    for patient in sorted(os.listdir(masks_dir)):
        patient_mask_dir  = os.path.join(masks_dir, patient)
        patient_label_dir = os.path.join(labels_dir, patient)
        if not os.path.isdir(patient_mask_dir):
            continue
        for filename in sorted(os.listdir(patient_mask_dir)):
            if not filename.endswith(".nrrd"):
                continue
            mask_path  = os.path.join(patient_mask_dir, filename)
            label_path = os.path.join(patient_label_dir, filename)
            if not os.path.exists(label_path):
                log.warning(f"No matching label for {mask_path} – skipping.")
                continue
            samples.append({
                "mask_path":  mask_path,
                "label_path": label_path,
                "patient":    patient,
                "filename":   filename,
                "patient_id": idx,
            })
            idx += 1
    log.info(f"Collected {len(samples)} sample(s) from {masks_dir}")
    return samples


def generate_random_affine_params():
    """Return a fresh dict of random affine parameters."""
    return {
        "rotate_params": (
            float(np.random.uniform(-np.pi / 10, np.pi / 10)),
            float(np.random.uniform(-np.pi / 10, np.pi / 10)),
            float(np.random.uniform(-np.pi / 10, np.pi / 10)),
        ),
        "translate_params": (
            int(np.random.randint(-3, 3)),
            int(np.random.randint(-3, 3)),
            int(np.random.randint(-3, 3)),
        ),
        "scale_params": (1.0, 1.0),
        "shear_params": (0, 0),
    }


def apply_random_affine(data, params):
    """
    Apply a random affine transformation to a 3-D numpy array.

    The same *params* dict should be used for both the image and its label so
    that spatial alignment is preserved.  Nearest-neighbour interpolation is
    used to keep label values integer.

    Returns the transformed array with the same dtype as *data*.
    """
    original_dtype = data.dtype
    padding = [(0, 0), (10, 10), (10, 10), (30, 30)]   # channel, x, y, z

    transform = Compose([
        Pad(to_pad=padding, mode="constant", constant_values=0),
        Affine(
            translate_params=params["translate_params"],
            rotate_params=params["rotate_params"],
            scale_params=params["scale_params"],
            padding_mode="zeros",
            mode="nearest",
        ),
    ])

    tensor = MetaTensor(np.expand_dims(data, axis=0).astype(np.float32))
    transformed, _ = transform(tensor)
    return np.squeeze(transformed.numpy()).astype(original_dtype)


# ===========================================================================
# Step 1 – Build the 5-fold CV split
# ===========================================================================

def build_cv_splits(samples):
    """
    Perform plain 5-fold CV on *samples*.

    Returns
    -------
    folds : list of NUM_FOLDS dicts, each with keys
        "train" : list of sample dicts for this fold's training set
        "test"  : list of sample dicts for this fold's test set
    """
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    folds = []
    for train_idx, test_idx in kf.split(samples):
        folds.append({
            "train": [samples[i] for i in train_idx],
            "test":  [samples[i] for i in test_idx],
        })
    for f, fold in enumerate(folds):
        log.info(
            f"Fold {f}: {len(fold['train'])} train / {len(fold['test'])} test"
        )
    return folds


# ===========================================================================
# Step 2 – Copy nrrd files into per-fold directory trees
# ===========================================================================

def copy_fold_trees(folds):
    """
    For each fold, copy mask and label nrrd files into:
        FOLD_DATA_ROOT/fold_{f}/train_data/masks/{patient}/{file}
        FOLD_DATA_ROOT/fold_{f}/train_data/labels/{patient}/{file}
        FOLD_DATA_ROOT/fold_{f}/test_data/masks/{patient}/{file}
        FOLD_DATA_ROOT/fold_{f}/test_data/labels/{patient}/{file}
    """
    for f, fold in enumerate(folds):
        for split_name, split_samples in [("train_data", fold["train"]),
                                           ("test_data",  fold["test"])]:
            for s in split_samples:
                for kind, src in [("masks", s["mask_path"]),
                                   ("labels", s["label_path"])]:
                    dst_dir = os.path.join(
                        FOLD_DATA_ROOT,
                        f"fold_{f}",
                        split_name,
                        kind,
                        s["patient"],
                    )
                    os.makedirs(dst_dir, exist_ok=True)
                    dst = os.path.join(dst_dir, s["filename"])
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        log.info(f"Copied {src} → {dst}")
                    else:
                        log.debug(f"Already exists, skipping: {dst}")

    log.info("Per-fold directory trees written.")


# ===========================================================================
# Step 3 – Save JSON route files for UNet / CSNet
# ===========================================================================

def save_route_jsons(folds):
    """
    For each fold write three JSON files to ROUTES_ROOT/dataset_paths_fold_{f}/:
        ids_data.json    – {"train": [...], "test": [...]}
        image_paths.json – {"id-N": "/abs/path/to/mask.nrrd", ...}
        labels.json      – {"id-N": "/abs/path/to/label.nrrd", ...}

    IDs are global (unique across folds) so that the same sample always has
    the same id string regardless of which fold is being used.
    """
    for f, fold in enumerate(folds):
        ids_data    = {"train": [], "test": []}
        image_paths = {}
        labels_map  = {}

        for split_name, split_samples in [("train", fold["train"]),
                                           ("test",  fold["test"])]:
            for s in split_samples:
                id_str = f"id-{s['patient_id']}"
                ids_data[split_name].append(id_str)
                image_paths[id_str] = os.path.abspath(s["mask_path"])
                labels_map[id_str]  = os.path.abspath(s["label_path"])

        out_dir = os.path.join(ROUTES_ROOT, f"dataset_paths_fold_{f}")
        save_json(ids_data,    "ids_data",    out_dir)
        save_json(image_paths, "image_paths", out_dir)
        save_json(labels_map,  "labels",      out_dir)

    log.info("JSON route files written.")


# ===========================================================================
# Step 4 – Copy ALL data to nnUNet imagesTr / labelsTr
# ===========================================================================

def copy_nnunet_train(samples):
    """
    Flat-copy every sample to nnUNet's imagesTr / labelsTr directories.
    nnUNet manages its own cross-validation internally; we just need all data
    present in these two folders.

    Output filenames follow nnUNet convention:
        imagesTr/IC_{ID:04d}_0000.nrrd
        labelsTr/IC_{ID:04d}.nrrd
    """
    images_tr = os.path.join(NNUNET_ROOT, "imagesTr")
    labels_tr  = os.path.join(NNUNET_ROOT, "labelsTr")
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr,  exist_ok=True)

    for s in samples:
        pid = s["patient_id"]
        mask_dst  = os.path.join(images_tr, f"IC_{pid:04d}_0000.nrrd")
        label_dst = os.path.join(labels_tr,  f"IC_{pid:04d}.nrrd")

        if not os.path.exists(mask_dst):
            shutil.copy2(s["mask_path"], mask_dst)
            log.info(f"nnUNet train | {s['mask_path']} → {mask_dst}")
        if not os.path.exists(label_dst):
            shutil.copy2(s["label_path"], label_dst)
            log.info(f"nnUNet train | {s['label_path']} → {label_dst}")

    log.info("nnUNet imagesTr / labelsTr populated.")


# ===========================================================================
# Step 5 – Generate nnUNet TTA test sets (per fold × per TTA index)
# ===========================================================================

def generate_nnunet_tta_test(folds, num_ttas=NUM_TTAS):
    """
    For each fold and each TTA index, apply a fresh random affine
    transformation to every test sample and write the result to:

        nnUNet_raw/Dataset100/imagesTs_fold_{f}_test_tta{n}/IC_{ID:04d}_0000.nrrd
        nnUNet_raw/Dataset100/labelsTs_fold_{f}_test_tta{n}/IC_{ID:04d}.nrrd
        nnUNet_raw/Dataset100/imagesTs_fold_{f}_test_tta{n}/random_params/
            IC_{ID:04d}_random_params.pkl

    The *same* random_params dict is applied to both image and label within a
    single (sample, tta_index) pair so spatial correspondence is maintained.
    """
    for f, fold in enumerate(folds):
        test_samples = fold["test"]
        for tta_idx in range(num_ttas):
            tag         = f"fold_{f}_test_tta{tta_idx}"
            images_ts   = os.path.join(NNUNET_ROOT, f"imagesTs_{tag}")
            labels_ts   = os.path.join(NNUNET_ROOT, f"labelsTs_{tag}")
            params_dir  = os.path.join(images_ts, "random_params")
            os.makedirs(images_ts,  exist_ok=True)
            os.makedirs(labels_ts,   exist_ok=True)
            os.makedirs(params_dir,  exist_ok=True)

            for s in test_samples:
                pid = s["patient_id"]
                mask_dst   = os.path.join(images_ts,  f"IC_{pid:04d}_0000.nrrd")
                label_dst  = os.path.join(labels_ts,   f"IC_{pid:04d}.nrrd")
                params_dst = os.path.join(params_dir,  f"IC_{pid:04d}_random_params.pkl")

                log.info(f"TTA | fold={f} tta={tta_idx} patient={pid:04d}")

                mask_data,  mask_header  = nrrd.read(s["mask_path"])
                label_data, label_header = nrrd.read(s["label_path"])

                # Single set of params shared by image + label
                params = generate_random_affine_params()

                mask_aug  = apply_random_affine(mask_data,  params)
                label_aug = apply_random_affine(label_data, params)

                # Update size metadata to reflect post-padding shape
                mask_header["sizes"]  = list(mask_aug.shape)
                label_header["sizes"] = list(label_aug.shape)

                nrrd.write(mask_dst,  mask_aug,  mask_header)
                nrrd.write(label_dst, label_aug, label_header)

                with open(params_dst, "wb") as fh:
                    pickle.dump(params, fh)

    log.info("nnUNet TTA test sets generated.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    # --- Validate input directories ----------------------------------------
    for d in (MASKS_DIR, LABELS_DIR):
        if not os.path.isdir(d):
            log.error(f"Input directory not found: {d}")
            return

    # --- Collect all samples ------------------------------------------------
    samples = collect_samples(MASKS_DIR, LABELS_DIR)
    if not samples:
        log.error("No valid samples found. Aborting.")
        return

    # --- Build CV splits (same seed as nnUNet) ------------------------------
    folds = build_cv_splits(samples)

    # --- Step 2: per-fold nrrd directory trees ------------------------------
    log.info("=== Step 2: Copying per-fold nrrd trees ===")
    copy_fold_trees(folds)

    # --- Step 3: JSON routes for UNet / CSNet -------------------------------
    log.info("=== Step 3: Saving JSON route files ===")
    save_route_jsons(folds)

    # --- Step 4: nnUNet imagesTr / labelsTr (all data, no split) ------------
    log.info("=== Step 4: Populating nnUNet imagesTr / labelsTr ===")
    copy_nnunet_train(samples)

    # --- Step 5: nnUNet TTA test sets ---------------------------------------
    log.info("=== Step 5: Generating nnUNet TTA test sets ===")
    generate_nnunet_tta_test(folds, num_ttas=NUM_TTAS)

    log.info("=== All done. ===")


if __name__ == "__main__":
    main()