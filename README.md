# Uncertainty-Aware Automated Labeling of Intracranial Arteries

This is the implementation of the paper [Uncertainty-Aware Automated Labeling of Intracranial Arteries](https://link.springer.com/article/10.1186/s12880-026-02276-5)

Follow steps for training and evaluation of UNet, CSNet3D, and nnUNet with test-time augmentation (TTA) and optional ensemble inference.
Cross-validation is performed using 5-fold CV.

---

![](imgs/Overview-updated4.jpg)

## Repository Structure

```
├── main.py                   # Training and evaluation entry point
├── train.py                  # Training and validation loop functions
├── test.py                   # Test evaluation with TTA and metric logging
├── prepare_data.py           # Dataset preparation: CV splits, route JSONs, nnUNet folders
├── custom_tta_nnunet.py      # TTA post-processing for nnUNet predictions
├── run_nnunet_eval.sh        # nnUNet inference + TTA post-processing script
│
├── src/
│   ├── dataloader.py         # MONAI transforms and label conversion
│   ├── losses.py             # Loss function definitions
│   ├── metrics.py            # Metric definitions
│   ├── models.py             # Model definitions (UNet, CSNet3D)
│   └── optimizers.py         # Optimizer definitions
│
└── config/
    ├── config_unet.yml       # Configuration for UNet
    └── config_csnet.yml      # Configuration for CSNet3D
```

---

## Dataset Structure

Place your data as follows before running `prepare_data.py`:

```
datasets/
└── data/
    ├── masks/
    │   └── {patient}/
    │       └── data.nrrd     # Binary vessel mask (input image)
    └── labels/
        └── {patient}/
            └── data.nrrd     # Integer label map (10 artery classes)
```

Each `{patient}` folder corresponds to one case. Labels use the following integer encoding:

| Value | Structure |
|-------|-----------|
| 1  | Basilar Artery (BA) |
| 2  | Left Anterior Cerebral Artery (LACA) |
| 3  | Left Internal Carotid Artery (LICA) |
| 4  | Left Middle Cerebral Artery (LMCA) |
| 5  | Right Anterior Cerebral Artery (RACA) |
| 6  | Right Internal Carotid Artery (RICA) |
| 7  | Right Middle Cerebral Artery (RMCA) |
| 8  | Right Posterior Cerebral Artery (RPCA) |
| 9  | Left Posterior Cerebral Artery (LPCA) |
| 10 | Non-annotated vessels (NA) |

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate ICUNET
```

---

## Pretrained Weights

Pretrained model weights for all five folds (UNet and CSNet) and all five folds (nnUNet) are available for download.

**Download:** [[HuggingFace]](https://huggingface.co/javierbz24/UA-UNet/tree/main)

### UNet / CSNet weights

After downloading, place the weight files in model-specific subdirectories under `wandb/`:

```
wandb/
├── UNet/
│   ├── fold_0_last_model.pt
│   ├── fold_1_last_model.pt
│   ├── fold_2_last_model.pt
│   ├── fold_3_last_model.pt
│   └── fold_4_last_model.pt
└── CSNet/
    ├── fold_0_last_model.pt
    ├── fold_1_last_model.pt
    ├── fold_2_last_model.pt
    ├── fold_3_last_model.pt
    └── fold_4_last_model.pt
```

Then set all `run_ids` to empty strings in your config file:

```yaml
wandb:
  run_ids: ["", "", "", "", ""]
```

The evaluation script will automatically detect this mode and select the correct subfolder based on the `model` field in your config (`wandb/UNet/` for `unet`; `wandb/CSNet/` for `CSNet`).

### nnUNet weights

Place the downloaded nnUNet checkpoints under `nnUNet_models/`, one subdirectory per fold:

```
nnUNet_models/
├── f0/
│   └── checkpoint_final.pth
├── f1/
│   └── checkpoint_final.pth
├── f2/
│   └── checkpoint_final.pth
├── f3/
│   └── checkpoint_final.pth
└── f4/
    └── checkpoint_final.pth
```

---

## Step 1 – Dataset Preparation

> **Skip this step** if you only want to run evaluation with pretrained weights and already have your test data in the expected layout.

Run `prepare_data.py` once before training. It performs the following steps in a single run:

1. **5-fold CV split** using `KFold(shuffle=True, random_state=12345)`.
2. **Copies nrrd files** into per-fold train/test directory trees under `datasets/data/fold_{f}/`.
3. **Saves JSON route files** (`ids_data.json`, `image_paths.json`, `labels.json`) for each fold under `dataset_paths_fold_{f}/`. These are consumed by the UNet/CSNet training pipeline.
4. **Populates `nnUNet_raw/Dataset100/imagesTr` and `labelsTr`** with all data (flat copy, no fold split — nnUNet manages its own CV internally using the same seed).
5. **Generates TTA test sets** for nnUNet: for each fold and each of the 7 TTA versions, applies a random affine transformation to the test images and saves them to `nnUNet_raw/Dataset100/imagesTs_fold_{f}_test_tta{n}/` and `labelsTs_fold_{f}_test_tta{n}/`. The random parameters for each sample are pickled alongside for reproducibility.

```bash
python prepare_data.py
```

Edit the constants at the top of `prepare_data.py` to point to your data directories if needed.

After this step, the output directory structure will look like:

```
datasets/data/
├── fold_0/
│   ├── train_data/masks/{patient}/data.nrrd
│   ├── train_data/labels/{patient}/data.nrrd
│   ├── test_data/masks/{patient}/data.nrrd
│   └── test_data/labels/{patient}/data.nrrd
├── fold_1/ ...
...

dataset_paths_fold_0/
├── ids_data.json
├── image_paths.json
└── labels.json

nnUNet_raw/Dataset100/
├── imagesTr/ICAD_{ID:04d}_0000.nrrd
├── labelsTr/ICAD_{ID:04d}.nrrd
├── imagesTs_fold_0_test_tta0/
├── labelsTs_fold_0_test_tta0/
...
```

---

## Step 2 – Configuration

Edit `config/config_unet.yml` or `config/config_csnet.yml` to set your training parameters.

Key fields under `base`:

| Field | Description |
|-------|-------------|
| `model` | One of `unet`, `CSNet` |
| `classes` | Number of output classes (11, including background) |
| `epochs` | Total training epochs |
| `learning_rate` | Initial learning rate |
| `batch_size` | Training batch size |
| `loss` | `Dice` or `DiceCE` |
| `model_evaluation_output` | Directory where prediction nrrd files are saved during evaluation |

The `dataset.route_paths` field must match the prefix of your JSON route folders. With the default value `dataset_paths`, routes are read from `dataset_paths_fold_{fold}/`.

---

## Step 3 – Weights & Biases

> **Skip this step** if you are only running evaluation with pretrained weights.

Training uses [Weights & Biases](https://wandb.ai/) for experiment tracking.

```bash
wandb login
```

Set `wandb.mode: offline` in the config file if you do not want to sync runs.

---

## Step 4 – Training

> **Skip this step** if you are only running evaluation with pretrained weights.

Train a single fold:

```bash
python main.py 0 --config-route config/config_unet.yml   # UNet, fold 0
python main.py 0 --config-route config/config_csnet.yml  # CSNet, fold 0
```

Repeat for folds 1–4. Each run saves the final model weights inside the corresponding `wandb/` run directory as `fold_{f}_last_model.pt`.

---

## Step 5 – Evaluation (UNet / CSNet)

### Using pretrained weights

Set `run_ids` to empty strings in the config (see [Pretrained Weights](#pretrained-weights)), then run:

```bash
# Single fold
python main.py 0 --config-route config/config_unet.yml --eval
python main.py 0 --config-route config/config_csnet.yml --eval

# Ensemble across all five folds
python main.py 0 --config-route config/config_unet.yml --eval --ensemble
python main.py 0 --config-route config/config_csnet.yml --eval --ensemble
```

### Using weights from your own training runs

Set `wandb.run_ids` in the config to the run IDs produced during training, then run the same commands above. The script will locate model files under `wandb/*/files/fold_{f}_last_model.pt`.

Outputs saved to the path specified by `model_evaluation_output` in the config:

| File | Description |
|------|-------------|
| `{ID}-input.nrrd` | Input image |
| `{ID}-true.nrrd` | Ground truth label |
| `{ID}-pred.nrrd` | Model prediction (no TTA) |
| `{ID}-mode.nrrd` | TTA prediction (coordinate-mapping aggregation) |
| `{ID}-old_mode.nrrd` | TTA prediction (inverse-transform aggregation) |
| `{ID}-error.nrrd` | Error map |
| `{ID}-std.nrrd`, `{ID}-old_std.nrrd` | Uncertainty maps |
| `fold_{f}_metrics.csv` | Per-sample per-class Dice scores |

---

## nnUNet

This repository generates data and TTA test sets compatible with nnUNet.
We recommend using the following fork for training:

> **[https://github.com/JavierBZ/nnUNet](https://github.com/JavierBZ/nnUNet)**

This fork comments out the intensity-based data augmentations that nnUNet applies by default.
Those augmentations are appropriate for grayscale images but distort the data in this setting, where **the inputs are binary vessel masks** (not raw MRA intensities). Disabling them yields more stable training.

Follow the nnUNet documentation for dataset fingerprinting, planning, and training. Use `Dataset100` as the dataset identifier, which matches the folder names generated by `prepare_data.py`.

### Evaluation (nnUNet)

A convenience script `run_nnunet_eval.sh` is provided. It runs nnUNet inference across all configured folds and TTA passes, then calls `custom_tta_nnunet.py` for aggregation.

**1. Configure the script**

Open `run_nnunet_eval.sh` and edit the variables at the top:

```bash
FOLDS="0 1 2 3 4"       # folds to evaluate (space-separated)
N_TTA=7                  # number of TTA versions (tta0 .. tta6)
CKPT_DIR="nnUNet_models" # directory containing f{fold}/checkpoint_final.pth
```

**2. Set nnUNet environment variables**

```bash
export nnUNet_raw="$(pwd)/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnUNet_results"
```

**3. Run**

```bash
bash run_nnunet_eval.sh
```

The script will:
- Run `nnUNetv2_predict` for each fold × TTA combination, writing predictions to `nnUNet_results/Dataset100/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/tta{n}_f{fold}/`
- Call `custom_tta_nnunet.py` to aggregate TTA predictions across passes

> **Note:** To evaluate a single fold only, set `FOLDS="0"` in the script.

### TTA Post-processing for nnUNet

`custom_tta_nnunet.py` aggregates per-fold predictions across all TTA passes using the same strategy (mode voting with coordinate-mapping or inverse-transform) used for the UNet/CSNet pipeline. Configure the path constants at the top of its `__main__` block before running independently:

```bash
python custom_tta_nnunet.py
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bisbal2026uncertainty,
  title={Uncertainty-aware automated labeling of intracranial arteries using deep learning},
  author={Bisbal, Javier and Winter, Patrick and Jofre, Sebastian and Ponce, Aaron and Ansari, Sameer A and Abdalla, Ramez and Markl, Michael and Odeback, Oliver Welin and Tejos, Cristian and Uribe, Sergio and Schnell, Susanne and Marlevi, David},
  journal={BMC Medical Imaging},
  year={2026},
  publisher={Springer}
}
```