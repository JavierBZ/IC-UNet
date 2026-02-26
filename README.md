# Uncertainty-Aware Automated Labeling of Intracranial Arteries Using Deep Learning

3D segmentation and labeling of intracranial arteries from Time-of-Flight MRA images.
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
│
├── src/
│   ├── dataloader.py         # MONAI transforms and label conversion
│   ├── losses.py             # Loss function definitions
│   ├── metrics.py            # Metric definitions
│   ├── models.py             # Model definitions (UNet, UNETR, SwinUNETR, CSNet3D)
│   └── optimizers.py         # Optimizer definitions
│
└── config/
    ├── config_unet.yml       # Configuration for UNet / UNETR / SwinUNETR
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
| 1     | Basilar Artery (BA) |
| 2     | Left Anterior Cerebral Artery (LACA) |
| 3     | Left Internal Carotid Artery (LICA) |
| 4     | Left Middle Cerebral Artery (LMCA) |
| 5     | Right Anterior Cerebral Artery (RACA) |
| 6     | Right Internal Carotid Artery (RICA) |
| 7     | Right Middle Cerebral Artery (RMCA) |
| 8     | Right Posterior Cerebral Artery (RPCA) |
| 9     | Left Posterior Cerebral Artery (LPCA) |
| 10    | Non-annotated vessels (NA) |

---

## Environment Setup

```bash
conda env create -f env/environment.yml
conda activate unet
```

---

## Step 1 – Dataset Preparation

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
| `model` | One of `unet`, `unetr`, `SwinUNETR`, `CSNet` |
| `classes` | Number of output classes (11, including background) |
| `epochs` | Total training epochs |
| `learning_rate` | Initial learning rate |
| `batch_size` | Training batch size |
| `loss` | `Dice` or `DiceCE` |
| `model_evaluation_output` | Directory where prediction nrrd files are saved during evaluation |

The `dataset.route_paths` field must match the prefix of your JSON route folders. With the default value `dataset_paths`, routes are read from `dataset_paths_fold_{fold}/`.

---

## Step 3 – Weights & Biases

Training uses [Weights & Biases](https://wandb.ai/) for experiment tracking.

```bash
wandb login
```

Set `wandb.mode: offline` in the config file if you do not want to sync runs.

---

## Step 4 – Training

Train a single fold:

```bash
python main.py 0 --config-route config/config_unet.yml   # UNet, fold 0
python main.py 0 --config-route config/config_csnet.yml  # CSNet, fold 0
```

Repeat for folds 1–4. Each run saves the final model weights inside the corresponding `wandb/` run directory as `fold_{f}_last_model.pt`.

---

## Step 5 – Evaluation

Evaluate using the saved model for a single fold:

```bash
python main.py 0 --config-route config/config_unet.yml --eval
```

Run ensemble evaluation across all five fold models:

```bash
python main.py 0 --config-route config/config_unet.yml --eval --ensemble
```

The script auto-discovers model files under `wandb/*/files/fold_{f}_last_model.pt`.

Outputs saved to the path specified by `model_evaluation_output` in the config:
- `{ID}-input.nrrd` — input image
- `{ID}-true.nrrd` — ground truth label
- `{ID}-pred.nrrd` — model prediction (no TTA)
- `{ID}-mode.nrrd` — TTA prediction (coordinate-mapping aggregation)
- `{ID}-old_mode.nrrd` — TTA prediction (inverse-transform aggregation)
- `{ID}-error.nrrd` — error map
- `{ID}-std.nrrd`, `{ID}-old_std.nrrd` — uncertainty maps
- `fold_{f}_metrics.csv` — per-sample per-class Dice scores

---

## nnUNet

This repository generates data and TTA test sets compatible with nnUNet.
We recommend using the following fork for training:

> **[https://github.com/JavierBZ/nnUNet](https://github.com/JavierBZ/nnUNet)**

This fork comments out the intensity-based data augmentations that nnUNet applies by default.
Those augmentations are appropriate for grayscale images but distort the data in this setting, where **the inputs are binary vessel masks** (not raw MRA intensities). Disabling them yields more stable training.

Follow the nnUNet documentation for dataset fingerprinting, planning, and training. Use `Dataset100` as the dataset identifier, which matches the folder names generated by `prepare_data.py`.

### TTA Post-processing for nnUNet

After running nnUNet inference on all TTA test folders, use `custom_tta_nnunet.py` to aggregate the predictions:

```bash
python custom_tta_nnunet.py
```

Configure the path constants at the top of the `__main__` block before running. The script loads the pickled random parameters saved by `prepare_data.py` and applies the same aggregation strategy (mode voting with coordinate-mapping or inverse-transform) used for the UNet/CSNet TTA evaluation.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bisbal2025automated,
  title={Automated Labeling of Intracranial Arteries with Uncertainty Quantification Using Deep Learning},
  author={Bisbal, Javier and Winter, Patrick and Jofre, Sebastian and Ponce, Aaron and Ansari, Sameer A and Abdalla, Ramez and Markl, Michael and Odeback, Oliver Welin and Uribe, Sergio and Tejos, Cristian and others},
  journal={arXiv preprint arXiv:2509.17726},
  year={2025}
}
```
