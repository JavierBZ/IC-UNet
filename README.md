# IC-UNet
Intracranial labelling using 3D semantic segmentation UNet

## Step 1
Create conda environment with

`cd env`

`conda env create -f env.yml`

## Step 2

Add segmentation file to `data` and weights to `models`

Convert segmentation to nrrd format:
* Change ` tof_mask_struct_path` (e.g `data/tof_mask_struct.mat`) directory in `to_nrrd.py`

## Step 3

run `python main.py`

Label predictions will be saved on `Prediction/test/1111-pred.nrrd`

## Step 4 (optional)

Add "vel_struct

