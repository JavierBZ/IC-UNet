import numpy as np
import torch

from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandRotated,
    Resized,
    ResizeWithPadOrCropd,
    Spacingd,
    RandAffined,
    Affined,
    EnsureChannelFirstd,
    MapTransform,
)


class ConvertToMultiChannelBasedOnICClassesd(MapTransform):
    """
    Convert an integer-label volume into a 10-channel one-hot tensor based on
    the intracranial artery class definitions:

        Channel 0  – Basilar Artery (BA)
        Channel 1  – Left Anterior Cerebral Artery (LACA)
        Channel 2  – Left Internal Carotid Artery (LICA)
        Channel 3  – Left Middle Cerebral Artery (LMCA)
        Channel 4  – Right Anterior Cerebral Artery (RACA)
        Channel 5  – Right Internal Carotid Artery (RICA)
        Channel 6  – Right Middle Cerebral Artery (RMCA)
        Channel 7  – Right Posterior Cerebral Artery (RPCA)
        Channel 8  – Left Posterior Cerebral Artery (LPCA)
        Channel 9  – Non-annotated vessels (NA)
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


def transformations(config, params=None, target_spacing=None):
    """
    Build training, validation, minimal, and coordinate transform pipelines.

    Args:
        config         (dict): Model configuration; must contain 'classes'.
        params         (dict): Optional affine augmentation parameters for validation TTA.
                               Keys: 'rotate_params', 'translate_params', 'scale_params'.
        target_spacing (tuple): If provided, use Spacingd instead of Resized.

    Returns:
        (train_transforms, val_transforms, minimal_transforms, coor_transforms)
    """
    if config["classes"] != 11:
        raise ValueError(f"Unsupported number of classes: {config['classes']}. Expected 11.")

    keys        = ["image", "label"]
    target_size = (256, 256, 128)

    if target_spacing is not None:
        resize_transform = Spacingd(keys=keys, pixdim=target_spacing, mode=("nearest", "nearest"))
    else:
        resize_transform = Resized(
            spatial_size=max(target_size), size_mode="longest", mode="nearest", keys=keys
        )

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnICClassesd(keys="label"),
        Orientationd(keys=keys, axcodes="RAS"),
        CropForegroundd(keys=keys, source_key="image", margin=30, select_fn=lambda x: x > 0),
        RandRotated(keys=keys, prob=0.5,
                    range_x=np.pi / 8, range_y=np.pi / 8, range_z=np.pi / 8,
                    mode="nearest"),
        resize_transform,
        ResizeWithPadOrCropd(spatial_size=target_size, keys=keys),
    ])

    if params is not None:
        affine_transform = Affined(
            keys=keys,
            rotate_params=params["rotate_params"],
            translate_params=params["translate_params"],
            scale_params=params["scale_params"],
            mode=["nearest", "nearest"],
        )

        val_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnICClassesd(keys="label"),
            Orientationd(keys=keys, axcodes="RAS"),
            CropForegroundd(keys=keys, source_key="image", margin=30, select_fn=lambda x: x > 0),
            affine_transform,
            resize_transform,
            ResizeWithPadOrCropd(spatial_size=target_size, keys=keys),
        ])

        coor_transforms = Compose([
            CropForegroundd(keys=keys, source_key="image", margin=30, select_fn=lambda x: x > 0),
            affine_transform,
            resize_transform,
            ResizeWithPadOrCropd(spatial_size=target_size, keys=keys),
        ])

    else:
        val_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnICClassesd(keys="label"),
            Orientationd(keys=keys, axcodes="RAS"),
            CropForegroundd(keys=keys, source_key="image", margin=30, select_fn=lambda x: x > 0),
            resize_transform,
            ResizeWithPadOrCropd(spatial_size=target_size, keys=keys),
        ])

        coor_transforms = Compose([
            CropForegroundd(keys=keys, source_key="image", margin=30, select_fn=lambda x: x > 0),
            resize_transform,
            ResizeWithPadOrCropd(spatial_size=target_size, keys=keys),
        ])

    minimal_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnICClassesd(keys="label"),
        Orientationd(keys=keys, axcodes="RAS"),
    ])

    return train_transforms, val_transforms, minimal_transforms, coor_transforms
