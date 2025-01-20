import numpy as np
import nrrd
import json
import torch
import matplotlib.pyplot as plt

from monai.data import MetaTensor

from monai.transforms import (
    #AddChanneld,
    EnsureTyped,
    Rand3DElastic,
    Compose,
    Transform,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    ToTensord,
    ScaleIntensityd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    Flipd,
    RandAffined,
    Resize,
    ResizeWithPadOrCropd,
    #AsChannelFirst,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    AsDiscrete,
    AsDiscreted,
    Padd,
    Pad,
    MapTransform,
    InvertibleTransform,
    CenterSpatialCropd,
    Affined,
    
)


class ConvertToMultiChannelBasedOnICClassesd(MapTransform):
    """
    Convert labels to multi channels based on intracranial classes:
    label 1 is the Basilar Artery (BA)
    label 2 is the Left Anterior Cerebral Artery (LACA)
    label 3 is the Left Internal Carotid artery (LICA)
    label 4 is the Left Middle Cerebral Artery (LMCA)
    label 5 is the Right Anterior Cerebral Artery (RACA)
    label 6 is the Right Internal Carotid artery (RICA)
    label 7 is the Right Middle Cerebral Artery (RMCA)
    label 8 is the Right Posterior Cerebral Artery (RPCA)
    label 9 is the Left Posterior Cerebral Artery (LPCA)
    label 10 is Non-annotated vessels (NA)

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = torch.stack([(d[key] == i).float() for i in range(1, 11)], axis=0)
        return d
    
    def inverse(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = torch.argmax(d[key], dim=0)
        return d




def transformaciones(config):
    if config["clases"] == 2:
        train_transforms = Compose(
            [   Resized(keys=["image","arteria_pulmonar"], spatial_size=(128,128,64), mode=('area')),
                ToTensord(keys=["image","arteria_pulmonar"]),
            ]
        )

        val_transforms = Compose(
            [   Resized(keys=["image","arteria_pulmonar"], spatial_size=(128,128,64), mode=('area')),
                ToTensord(keys=["image","arteria_pulmonar"]),
            ]
        )

    elif config["clases"] == 4:
        train_transforms = Compose(
            [   ToTensord(keys=["image","seg_art_principal","seg_art_izquierda","seg_art_derecha"]),
            ]
        )

        val_transforms = Compose(
            [   ToTensord(keys=["image","seg_art_principal","seg_art_izquierda","seg_art_derecha"]),
            ]
        )
    elif config["clases"] == 11:

        # datos_keys = [
        #     "image", "ba", "laca", "lica", "lmca", "raca", "rica", "rmca", "rpca", "lpca", "NA"
        # ]

        datos_keys = [
            "image", "label"
        ]

        target_size = (256,256,128)

        """ train_transforms = Compose(
            [   
                Resized(spatial_size=target_size, mode='nearest', keys=datos_keys),
                ToTensord(keys=datos_keys),
            ]
        )

        val_transforms = Compose(
            [   
                Resized(spatial_size=target_size, mode='nearest', keys=datos_keys),
                ToTensord(keys=datos_keys),
            ]
        ) """


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_transforms = Compose(
            [   
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnICClassesd(keys="label"),
                Orientationd(keys=datos_keys, axcodes="RAS"),
                CropForegroundd(keys=datos_keys, source_key="image", margin=30, select_fn=lambda x: x > 0),
                RandRotated(keys=datos_keys, prob=0.5, range_x=np.pi/8, range_y=np.pi/8, range_z=np.pi/10,
                 mode='nearest'),
                Resized(spatial_size=max(target_size),size_mode="longest",
                mode='nearest', keys=datos_keys),
                ResizeWithPadOrCropd(spatial_size=target_size, keys=datos_keys),
                # RandAffined(keys=datos_keys, prob=0.5, rotate_range=(np.pi/8, np.pi/8, np.pi/8), 
                #             translate_range=(15,15,15), scale_range=(0.1, 0.1, 0.1), mode='nearest'),
                # #RandRotated(keys=datos_keys, prob=0.5, range_y=3.0, mode='nearest'),
                
                # EnsureTyped(keys=datos_keys, device=device, track_meta=False),
                # ToTensord(keys=datos_keys),
            ]
        )

        val_transforms = Compose(
            [   
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnICClassesd(keys="label"),
                Orientationd(keys=datos_keys, axcodes="RAS"),
                CropForegroundd(keys=datos_keys, source_key="image", margin=30, select_fn=lambda x: x > 0),
                Resized(spatial_size=max(target_size),size_mode="longest", 
                mode='nearest', keys=datos_keys),
                ResizeWithPadOrCropd(spatial_size=target_size, keys=datos_keys),
                # EnsureTyped(keys=datos_keys, device=device, track_meta=False),
                # ToTensord(keys=datos_keys),
            ]
        )

        # val_transforms = Compose(
        #     [   
        #         LoadImaged(keys=["image", "label"]),
        #         EnsureChannelFirstd(keys="image"),
        #         ConvertToMultiChannelBasedOnICClassesd(keys="label"),
        #         Orientationd(keys=datos_keys, axcodes="RAS"),
        #         CropForegroundd(keys=datos_keys, source_key="image", margin=30, select_fn=lambda x: x > 0),
        #         Spacingd(keys=datos_keys, pixdim=(0.4,0.4,0.4), mode=("nearest", "nearest")),
        #         # Resized(spatial_size=max(target_size),size_mode="longest", 
        #         # mode='nearest', keys=datos_keys),
        #         # CenterSpatialCropd(keys=datos_keys, roi_size=(50,50,25)),
        #         # Affined(keys=datos_keys, rotate_params=(np.pi/10, np.pi/10, np.pi/10) ,mode=['nearest','nearest'],
        #         #     translate_params=(3,3,3))

        #         # ResizeWithPadOrCropd(spatial_size=target_size, keys=datos_keys),
        #         # EnsureTyped(keys=datos_keys, device=device, track_meta=False),
        #         # ToTensord(keys=datos_keys),
        #     ]
        # )
        
        minimal_transforms = Compose(
            [   
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnICClassesd(keys="label"),
                Orientationd(keys=datos_keys, axcodes="RAS"),
            ]
        )

        # train_transforms = Compose(
        #     [   
        #         RandRotated(keys=datos_keys, prob=0.5, range_x=np.pi/12, range_y=np.pi/12, range_z=np.pi/12,
        #          mode=['nearest']*11),
        #         # RandAffined(keys=datos_keys, prob=0.5, rotate_range=(np.pi/12, np.pi/12, np.pi/12), 
        #         #             translate_range=(15,15,15), scale_range=(0.1, 0.1, 0.1), mode='nearest'),
        #         Orientationd(keys=datos_keys, axcodes="RAS"),
        #         # #RandRotated(keys=datos_keys, prob=0.5, range_y=3.0, mode='nearest'),
        #         Resized(spatial_size=max(target_size),size_mode="longest",
        #         mode=['nearest']*11, keys=datos_keys),
        #         ResizeWithPadOrCropd(spatial_size=target_size, keys=datos_keys),
        #         # EnsureTyped(keys=datos_keys, device=device, track_meta=False),
        #         # ToTensord(keys=datos_keys),
        #     ]
        # )

        # val_transforms = Compose(
        #     [   
        #         # RandRotated(keys=datos_keys, prob=0.5, range_x=10.0, mode='nearest'),
        #         #RandRotated(keys=datos_keys, prob=0.5, range_y=3.0, mode='nearest'),
        #         Orientationd(keys=datos_keys, axcodes="RAS"),
        #         Resized(spatial_size=max(target_size),size_mode="longest", 
        #         mode=['nearest']*11, keys=datos_keys),
        #         ResizeWithPadOrCropd(spatial_size=target_size, keys=datos_keys),
        #         # EnsureTyped(keys=datos_keys, device=device, track_meta=False),
        #         # ToTensord(keys=datos_keys),
        #     ]
        # )


    return train_transforms, val_transforms, minimal_transforms
    