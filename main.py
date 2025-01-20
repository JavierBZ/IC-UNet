
import torch
import numpy as np
# from monai.data import Dataset, DataLoader
from monai.data import CacheDataset, ThreadDataLoader, DataLoader
from monai.networks.nets import UNet

import random
import torch
import os
import yaml

from src.dataloader import transformaciones

from src.modelos import modelo_neuronal

from src.pred import validacion_11_clase

import time


def evaluation(config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Se le pasa el config de wandb
    modelo = modelo_neuronal(config_interna).to(device) # -> configuracion wandb
    
    train_files = [{"image": "data/1111.nrrd" , "label": "data/1111.nrrd"}]
    valid_files = [{"image": "data/1111.nrrd" , "label": "data/1111.nrrd"}]
    
    train_transforms, val_transforms, minimal_transforms = transformaciones(config_interna)

    set_data_val_min = CacheDataset(data=valid_files, transform=minimal_transforms,cache_rate=1.0, num_workers=0,copy_cache=False)
    set_data_val = CacheDataset(data=valid_files, transform=val_transforms,cache_rate=1.0, num_workers=0,copy_cache=False)
    cargador_val = DataLoader(set_data_val, batch_size=config["batch_size_test"], num_workers=0, shuffle=False)
    cargador_min = DataLoader(set_data_val_min, batch_size=config["batch_size_test"], num_workers=0, shuffle=False)

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    modelo_best = modelo_neuronal(config_interna)
    modelo_best.load_state_dict(torch.load(config_interna["model_evaluation_path"],map_location=map_location))

    if config_interna["clases"] == 11:
        detecciones, true_etiqueta, inputs = validacion_11_clase(config_interna["model_evaluation_output"],modelo_best,cargador_val,None,None,0,
                    log=False,fold=config_interna["fold"],device=device,valid_files=valid_files,val_transforms=val_transforms,cargador_min=cargador_min,minimal_transforms=minimal_transforms)


if __name__ == "__main__":

    with open('config/config.yml', 'r') as archivo:
        config = yaml.safe_load(archivo)

    config["model_path"] = "/models/fold_1best_model.pt"
    config_interna = config["base"]
    config_interna["fold"] = 0
    evaluation(config_interna)

