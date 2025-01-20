
import time
import logging
import torch
import pickle
from monai.data import decollate_batch

import numpy as np

import os

import h5py
import nrrd


from monai.transforms.utils import allow_missing_keys_mode

from monai.handlers.utils import from_engine

from scipy.spatial import cKDTree

from monai.transforms import (
    BatchInverseTransform,
    Invertd,
    Compose, 
    AsDiscrete,
    AsDiscreted,
    Activationsd,
    Activations,
    CropForeground,
    CropForegroundd,
    Resized,
    ResizeWithPadOrCropd,
    Spacingd,
    RandAffined,
    CenterSpatialCropd,
    Identityd,
    Affined,
    KeepLargestConnectedComponent,
)

from monai.data import MetaTensor

from tqdm import tqdm
import pickle

# Efficient computation of 8 closest points using cKDTree
def find_k_closest_points(original, transformed, k=8, chunk_size=10_000):
    num_orig = original.shape[0]
    num_trans = transformed.shape[0]

    # Initialize placeholders for k closest indices and distances
    k_closest_indices = np.zeros((num_orig, k), dtype=np.int64)
    k_closest_distances = np.full((num_orig, k), float('inf'))

    # Build the KDTree
    tree = cKDTree(transformed)

    # Process in chunks
    for i in tqdm(range(0, num_orig, chunk_size), desc="Processing chunks"):
        chunk = original[i:i + chunk_size]  # Shape: [chunk_size, 3]
        distances, indices = tree.query(chunk, k=k)

        if k == 1:
            k_closest_indices[i:i + chunk_size] = np.expand_dims(indices,axis=-1)
            k_closest_distances[i:i + chunk_size] = np.expand_dims(distances,axis=-1)
        else:
            k_closest_indices[i:i + chunk_size] = indices
            k_closest_distances[i:i + chunk_size] = distances

    return k_closest_indices, k_closest_distances

def compute_final_labels(batch_orig, transform_label, transforms, device):
    
    dimensions = batch_orig['image'].shape[1:]
    x = np.linspace(0, dimensions[0] - 1, dimensions[0])
    y = np.linspace(0, dimensions[1] - 1, dimensions[1])
    z = np.linspace(0, dimensions[2] - 1, dimensions[2])
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')

    X_dic = {'image':batch_orig['image'].type(torch.float32),'label':MetaTensor(torch.tensor(X).unsqueeze(0).type(torch.float32), meta=batch_orig['label'].meta)}
    Y_dic = {'image':batch_orig['image'].type(torch.float32),'label':MetaTensor(torch.tensor(Y).unsqueeze(0).type(torch.float32), meta=batch_orig['label'].meta)}
    Z_dic = {'image':batch_orig['image'].type(torch.float32),'label':MetaTensor(torch.tensor(Z).unsqueeze(0).type(torch.float32), meta=batch_orig['label'].meta)}

    X_trans_b = transforms(X_dic)
    Y_trans_b = transforms(Y_dic)
    Z_trans_b = transforms(Z_dic)
    X_trans = X_trans_b['label'].flatten()
    Y_trans = Y_trans_b['label'].flatten()
    Z_trans = Z_trans_b['label'].flatten()

    SEG_orig = batch_orig['image'].flatten()
    SEG_trans = X_trans_b['image'].flatten()

    X = torch.tensor(X).type(torch.float32).flatten()
    Y = torch.tensor(Y).type(torch.float32).flatten()
    Z = torch.tensor(Z).type(torch.float32).flatten()

    X = X[SEG_orig > 0].unsqueeze(1)
    Y = Y[SEG_orig > 0].unsqueeze(1)
    Z = Z[SEG_orig > 0].unsqueeze(1)

    X_trans = X_trans[SEG_trans > 0].unsqueeze(1)
    Y_trans = Y_trans[SEG_trans > 0].unsqueeze(1)
    Z_trans = Z_trans[SEG_trans > 0].unsqueeze(1)

    # Stack the coordinates into tensors for vectorized computation
    original_coords = torch.cat((X, Y, Z), dim=1).cpu().numpy()  # Shape: [N, 3]
    transformed_coords = torch.cat((X_trans, Y_trans, Z_trans), dim=1).cpu().numpy()  # Shape: [M, 3]

    # # Round the transformed coordinates
    rounded_transformed_coords = np.round(transformed_coords)

    closest_indices, closest_distances = find_k_closest_points(original_coords, rounded_transformed_coords,k=1)

    # Initialize an empty tensor to store the weighted values
    closest_labels = torch.zeros([10, *batch_orig["label"].shape[1:]], dtype=torch.float32,device=device)
    transform_label = transform_label.to(device)

    for channel in range(transform_label.shape[0]):
        # Extract the label data for the current channel
        label_data = transform_label[channel,...].flatten()[SEG_trans.flatten() > 0]
        # Gather the label of the closest point for each coordinate
        closest_label = label_data[closest_indices[:,0]]  # Use only the closest point
        # Reshape the closest labels to the original shape and store them in the closest_labels tensor
        closest_labels_aux = closest_labels[channel,...].flatten()
        closest_labels_aux[SEG_orig.flatten() > 0] = closest_label
        closest_labels[channel,...] = closest_labels_aux.reshape(batch_orig["label"].shape[1:])

    return closest_labels



def validacion_11_clase(directorio,modelo,cargador_de_datos,funcion_de_perdida,metrica_de_evaluacion,epoca,log=False,device="cuda",
                fold=0,valid_files=None,val_transforms=None,cargador_min=None,minimal_transforms=None):

    if not os.path.exists(directorio):
        os.mkdir(directorio)

    post_label = Compose([AsDiscrete(to_onehot=11)])
    # post_pred = Compose([AsDiscrete(argmax=True, to_onehot=11, threshold=0.5)])
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=11, threshold=0.5),KeepLargestConnectedComponent(applied_labels=[1,2,3,4,5,6,7,8,9])])

    detecciones = []
    true_etiqueta = []
    inputs_nn = []

    modelo.eval()

    name_vessels = ["BA", "LACA", "LICA", "LMCA", "RACA", "RICA", "RMCA", "RPCA", "LPCA", "NA"]

    with torch.no_grad():
        
        # for paso, (batch_data, batch_orig) in enumerate(zip(cargador_de_datos,cargador_min)):
        for paso, FILE in enumerate(valid_files):

            batch_data = val_transforms(FILE)
            batch_orig = minimal_transforms(FILE)
            
            info = nrrd.read_header(FILE['image'])
           
            spacing = [info['space directions'][0][0],info['space directions'][1][1],info['space directions'][2][2]]
            # ID = int(valid_files[paso]["image"].split('/')[-2].split("_")[1])
            ID = 1111

            datos_keys = ["image","label"]
            target_size = (256,256,128)
            
            coor_transforms = Compose(
                    [   
                    CropForegroundd(keys=datos_keys, source_key="image", margin=30, select_fn=lambda x: x > 0),
                    # Spacingd(keys=datos_keys, pixdim=(0.4,0.4,0.4), mode=("nearest", "nearest")),
                    Resized(spatial_size=max(target_size),size_mode="longest", 
                    mode='nearest', keys=datos_keys),
                    ResizeWithPadOrCropd(spatial_size=target_size, keys=datos_keys),
                    # Affined(keys=datos_keys, rotate_params=(np.pi/10, np.pi/10, np.pi/10) ,mode=['nearest','nearest'],
                    # translate_params=(3,3,3))
                    ]
                )


            inputs, labels  = (
                            batch_data["image"].unsqueeze(0).to(device),
                            batch_data["label"].unsqueeze(0).to(device)
                        )
            
            labels = labels[:,0,:,:,:] + labels[:,1,:,:,:] * 2 + labels[:,2,:,:,:] * 3 + labels[:,3,:,:,:] * 4 + labels[:,4,:,:,:] * 5 \
                + labels[:,5,:,:,:] * 6 + labels[:,6,:,:,:] * 7 + labels[:,7,:,:,:] * 8 + labels[:,8,:,:,:] * 9 + labels[:,9,:,:,:] * 10
            labels = labels.unsqueeze(1)
            
            inputs_orig, labels_orig  = (
                            batch_orig["image"].unsqueeze(0).to(device),
                            batch_orig["label"].unsqueeze(0).to(device)
                        )
        
            labels_orig = labels_orig[:,0,:,:,:] + labels_orig[:,1,:,:,:] * 2 + labels_orig[:,2,:,:,:] * 3 + labels_orig[:,3,:,:,:] * 4 + labels_orig[:,4,:,:,:] * 5 \
                + labels_orig[:,5,:,:,:] * 6 + labels_orig[:,6,:,:,:] * 7 + labels_orig[:,7,:,:,:] * 8 + labels_orig[:,8,:,:,:] * 9 + labels_orig[:,9,:,:,:] * 10
            labels_orig = labels_orig.unsqueeze(1)

            logit_map = modelo(inputs)

            labels_pred = [post_pred(i) for i in decollate_batch(logit_map)]


            labels_pred = labels_pred[0][1:,...]            


            closest_labels = compute_final_labels(batch_orig, labels_pred, coor_transforms, device)

            labels_pred = closest_labels[0,:,:,:] + closest_labels[1,:,:,:] * 2 + closest_labels[2,:,:,:] * 3 + closest_labels[3,:,:,:] * 4 + closest_labels[4,:,:,:] * 5 \
                + closest_labels[5,:,:,:] * 6 + closest_labels[6,:,:,:] * 7 + closest_labels[7,:,:,:] * 8 + closest_labels[8,:,:,:] * 9 + closest_labels[9,:,:,:] * 10

            labels_pred = labels_pred.unsqueeze(0).unsqueeze(0)


            val_outputs = [post_pred(i) for i in decollate_batch(logit_map)]

            val_outputs_inverted = [post_label(i) for i in decollate_batch(labels_pred)]

            labels_orig = [post_label(i) for i in decollate_batch(labels_orig)]



            if len(val_outputs) == 1:

                ID = 1111

                nrrd.write(f"{directorio}/{ID}-input.nrrd", np.squeeze(inputs_orig.detach().cpu().numpy()).astype(np.uint8))

                true_data = np.squeeze(labels_orig[0].detach().cpu().numpy()).astype(np.uint8)
                pred_data = np.squeeze(val_outputs_inverted[0].detach().cpu().numpy()).astype(np.uint8)
                pred_data_sum = np.zeros(pred_data.shape[1:], dtype=np.uint8)
                true_data_sum = np.zeros(true_data.shape[1:], dtype=np.uint8)
                for label in range(1,pred_data.shape[0]):
                    pred_data_sum += pred_data[label,...] * (label)
                    true_data_sum += true_data[label,...] * (label)
                nrrd.write(f"{directorio}/{ID}-pred.nrrd", pred_data_sum)


        # metric = dice_metric.aggregate().item()
        

    return detecciones, true_etiqueta, inputs