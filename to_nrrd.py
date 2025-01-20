import nrrd
from scipy.io import loadmat
import numpy as np


def to_nrrd(mat_file, nrrd_file):

    # Load the mat file
    tof_mask_struct = loadmat(tof_mask_struct_path,struct_as_record=False)

    data = tof_mask_struct['mrStruct'][0,0].dataAy
    VoxelSize = tof_mask_struct['mrStruct'][0,0].vox

    kinds = ['domain', 'domain', 'domain']
    header = {
        'kinds': kinds,
        # 'units': ['mm', 'mm', 'mm'],
        'space': 'right-anterior-superior',
        'sizes': data.shape,
        'space origin': [0., 0., 0.],
        # 'space': 'left-posterior-superior',
        # 'spacings': VoxelSize['VoxelSize'][0],                
        'space directions': np.array([
        [VoxelSize[0][0], 0., 0.],
        [0., VoxelSize[0][1], 0.],
        [0., 0., VoxelSize[0][2]]
        ])
        # 'space directions': np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) 
            }

    nrrd.write(nrrd_file, data, header)
    
    # # Save the volume as nrrd
    # nrrd.write(nrrd_file, volume)


if  __name__ == '__main__':
    tof_mask_struct_path = 'data/tof_mask_struct.mat'
    nrrd_file = 'data/1111.nrrd'
    to_nrrd(tof_mask_struct_path, nrrd_file)
