
import numpy as np
import torch
from .DPC.utils.tensor_utils import knn


def flow_to_corr(flow:np.ndarray, target_pc:np.ndarray, source_pc:np.ndarray): #[N, 3] # duplication of flow_n_corr_utils.flow_to_corr()
    target_as_int = np.round(target_pc).astype(int)
    flow_in_target_coords = flow[target_as_int[:,0], target_as_int[:,1], target_as_int[:,2], :]
    source_estimated_coords = target_as_int + flow_in_target_coords
        
    indices = knn(torch.tensor(source_estimated_coords), torch.tensor(source_pc), 1)[1]

    # now source_estimated_coords[indices] =~ source_pc
    # now target_pc[indices] corresponds to source_pc

    return indices


# import h5py
# source_pc_f = h5py.File("/home/shahar/cardio_corr/outputs/synthetic_dataset37/28/orig/h5_datasets/point_cloud_from_mesh_smooth_dataset.hdf5")
# source_verts = source_pc_f["28_vertices"][:]
# target_pc_f = h5py.File("/home/shahar/cardio_corr/outputs/synthetic_dataset37/18/orig/h5_datasets/point_cloud_from_mesh_smooth_dataset.hdf5")
# target_verts = target_pc_f["18_vertices"][:]
# flow = np.load("/home/shahar/cardio_corr/outputs/synthetic_dataset37/thetas_90.0_0.0_rs_0.9_0.9_h_0.9_random_3_0.4_mask_True_blur_radious_7/flow_for_mask_thetas_90.0_0.0_rs_0.9_0.9_h_0.9_random_3_0.4_mask_True_blur_radious_7.npy")

# flow_to_corr(flow, target_verts, source_verts)