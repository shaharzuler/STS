
import os

import numpy as np 
from torch.utils.data import Dataset
import h5py





def create_sts_dataset(args):
    name =  args.dataset_name
    print(name)
    if name == 'cardio':
        dataset = CardioDataset(args)
    return dataset


class CardioDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config=config
        
        self.get_pair(config)

    def get_pair(self, config):
        template_h5 = h5py.File(config.template_h5_path, 'r')
        template_timestep = config.template_timestep
        template_vertices = template_h5[f"{template_timestep}_vertices"][:]

        unlabeled_h5 = h5py.File(config.unlabeled_h5_path, 'r')
        unlabeled_timestep = config.unlabeled_timestep
        unlabeled_vertices = unlabeled_h5[f"{unlabeled_timestep}_vertices"][:]

        self.num_points = min(template_vertices.shape[0], unlabeled_vertices.shape[0], self.config.n_points)

        if os.path.isfile(self.config.consistant_shape_indices_filename):
            indices_npz_file = np.load(self.config.consistant_shape_indices_filename)
            self.template_indices = indices_npz_file["template_indices"]
            self.unlabeled_indices = indices_npz_file["unlabeled_indices"]
        else:
            self.template_indices = np.random.choice(template_vertices.shape[0], self.num_points, replace=False)
            self.unlabeled_indices = np.random.choice(unlabeled_vertices.shape[0], self.num_points, replace=False)   

        template_vertices_sampeled = template_vertices[self.template_indices]
        template_eigenvectors = template_h5[f"{template_timestep}_eigenvectors"][:]           
        unlabeled_vertices_sampeled = unlabeled_vertices[self.unlabeled_indices]
        unlabeled_eigenvectors = unlabeled_h5[f"{unlabeled_timestep}_eigenvectors"][:]
        
        self.k_lbo = min(template_eigenvectors.shape[1], unlabeled_eigenvectors.shape[1], self.config.k_lbo)

        template_eigenvectors_sampeled = template_eigenvectors[self.template_indices, :self.k_lbo]
        template_area_weights = template_h5[f"{template_timestep}_area_weights"][:]
        template_area_weights_sampeled = template_area_weights[self.template_indices]
        template_eigenvalues = template_h5[f"{template_timestep}_eigenvalues"][:][:self.k_lbo]

        unlabeled_eigenvectors_sampeled = unlabeled_eigenvectors[self.unlabeled_indices, :self.k_lbo]
        unlabeled_area_weights = unlabeled_h5[f"{unlabeled_timestep}_area_weights"][:]
        unlabeled_area_weights_sampeled = unlabeled_area_weights[self.unlabeled_indices]
        unlabeled_eigenvalues = unlabeled_h5[f"{unlabeled_timestep}_eigenvalues"][:][:self.k_lbo]

        self.pair = {}
        self.pair['key'] = f"{template_timestep}_{unlabeled_timestep}"
        self.pair["vertices"] = np.concatenate([template_vertices_sampeled[...,None], unlabeled_vertices_sampeled[...,None]], axis=-1).astype(np.float16)
        self.pair["eigenvectors"] = np.concatenate([template_eigenvectors_sampeled[...,None], unlabeled_eigenvectors_sampeled[...,None]], axis=-1).astype(np.float16)
        self.pair["area_weights"] = np.concatenate([template_area_weights_sampeled[...,None], unlabeled_area_weights_sampeled[...,None]], axis=-1).astype(np.float16)
        self.pair["eigenvalues"] = np.concatenate([template_eigenvalues[...,None], unlabeled_eigenvalues[...,None]], axis=-1).astype(np.float16)
        self.pair["indices"] = np.concatenate([self.template_indices[...,None], self.unlabeled_indices[...,None]], axis=-1)
        


    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.pair
        