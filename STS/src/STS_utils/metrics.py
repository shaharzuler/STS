from locale import NOEXPR
from tkinter.messagebox import NO
from numpy.ma.core import doc_note
# from torchmetrics import Accuracy
import torch
import numpy as np
import pygeodesic.geodesic as geodesic
import potpourri3d as pp3d
import time
import h5py
import matplotlib.pyplot as plt
import os, sys
import numpy as np


class AccuracyAssumeEyeForFlow_NEW():
    def __init__(self, k=None):
        self.top_k = k

    def __call__(self, mapping: torch.Tensor):

        if len(mapping.shape) == 2:
            mapping = mapping.unsqueeze(0)
        # labels = torch.arange(mapping.shape[1]).repeat(mapping.shape[0],1).to(mapping.device)
        labels = torch.arange(mapping.shape[1]).to(mapping.device)
        # correct = mapping == labels[:,:, None]
        correct = mapping == labels[None, :, None]

        # if not (correct == correct_2).all():
        #     print('Erorr!')


        return torch.sum(correct) / (correct.shape[0] * correct.shape[1])

    def calc_k_acc_list(self, mapping):

        max_k = mapping.shape[-1]
        acc_list = []
        k_list = []
        for i in range(1, max_k, 2):
            curr_acc = self.__call__(mapping[:,:i])
            acc_list.append(curr_acc.item())
            k_list.append(i)

        return k_list, acc_list

def compute_geo_dist_batch_old(mapping, geo_dist, is_P_mapping= True, dont_mean=True, full_arr = False, gt=None ):
    # mapping: batch, N, 1
    max_dist = geo_dist.max()

    B= None

    if not is_P_mapping:
        B, source, target = mapping.shape
        N = target
        if source > target:
            if gt is not None:
                mapping = mapping[:,gt[0].long(),:] 
                mapping = mapping[:,:N,:]                  ## assume batch size = 1 
                gt = None
            else:
                mapping = mapping[:,:target,:]
            mapping = mapping.argmax(-1)

        elif source < target:
            mapping = mapping.argmax(-1)
            N=source
        else:
            mapping = mapping.argmax(-1)

    else:
        try:
            B, N  = mapping.shape
            if N > geo_dist.shape[-1]:
                mapping = mapping[:,:geo_dist.shape[-1]]
                N = geo_dist.shape[-1]
                if gt is not None:
                    gt = gt[:,:N]
        except:
            print('stop me')
    device = geo_dist.device
    mapping = mapping.clone().to(device)

    if gt is None:
        indexes = torch.arange(N).repeat(B,1).reshape(-1).to(device)
    else:
        indexes = gt.reshape(-1).to(device)

    indexes_batch = torch.arange(B).repeat(N, 1).permute(1,0).reshape(-1).to(device)
    try:
        all_index = torch.vstack([indexes_batch, mapping.reshape(-1), indexes])
    except:
        pass

    s = torch.sparse_coo_tensor(all_index, torch.ones(all_index.shape[-1]).to(device)).to_dense()
    b, nn, n = s.shape

    m = min(n, nn)
    p =  geo_dist.shape[-1]
    ss = torch.zeros((b, p, p)).to(device)
    try:
        ss[:,:s.shape[-2],:s.shape[-1]] = s      
    except:
        print('Error')

    try:
        geo_dist_b = ss * geo_dist
    except:
        print('Omri')
    geo_dist_arr = geo_dist_b.sum(1)
    geo_dist_9 = geo_dist_arr.quantile(0.9, 1)
    geo_dist_b = geo_dist_arr.mean(1)
    if not dont_mean:
        geo_dist_b = torch.mean(geo_dist_b)

    acc_arr = []
    if not full_arr:

        geo_dist_arr = geo_dist_arr[:,:,None]/max_dist <= torch.arange(0, 1, step=0.002).to(device)[None, None, :]
        geo_dist_arr = geo_dist_arr.sum(1) / p


    return geo_dist_b, geo_dist_arr, torch.mean(geo_dist_9)

def compute_geo_dist_batch(mapping, geo_dist, is_P_mapping= True, dont_mean=True, full_arr = False, gt=None , samples=None):
    # mapping: batch, N, 1
    if samples is not None:
        samples = torch.Tensor(samples)

    B= geo_dist.shape[0]
    N = geo_dist.shape[1]
    if not is_P_mapping:
        B, source, target = mapping.shape
        N = target
        mapping = mapping.argmax(-1)
    device = geo_dist.device
    mapping = mapping.clone().to(device)


    if gt is None:
        indexes = torch.arange(N).repeat(B,1).to(device)
    else:
        indexes = gt.to(device)

    geo_dist_tmp = []
    max_dist = []

    for i in range(B):
        curr_max = geo_dist[i].max()
        max_dist.append(curr_max)
        tmp = geo_dist[i,indexes[i].long(), mapping[i].long()]
        geo_dist_tmp.append(tmp.unsqueeze(0))

    geo_dist_arr = torch.cat(geo_dist_tmp, dim=0).float()
    max_dist = torch.Tensor(max_dist).to(device)
    if len(geo_dist_arr.shape) ==1:
        geo_dist_arr = geo_dist_arr[None,:]
    
    if samples is not None:
        geo_dist_arr = geo_dist_arr[:, samples.long()]

    try:
        geo_dist_9 = geo_dist_arr.quantile(0.9, 1)
    except:
        geo_dist_9 = torch.Tensor([-1])
    geo_dist_b = geo_dist_arr.mean(1)
    if not dont_mean:
        geo_dist_b = torch.mean(geo_dist_b)

    acc_arr = []
    if not full_arr:

        geo_dist_arr =  geo_dist_arr[:,:,None]/max_dist[:,None,None]  <= torch.arange(0, 1, step=0.002).to(device)[None, None, :]
        p = geo_dist_arr.shape[-2]
        geo_dist_arr = geo_dist_arr.sum(1).float() / p

    return geo_dist_b, geo_dist_arr, torch.mean(geo_dist_9)

class GeodisicDist(object):

    def __init__(self, P, pc, faces, gt_mapping=None, distances_arr=None, is_P_mapping= False, dist_arr=None):
        '''

        :param P:
        :param pc: [N, 3 (D)]   NO BATCH!!!!!!
        :param faces:
        :param label:
        '''


        if torch.is_tensor(P):
            P = P.detach().cpu().numpy()

        if torch.is_tensor(pc):
            pc = pc.clone().detach().cpu().numpy()

        if pc.shape[0] == 3: # todo - change to a configurable param
            pc = pc.T
        if distances_arr is None:
            self.distances = np.arange(0, 1, step=0.05)
        else:
            self.distances = distances_arr

        self.P = P
        self.N = pc.shape[0]
        if is_P_mapping:
            self.mapping = P
        else:
            self.mapping = np.argmax(P, axis=1)
        # s = time.time()
        self.geo_func = geodesic.PyGeodesicAlgorithmExact(pc, faces)
        # e = time.time()
        self.solver = pp3d.MeshHeatMethodDistanceSolver(pc,faces)
        # ee = time.time()

        self.gt_mapping = gt_mapping

        self.dist_arr = []
        self.dist_arr_2 = []
        if dist_arr is None:
            self.full_dist_arr = None
        else:
             self.full_dist_arr = dist_arr


    def build_dist_arr(self):

        if self.gt_mapping is None:
            range_list = list(range(self.N))
        else:
            range_list = self.gt_mapping
        s = time.time()
        # for i, j in enumerate(range_list): 
        #     rel_index = self.mapping[j]
        #     cur_dist, _ = self.geo_func.geodesicDistance(i,rel_index)
        #     e = time.time()
        #     self.dist_arr.append(cur_dist)

        e = time.time()
        for i, j in enumerate(range_list): 
            rel_index = self.mapping[j]
            curr_dist = self.solver.compute_distance(i)[rel_index]
            self.dist_arr.append(curr_dist)

        # ee = time.time()
        # print('method 1:%f' % (e-s))
        # print('method 2:%f' % (ee-e))

        self.dist_arr = np.array(self.dist_arr)
        self.avg_dist = np.mean(self.dist_arr)


    def calc_acc_per_dist(self):

        acc_arr = []
        total = len(self.dist_arr)
        for i in self.distances:
            curr_acc = np.sum(self.dist_arr < i) / total
            acc_arr.append(curr_acc)
        self.acc_arr = acc_arr

    def calc_acc_wrapper(self):
        if len(self.dist_arr) == 0:
            self.build_dist_arr()
        self.calc_acc_per_dist()

        return self.distances, self.acc_arr, self.avg_dist

    def calc_mean_dist(self):

        if len(self.dist_arr) == 0:
            self.build_dist_arr()
        
        return self.avg_dist

    def calc_full_geo_dist_arr(self, fast = False):

        full_arr = []

        for i in list(range(self.N)):
            print(f"Calculating geodisic distance for point {i} of {self.N}") if i%100==0 else None
            if fast:
                curr_dist = self.solver.compute_distance(i)
            else:
                curr_dist, _ = self.geo_func.geodesicDistances([i])

            full_arr.append(curr_dist)
        
        full_arr = np.array(full_arr)
        self.full_dist_arr = full_arr
        return full_arr

    def get_mean_erros_given_p(self, P, geo_dist=None, is_p_mapping=True):

        if not is_p_mapping:
            mapping = P.argmax(0)
        else:
            mapping = P
        
        if mapping.is_cuda:
            mapping = mapping.clone().detach().cpu().numpy()

        dist_arr = []

        if geo_dist is not None:
            self.full_dist_arr= geo_dist


        if self.full_dist_arr is None:
            self.calc_full_geo_dist_arr()

        if self.gt_mapping is None:
            range_list = list(range(self.N))
        else:
            range_list = self.gt_mapping

        for i, j in enumerate(range_list): 
            rel_index = mapping[j]
            curr_err = self.full_dist_arr[i,rel_index]
            dist_arr.append(curr_err)

        mean_dist = torch.mean(torch.Tensor(dist_arr))

        return mean_dist


if __name__ == '__main__':
    MALE_DATA_PATH = '/home/eomri/dfaust_project/data/registrations_m.hdf5'
    FEMALE_DATA_PATH = '/home/eomri/dfaust_project/data/registrations_f.hdf5'


    OUT_P = '/home/eomri/dfaust_project/data/registrations_f_geo_dist.hdf5'



    male_h5 = h5py.File(FEMALE_DATA_PATH, 'r')

    faces = male_h5['faces'][:]


    sub_files = list(male_h5.keys())

    for k in sub_files:
        if k =='faces': continue
        print(k)
        curr_data = male_h5[k]
        all_arrays = []
        OUT_P_tmp = OUT_P.replace('.hdf5', "_" + k + '.hdf5')

        if os.path.exists(OUT_P_tmp):
            print('SKIIPPING %s' % k)
            continue
        out_h5 = h5py.File(OUT_P_tmp, 'w')

        for j in range(0, curr_data.shape[-1], 10):
            curr_geo_arr = GeodisicDist(None, curr_data[:,:,j], faces, is_P_mapping=True).calc_full_geo_dist_arr()
            all_arrays.append(curr_geo_arr.astype('float16'))
        all_arrays = np.array(all_arrays).astype('float16')
        out_h5.create_dataset(k , data=all_arrays, compression="gzip")
        out_h5.close()
        print('%s saved' %k)


    






















