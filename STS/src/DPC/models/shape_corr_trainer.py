"""Shape correspondence template."""
from argparse import Namespace
import collections
import os

from torch.utils.data import DataLoader
import torch
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
import numpy as np
import h5py

from ..utils import switch_functions
from .correspondence_utils import square_distance
from ..utils.tensor_utils import to_numpy
from .metrics.metrics import AccuracyAssumeEye


class ShapeCorrTemplate(LightningModule):

    def __init__(self, hparams, **kwargs):
        super(ShapeCorrTemplate, self).__init__()
        load_hparams = vars(hparams) if isinstance(hparams, Namespace) else hparams
        for k,v in load_hparams.items():
            setattr(self.hparams,k,v)
        self.save_hyperparameters()
        self.config = hparams
        self.train_accuracy = AccuracyAssumeEye()
        self.val_accuracy = AccuracyAssumeEye()
        self.test_accuracy = AccuracyAssumeEye()
        self.losses = {}
        self.tracks = {}
 
    def setup(self, stage):
        self.hparams.consistant_shape_indices_filename = os.path.join(self.config.log_to_dir, "consistant_shape_indices.npz")
        self.train_dataset = switch_functions.load_dataset_spectral(self.hparams) 
        self.test_dataset = self.train_dataset#switch_functions.load_dataset_spectral(self.hparams) 
        self.store_data_indices()
    
    def store_data_indices(self):  
        if not os.path.isfile(self.hparams.consistant_shape_indices_filename):
            np.savez(
                self.hparams.consistant_shape_indices_filename,
                template_indices=self.train_dataset.template_indices,
                unlabeled_indices=self.train_dataset.unlabeled_indices
                )

    def training_step(self, batch, batch_idx, mode="train"):
        self.batch = self.covnert_batch_from_spectral_to_DPC(batch) # STS
        
        self.losses = {}
        self.tracks = {}
        self.hparams.batch_idx = batch_idx
        self.hparams.mode = mode
            
        batch = self(batch)
        
        # self.log_weights_norm()
        
        if len(self.losses) > 0:
            loss = sum(self.losses.values()).mean() #TODO scale lambdas
            self.tracks[f"{mode}_tot_loss"] = loss
        else:
            loss = None

        all = {k: to_numpy(v) for k, v in {**self.tracks, **self.losses}.items()}
        getattr(self, f"{mode}_logs", None).append(all)

        # TODO:
        # if (batch_idx % (self.hparams.log_every_n_steps if self.hparams.mode != 'test' else 1) == 0):
        #     for k, v in all.items():
        #         self.logger.experiment.add_scalar(f"{k}/step", v,self.global_step)

        if self.vis_iter():
            self.visualize(batch, mode=mode) #TODO go over it and compare to old code. remove p matrices!

        output = collections.OrderedDict({"loss": loss})
        return output

        

    def vis_iter(self):
        return ((self.current_epoch % self.hparams.train_vis_interval) == 0) and self.hparams.show_vis

    def configure_optimizers(self):
        self.optimizer = switch_functions.choose_optimizer(self.hparams, self.parameters())
        # self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: (1 - epoch / self.hparams.max_epochs))
        return self.optimizer  # , [self.optimizer],[self.scheduler]

    def dataloader(self, dataset):
        loader = DataLoader(
            dataset=dataset,
            batch_size=getattr(self.hparams, "train_batch_size", 1),
            num_workers=self.hparams.num_data_workers,
            shuffle=False,
            drop_last=True,
            sampler=None,
        )
        return loader

    def covnert_batch_from_spectral_to_DPC(self, batch):
        batch['source'] = {'pos':batch['vertices'][:, :, :, 0].float()}
        batch['target'] = {'pos':batch['vertices'][:, :, :, 1].float()}
        return batch

    def train_dataloader(self):
        return self.dataloader(self.train_dataset)

    def on_train_epoch_start(self):
        self.train_logs = []
        self.hparams.current_epoch = self.current_epoch

    def on_train_epoch_end(self):
        logs = getattr(self, f"{self.hparams.mode}_logs", None)
        dict_of_lists = {k: [dic[k] for dic in logs] for k in logs[0]}
        for key, lst in dict_of_lists.items():
            s = 0
            for item in lst:
                s += item.sum()
            name = f"{self.hparams.mode}/{key}/epoch"
            val = s / len(lst)
            self.tracks[name] = val

            # self.logger.experiment.add_scalar(name, val, self.current_epoch)  # Old version command
            self.log_dict({name: val}) #, on_epoch=False) 

        return dict_of_lists

    @staticmethod
    def compute_acc(label, ratio_list, soft_labels, p, input2, track_dict={}, hparams=Namespace()):
        corr_tensor = ShapeCorrTemplate._prob_to_corr_test(p)

        hit = label.argmax(-1).squeeze(0)
        pred_hit = p.squeeze(0).argmax(-1)
        target_dist = square_distance(input2.squeeze(0), input2.squeeze(0)) 
        track_dict["acc_mean_dist"] = target_dist[pred_hit,hit].mean().item()
        if(getattr(hparams,'dataset_name','') == 'tosca' or (hparams.mode == 'test' and hparams.test_on_tosca)):
            track_dict["acc_mean_dist"] /= 3 # TOSCA is not scaled to meters as the other datasets. /3 scales the shapes to be coherent with SMAL (animals as well)


        acc_000 = ShapeCorrTemplate._label_ACC_percentage_for_inference(corr_tensor, label.unsqueeze(0))
        track_dict["acc_0.00"] = acc_000
        for idx,ratio in enumerate(ratio_list):
            track_dict["acc_" + str(ratio)] = ShapeCorrTemplate._label_ACC_percentage_for_inference(corr_tensor, soft_labels[f"{ratio}"].unsqueeze(0)).item()
        return track_dict

    @staticmethod
    def _label_ACC_percentage_for_inference(label_in, label_gt):
        assert (label_in.shape == label_gt.shape)
        bsize = label_in.shape[0]
        b_acc = []
        for i in range(bsize):
            element_product = torch.mul(label_in[i], label_gt[i])
            N1 = label_in[i].shape[0]
            sum_row = torch.sum(element_product, dim=-1)  # N1x1

            hit = (sum_row != 0).sum()
            acc = hit.float() / torch.tensor(N1).float()
            b_acc.append(acc * 100.0)
        mean = torch.mean(torch.stack(b_acc))
        return mean

    @staticmethod
    def _prob_to_corr_test(prob_matrix):
        c = torch.zeros_like(prob_matrix)
        idx = torch.argmax(prob_matrix, dim=2, keepdim=True)
        for bsize in range(c.shape[0]):
            for each_row in range(c.shape[1]):
                c[bsize][each_row][idx[bsize][each_row]] = 1.0

        return c

    def save_inference(self, pred): #TODO add saving of downsampling indices
        self.hparams.output_inference_dir = self.hparams.output_inference_dir
        os.makedirs(self.hparams.output_inference_dir, exist_ok=True)

        with open(os.path.join(self.hparams.output_inference_dir, "orig_model_info.txt"), "w") as f:
            f.write(f"orig checkpoints: \n{self.hparams.resume_from_checkpoint}")
        p = pred["P_normalized"].clone()
        f =  h5py.File(os.path.join(self.hparams.output_inference_dir, "model_inference.hdf5"), 'a')

        p_file_name = f"p_{pred['id'][0]}"
        f.create_dataset(
                    name=p_file_name, 
                    data=p[0].cpu().numpy(), 
                    compression="gzip")
        
        source_neigh_idxs_file_name = f"source_neigh_idxs_{pred['id'][0]}"
        f.create_dataset(
                    name=source_neigh_idxs_file_name, 
                    data=pred["source"]["neigh_idxs"][0].cpu().numpy(), 
                    compression="gzip")

        target_neigh_idxs_file_name = f"target_neigh_idxs_{pred['id'][0]}"
        f.create_dataset(
                    name=target_neigh_idxs_file_name, 
                    data=pred["target"]["neigh_idxs"][0].cpu().numpy(), 
                    compression="gzip")

        source_file_name = f"source_{pred['id'][0]}"
        f.create_dataset(
                    name=source_file_name, 
                    data=pred["source"]["pos"][0].cpu().numpy(), 
                    compression="gzip")

        target_file_name = f"target_{pred['id'][0]}"
        f.create_dataset(
                    name=target_file_name, 
                    data=pred["target"]["pos"][0].cpu().numpy(), 
                    compression="gzip")

        # print(f"Saved inference to {os.path.join(self.output_inference_dir, batch['id'][0])}...")
        f.close()