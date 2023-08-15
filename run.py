from easydict import EasyDict

from STS import train_sts_dpc, infer_sts_dpc, get_default_config

config = EasyDict(get_default_config())

config.max_epochs=5000
config.log_every_n_steps=10
config.save_checkpoints_every_n_epochs=10
config.template_h5_path =  "/home/shahar/cardio_corr/outputs/synthetic_dataset37/28/orig/h5_datasets/point_cloud_from_mesh_smooth_dataset.hdf5"
config.unlabeled_h5_path = "/home/shahar/cardio_corr/outputs/synthetic_dataset37/18/orig/h5_datasets/point_cloud_from_mesh_smooth_dataset.hdf5"
config.gt_flow_path =      "/home/shahar/cardio_corr/outputs/synthetic_dataset37/thetas_90.0_0.0_rs_0.9_0.9_h_0.9_random_3_0.4_mask_True_blur_radious_7/flow_for_mask_thetas_90.0_0.0_rs_0.9_0.9_h_0.9_random_3_0.4_mask_True_blur_radious_7.npy"
config.log_to_dir = "/home/shahar/cardio_corr/outputs/sts_training"
config.gpus = [0]
config.num_points = 10000
config.train_vis_interval = 20

best_model_path, config = train_sts_dpc(config)

infer_output_path = infer_sts_dpc(config, best_model_path)

print(1)