from mmcv import Config
from mmdet.apis import set_random_seed
import mmcv
import os
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import numpy as np
import os.path as osp

pretrained_models = {
    "nopretrain": None,
    "coco": "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth",
    "twincity3": "checkpoints/twincity_3classes.pth"
}


def benchmark_finetuning(exp_folder, ade_size, pretrained_model=None, seed=0, max_epochs = 12, use_tensorboard=True):

        #%% cfg base
        cfg = Config.fromfile('configs/faster_rcnn_r50_fpn_1x_cocotwincityade20kmerged.py') # Here val is ADE20k

        #%% Data
        cfg_data_ade20k = Config.fromfile("../synthetic_cv_data_benchmark/datasets/ade20k.py")
        cfg_data_ade20k.data.train.ann_file = f'../../datasets/ADE20K_2021_17_01/coco-training_{ade_size}.json'

        # Concatenate Datasets or not
        datasets = [build_dataset([cfg_data_ade20k.data.train])]

        #%% Model
        cfg.model.roi_head.bbox_head.num_classes = 3
        cfg.load_from = pretrained_models[pretrained_model]
        model = build_detector(cfg.model)

        #%% Runner
        cfg.runner.max_epochs = max_epochs
        cfg.evaluation.interval = 5
        cfg.log_config.interval = 10
        cfg.checkpoint_config.interval = max_epochs
        cfg.seed = seed
        set_random_seed(seed, deterministic=False)

        #%% CUDA
        cfg.data.workers_per_gpu = 0
        cfg.gpu_ids = range(1)
        cfg.device = 'cuda'

        # %% Logs, working dir to save files and logs.
        if use_tensorboard:
            cfg.log_config.hooks = [
                dict(type='TextLoggerHook'),
                dict(type='TensorboardLoggerHook')]
        else:
            cfg.log_config.hooks = [
                dict(type='TextLoggerHook')]

        cfg.work_dir = f'{exp_folder}/ade{ade_size}_pretrain{pretrained_model}'
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

        #%% Dump config file
        cfg.dump(osp.join(cfg.work_dir, "cfg.py"))

        #%%
        print(cfg.data.train[0].classes)
        print(cfg.data.train[1].classes)
        print(cfg.data.val.classes)

        #%% Launch
        train_detector(model, datasets, cfg, distributed=False, validate=True)


if __name__ == '__main__':

    max_epochs = 20
    exp_folder = "exps/benchmark_finetuning"
    myseed = 0
    ade_size = 128
    pretrained_model = "twincity3"

    benchmark_finetuning(exp_folder, ade_size, pretrained_model , myseed, max_epochs=max_epochs)

    classes = ('Window', 'Person', 'Vehicle')
    workdir = f'{exp_folder}/ade{ade_size}_pretrain{pretrained_model}'

    # img, result = inspect_results(workdir, classes, 20)




    """
    # With 3 classes
    for pretrained_model in pre_trained_models.keys():
        for ade_size in [64, 128, 256, 512]:
            benchmark_finetuning(exp_folder, ade_size, pretrained_model, myseed, max_epochs=max_epochs)
    """


