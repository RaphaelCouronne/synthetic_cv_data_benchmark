from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import numpy as np




def benchmark(add_twincity, ade_size, i, exp_folder, seed=0, classes=None, max_epochs = 12):

        #%% cfg base
        cfg = Config.fromfile('configs/faster_rcnn_r50_fpn_1x_cocotwincityade20kmerged.py') # Here val is ADE20k

        #%% Data
        cfg_data_twincity = Config.fromfile("../synthetic_cv_data_benchmark/datasets/twincity.py")
        cfg_data_ade20k = Config.fromfile("../synthetic_cv_data_benchmark/datasets/ade20k.py")
        cfg_data_ade20k.data.train.ann_file = f'../../datasets/ADE20K_2021_17_01/coco-training_{ade_size}.json'

        # Classes
        if classes is not None:
            cfg_data_ade20k.data.train.classes = classes
            cfg_data_ade20k.data.val.classes = classes
            cfg_data_twincity.data.train.classes = classes

        # Concatenate Datasets or not
        if add_twincity:
            datasets = [build_dataset([cfg_data_ade20k.data.train, cfg_data_twincity.data.train])]
        else:
            datasets = [build_dataset([cfg_data_ade20k.data.train])]

        #%% Model
        cfg.model.roi_head.bbox_head.num_classes = 3
        if classes is not None:
            cfg.model.roi_head.bbox_head.num_classes = len(classes)
        model = build_detector(cfg.model)

        #%% Runner
        cfg.runner.max_epochs = max_epochs
        cfg.evaluation.interval = max_epochs
        cfg.checkpoint_config.interval = max_epochs
        cfg.seed = seed
        set_random_seed(seed, deterministic=False)

        #%% CUDA
        cfg.data.workers_per_gpu = 0
        cfg.gpu_ids = range(1)
        cfg.device = 'cuda'

        # %% Logs, working dir to save files and logs.
        cfg.log_config.hooks = [
            dict(type='TextLoggerHook')]
        cfg.work_dir = f'{exp_folder}/ade+TC-{add_twincity}_{ade_size}_{i}'
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

        #%% Launch
        train_detector(model, datasets, cfg, distributed=False, validate=True)


if __name__ == '__main__':

    exp_folder = "exps/exp_bench-v3"
    myseed = np.random.randint(10000)
    classes = ('Person', 'Vehicle') # ('Window', 'Person', 'Vehicle')

    for i in range(1):
        for add_twincity in [False]:
            for ade_size in [128]: #[16, 64, 128, 256, 512]:
                benchmark(add_twincity, ade_size, i, exp_folder, myseed, classes)