from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from os import path as osp
import json


def plot_results_ade20k(idx_img, model):
    ade20k_folder = "../../datasets/ADE20K_2021_17_01/"
    ade20k_coco_json_path = osp.join(ade20k_folder, "coco-training.json")
    with open(ade20k_coco_json_path) as jsonFile:
        ade20k_coco_json = json.load(jsonFile)

    # Image ade
    ade20k_example = osp.join("../../datasets/ADE20K_2021_17_01/images/ADE/training/urban/street",
                                ade20k_coco_json["images"][idx_img]["file_name"].replace("\\", "/"))
    img_ade20k = mmcv.imread(ade20k_example)
    result_ade20k = inference_detector(model, img_ade20k)

    show_result_pyplot(model, img_ade20k, result_ade20k)

    return img_ade20k, result_ade20k

def plot_results(idx_img, model):

    twincity_folder = "../../datasets/twincity-dataset/"
    coco_json_path = osp.join(twincity_folder, "coco-train.json")
    with open(coco_json_path) as jsonFile:
        twincity_coco_json = json.load(jsonFile)

    ade20k_folder = "../../datasets/ADE20K_2021_17_01/"
    ade20k_coco_json_path = osp.join(ade20k_folder, "coco-training.json")
    with open(ade20k_coco_json_path) as jsonFile:
        ade20k_coco_json = json.load(jsonFile)

    # cfg_dict = {"model": {"roi_head": {"bbox_head": {"num_classes": 3}}}}

    # Specify the path to model config and checkpoint file
    #config_file = 'configs/faster_rcnn_r50_fpn_1x_cocotwincityade20kmerged.py'
    #checkpoint_file = 'exps/exp_bench/twincity+ade-64/latest.pth'
    # build the model from a config file and a checkpoint file
    #model = init_detector(config_file, checkpoint_file, cfg_options=cfg_dict)  # , device='cuda:0')

    # Image twincity
    twincity_example = osp.join(twincity_folder, twincity_coco_json["images"][idx_img]["file_name"].replace("\\", "/"))
    img_twincity = mmcv.imread(twincity_example)
    result_twincity = inference_detector(model, img_twincity)


    # Image twincity
    ade20k_example = osp.join("../../datasets/ADE20K_2021_17_01/images/ADE/training/urban/street",
                                ade20k_coco_json["images"][idx_img]["file_name"].replace("\\", "/"))
    img_ade20k = mmcv.imread(ade20k_example)
    result_ade20k = inference_detector(model, img_ade20k)


    # model.cfg = cfg
    show_result_pyplot(model, img_ade20k, result_ade20k)
    show_result_pyplot(model, img_twincity, result_twincity)

    return 1