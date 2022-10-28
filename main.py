
from benchmark import benchmark
from benchmark_finetuning import benchmark_finetuning



if __name__ == '__main__':



    """
    Test launch
    """

    pre_train = False
    add_twincity = False
    i = 1
    exp_folder = "exps/launch_test"
    myseed = 0
    classes = ('Window', 'Person', 'Vehicle')
    max_epochs = 10

    """
    ade_size = 64
    benchmark(pre_train, True, i, exp_folder, ade_size, myseed, classes,
              max_epochs=max_epochs,
              log_config_interval=int(ade_size / 64), evaluation_interval=int(max_epochs / 5))
    """

    ade_size = 128
    benchmark(pre_train, add_twincity, i, exp_folder, ade_size, myseed, classes,
              max_epochs=max_epochs,
              log_config_interval=int(ade_size / 64), evaluation_interval=1)

    ade_size = 128
    benchmark(pre_train, add_twincity, i, exp_folder, ade_size, myseed, classes,
              max_epochs=max_epochs,
              log_config_interval=int(ade_size / 64), evaluation_interval=int(max_epochs / 5))

    """
    First Benchmark : What does adding Twin City do to performances on ADE20K of varying size ?
    
    Note that we handle both cases of having seen and unseen classes
    """



    exp_folder = "exps/firstbenchmark-minimal-adeOK"
    myseed = 0

    for i in range(1):
        for pre_train in [False]:
            for classes in [('Window', 'Person', 'Vehicle')]:
                for add_twincity in [True, False]:
                    for ade_size in [128, 512, 2054]:
                        max_epochs = int(20)
                        benchmark(pre_train, add_twincity, ade_size, i, exp_folder, myseed, classes,
                                  max_epochs=max_epochs,
                                  log_config_interval=int(ade_size / 64), evaluation_interval=int(max_epochs / 5))


    """

    for i in range(1):
        for pre_train in [True, False]:
            for classes in [('Window', 'Person', 'Vehicle'), ('Person', 'Vehicle')]:
                for add_twincity in [False, True]:
                    for ade_size in [64, 128, 256, 1024]:

                        if not add_twincity:
                            max_epochs = int(20*(512/ade_size))
                        else:
                            max_epochs = int(20)

                        benchmark(pre_train, add_twincity, ade_size, i, exp_folder, myseed, classes, max_epochs=max_epochs)
    """


    """
    Second Benchmark : Compare pre-training methods
    
    i.e. is twin city a valid pre-trainig method ? When we have all classes at pretraining ? When we don't (eg windows) ?
    
    Later : study for 1 class specific also (eg windows) %TODO TO THINK ABOUT
    """



    max_epochs = 20
    exp_folder = "exps/benchmark_finetuning-finalv3"
    myseed = 0

    pretrained_models = {
        # "nopretrain": None,
        "coco": "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth",
        "twincity3": "checkpoints/twincity_3classes.pth",
        # "twincity1": "checkpoints/twincity_1class.pth",
    }

    """
    ade_size = 512
    pretrained_model_name = "coco"
    pretrained_model_path = "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"

    """

    # With 3 classes
    for (pretrained_model_name, pretrained_model_path) in pretrained_models.items():
        print(f"=== {pretrained_model_name} ===")
        for ade_size in [64, 256, 1024]:
            print(f"=== {ade_size} ===")
            max_epochs = int(20 * (512 / ade_size))
            # benchmark_finetuning(exp_folder, ade_size, pretrained_model_name, pretrained_model_path, myseed,
             #                    max_epochs=max_epochs,
             #                    log_config_interval=int(ade_size / 64), evaluation_interval=int(max_epochs / 5))

    #%% Check ADE
    import os.path as osp
    import json
    ade20k_folder = "../../datasets/ADE20K_2021_17_01/"
    ade20k_coco_json_path = osp.join(ade20k_folder, "coco-training_512.json")
    with open(ade20k_coco_json_path) as jsonFile:
        ade20k_coco_json = json.load(jsonFile)
    print(len(ade20k_coco_json["images"]))
