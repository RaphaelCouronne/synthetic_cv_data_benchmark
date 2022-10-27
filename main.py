
from benchmark import benchmark
from benchmark_finetuning import benchmark_finetuning



if __name__ == '__main__':


    """
    First Benchmark : What does adding Twin City do to performances on ADE20K of varying size ?
    
    Note that we handle both cases of having seen and unseen classes
    """

    exp_folder = "exps/firstbenchmark"
    myseed = 0
    classes = ('Window', 'Person', 'Vehicle')

    for i in range(1):
        for pre_train in [False, True]:
            for classes in [('Window', 'Person', 'Vehicle'), ('Person', 'Vehicle')]:
                for add_twincity in [False, True]:
                    for ade_size in [64, 128, 256]:

                        if not add_twincity:
                            max_epochs = int(20*(512/ade_size))
                        else:
                            max_epochs = int(20)

                        benchmark(pre_train, add_twincity, ade_size, i, exp_folder, myseed, classes, max_epochs=max_epochs)



    """
    Second Benchmark : Compare pre-training methods
    
    i.e. is twin city a valid pre-trainig method ? When we have all classes at pretraining ? When we don't (eg windows) ?
    
    Later : study for 1 class specific also (eg windows) %TODO TO THINK ABOUT
    """

    max_epochs = 20
    exp_folder = "exps/benchmark_finetuning"
    myseed = 0

    pretrained_models = {
        "nopretrain": None,
        "coco": "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth",
        "twincity3": "checkpoints/twincity_3classes.pth",
        "twincity1": "checkpoints/twincity_1class.pth",
    }

    # With 3 classes
    for pretrained_model in pretrained_models.keys():
        for ade_size in [64, 128, 256, 512]:
            benchmark_finetuning(exp_folder, ade_size, pretrained_model, myseed, max_epochs=max_epochs)
