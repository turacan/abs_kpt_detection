import sys
sys.path.append("/workspace/repos")
from twoD_pose_estimator.registerDatasetCatalog import register_data
from twoD_pose_estimator.keypoint_trainer import CustomTrainer
from twoD_pose_estimator.model_config import KEYPOINT_NAMES, KEYPOINT_FLIP_MAP
import json
import sys
import os
import time
from datetime import datetime
import numpy as np

from detectron2.config import get_cfg
from detectron2 import model_zoo

import torch.multiprocessing as tmp


def setup_parameters(cfg):
    """creates a list of all parameter configurations to be used for trainings

    Returns:
        [list]: list of all parameter configurations
    """

    parameters_list = []

    # define all models to be used for training
    model_configs = cfg["models"]["configs"]
    model_weights = cfg["models"]["weights"]

    # define all solver base lr values to be used for training
    solver_base_lr_list = cfg["training_params"]["solver_base_learning_rate"]

    # define all roi head batch sizes to be used for training
    # is a parameter that is used to sample a subset of proposals coming out of RPN to calculate cls and reg loss during training.
    roi_heads_batch_size_per_image_list = cfg["training_params"][
        "roi_heads_batch_size_per_image"
    ]
    anchor_gen_sizes = cfg["training_params"]["anchor_generator_sizes"]
    anchor_gen_ratios = cfg["training_params"]["anchor_generator_aspect_ratio"]

    # create parameter list
    for idx, model in enumerate(model_configs):

        for base_lr in solver_base_lr_list:

            for anchor_size in anchor_gen_sizes:

                for anchor_ratio in anchor_gen_ratios:

                    for heads_batch in roi_heads_batch_size_per_image_list:

                        name = (model.split("/")[1]).split(".")[0]

                        if cfg["training_config"]["with_pretrain"]:
                            name += "_pre_"
                        
                        now = datetime.now()

                        name += "_augs"
                        name += "_" + str(base_lr)
                        name += "_" + now.strftime("%Y-%m-%d_%H-%M-%S")
                        # name += "_" + str(heads_batch)
                        # name += "_" + "_".join([str(a) for a in anchor_size])
                        # name += "_" + "_".join([str(a) for a in anchor_ratio])

                        parameters_list.append(
                            {
                                "model_config": model,
                                # "model_weights": os.path.join(
                                #     cfg["paths"]["model_zoo_path"], model_weights[idx]
                                # ),
                                "model_weights": model_weights[idx],
                                "base_lr": base_lr,
                                "anchor_sizes": anchor_size,
                                "anchor_ratios": anchor_ratio,
                                "heads_batch": heads_batch,
                                "result_path": name,
                            }
                        )

    return parameters_list


def setup_config(config_dict, param_dict) -> dict:
    """creates a config dictionary for a single training

    Args:
        config_dict (dict): config dictionary
        param_dict (dict): parameter dictionary

    Returns:
        detectron2 cfg
    """

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    baseline_model = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml" # 'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x'
    #baseline_model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    #cfg.merge_from_file(model_zoo.get_config_file(baseline_model))
    if config_dict["training_config"]["with_pretrain"]:
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(param_dict["model_config"])
        # model_opts = ["MODEL.WEIGHTS", model_zoo.get_checkpoint_url(
        #     param_dict["model_config"]
        # )]
        cfg.merge_from_file(param_dict["model_config"])
        cfg.MODEL.WEIGHTS = param_dict["model_weights"] 
    else:
        cfg.merge_from_file(model_zoo.get_config_file(baseline_model))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(baseline_model)  # Let training initialize from model zoo
        #cfg.MODEL.WEIGHTS = None
        # model_opts = ["MODEL.WEIGHTS", None]

    # MODEL 
    cfg.MODEL.DEVICE = "cuda" # cuda

    cfg.MODEL.BACKBONE.FREEZE_AT = config_dict["training_config"]["model_backbone_freeze_at"]   # Freeze the backbone up to which stage

    # RPN HEAD
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000

    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
    ##
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = param_dict["anchor_sizes"]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = param_dict["anchor_ratios"]

    # ROI Box HEAD
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = param_dict["heads_batch"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config_dict["training_params"]["num_classes"]
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]  # default: 0.5
    
    # Keypoint specific parameters (see: https://detectron2.readthedocs.io/en/latest/modules/config.html)
    cfg.MODEL.KEYPOINT_ON = config_dict["training_config"]["with_keypoints"]
    # ROI Keypoint Head
    if config_dict["training_config"]["with_keypoints"]:
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = config_dict["keypoint_params"]["num_keypoints"]
        cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 16    # minimum 1 person
        cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = config_dict["keypoint_params"]["roi_kp_head_norm_loss_by_visible_kps"]    # True
        # 1.5 config_dict["keypoint_params"]["roi_kp_heads_loss_weight"]
        cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = config_dict["keypoint_params"]["roi_kp_head_pooler_type"]
        cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14  # standard is 14, tested with 28 no acc increase

    # Segmentation specific parameters
    cfg.MODEL.MASK_ON = config_dict["training_config"]["with_segmentation"]
    # ROI Mask Head
    if config_dict["training_config"]["with_segmentation"]:
        cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14


    # RPN LOSS
    cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
    cfg.MODEL.RPN.LOSS_WEIGHT = 1.0

    # ROI HEADS with adjustable weights
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1" # Options are: "smooth_l1", "giou", "diou", "ciou"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0
    # cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT,  Recommended values:
    #   - use 1.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is True
    #   - use 4.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is False
    if config_dict["training_config"]["with_keypoints"]:
        cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0

    # DATASETS 
    cfg.DATASETS.TRAIN = ("carla/train",)
    cfg.DATASETS.TEST = ("carla/test",)

    keypoint_hflip_indices = np.arange(len(KEYPOINT_NAMES))
    for src, dst in KEYPOINT_FLIP_MAP:
        src_idx = KEYPOINT_NAMES.index(src)
        dst_idx = KEYPOINT_NAMES.index(dst)
        keypoint_hflip_indices[src_idx] = dst_idx
        keypoint_hflip_indices[dst_idx] = src_idx
    cfg.DATASETS.KEYPOINT_HFLIP_INDICES = keypoint_hflip_indices.tolist()
    
    # Regular parameters (see: https://detectron2.readthedocs.io/en/latest/modules/config.html)
    
    # DATALOADER 
    cfg.DATALOADER.NUM_WORKERS = config_dict["training_params"]["dataloader_num_workers"]
    
    # CUDNN
    cfg.CUDNN_BENCHMARK = config_dict["training_config"]["with_cudnn_bm"]
    
    # TEST
    cfg.TEST.DETECTIONS_PER_IMAGE = 30  # 30
    cfg.TEST.EVAL_PERIOD = config_dict["training_params"]["test_eval_period"] # Evaluator is called with the whole Validation dataset
    cfg.TEST.LOSS_VALIDATION_PERIOD = config_dict["training_params"]["test_loss_validation_period"]
    cfg.TEST.KEYPOINT_OKS_SIGMAS = config_dict["training_params"]["test_keypoint_oks_sigmas"]
    cfg.TEST_TASKS = tuple(config_dict["training_params"]["test_tasks"])
    # if baseline_model in cfg.MODEL.WEIGHTS:
    # # if param_dict["model_config"] == "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml":
    #     cfg.SOLVER.IMS_PER_BATCH = 4    # 1
    # else:

    # SOLVER
    cfg.SOLVER.CHECKPOINT_PERIOD = config_dict["training_params"]["solver_checkpoint_period"]   # The model will be saved to .pth format during training after every SOLVER.CHECKPOINT_PERIOD iterations.
    cfg.SOLVER.IMS_PER_BATCH = config_dict["training_params"]["solver_images_per_batch"]
    cfg.SOLVER.WARMUP_ITERS = 1000  # 1000
    cfg.SOLVER.BASE_LR = param_dict["base_lr"]
    cfg.SOLVER.MAX_ITER = config_dict["training_params"]["solver_max_iter"]
    cfg.SOLVER.STEPS = tuple(
        map(
            lambda x: int(x * cfg.SOLVER.MAX_ITER),
            config_dict["training_params"]["solver_steps"],
        )
    )
    
    # INPUT, uses T.ResizeShortestEdge and trys to maintain aspect ratio > NOT desired, because image width should be maintain
    cfg.INPUT.MIN_SIZE_TRAIN = (config_dict["training_params"]["input_image_min_train_size"],)
    cfg.INPUT.MAX_SIZE_TRAIN = config_dict["training_params"]["input_image_max_train_size"]
    cfg.INPUT.MIN_SIZE_TEST = config_dict["training_params"]["input_image_min_test_size"]
    cfg.INPUT.MAX_SIZE_TEST = config_dict["training_params"]["input_image_max_test_size"]
    
    cfg.INPUT.FORMAT = config_dict["training_params"]["input_format"]
    cfg.INPUT.MASK_FORMAT = config_dict["training_params"]["input_mask_format"]  # polygon
    
    # Augmentation specific parameters
    cfg.INPUT.MAX_OBJ_DISTANCE = config_dict["training_params"]["max_obj_distance"]
    cfg.INPUT.USE_AUGMENTATION = config_dict["augmentation_params"]["with_AUG"]
    cfg.INPUT.RANDOM_FLIP = config_dict["augmentation_params"]["random_flip_orientation"]    # choose one of ["horizontal, "vertical", "none"]
    cfg.INPUT.AUG_ZOOM_TRAIN = config_dict["augmentation_params"]["input_aug_zoom"] # height 128=1/8, 256=1/4

    # OUTPUT
    cfg.OUTPUT_DIR = os.path.join(
        config_dict["paths"]["output_path"], param_dict["result_path"]
    )
    cfg.freeze()

    # set config dictionary
    return cfg


def run_training(config_dict, param_dict, train_id, gpu_id):
    """train a detectron2 model with registered dataset and config params

    Args:
        param_dict ([list]): a list of dict containing different training parameter sets to be used
        train_id ([int]): index of current training parameter set
        gpu_id ([int]): id of system gpu to be used for training
    """

    # Console feedback
    print("\tStarting training: ", str(train_id + 1), " on gpu: ", gpu_id)

    # Create result data dir for weights, log and eval storing
    os.makedirs(
        os.path.join(config_dict["paths"]["output_path"], param_dict["result_path"]),
        exist_ok=True,
    )

    # Stop this process from writing train feedback to console, instead write all to log file
            # sys.stdout = open(
            #     os.path.join(
            #         config_dict["paths"]["output_path"],
            #         param_dict["result_path"],
            #         "train_log.out",
            #     ),
            #     "w+",
            # )
            # sys.stderr = sys.stdout

    # Set system gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Setup detectron2 training config
    cfg = setup_config(config_dict, param_dict)

    # dump the config in the train folder
    with open(os.path.join(cfg.OUTPUT_DIR, "model_config.yaml"), "w+") as file:
        file.write(cfg.dump())

    # Init trainer and start training
    trainer = CustomTrainer(cfg) 
    trainer.resume_or_load(resume=config_dict["training_config"]["with_resume"])
    trainer.train()

    # trainer = KeypointTrainer(cfg)
    # trainer.resume_or_load(resume=config_dict["training_config"]["with_resume"])
    # trainer.train()

    return


def start_trainings(cfg: dict):
    # register Dataset- and Metadatactalogs fpr train and teþst datasets
    register_data(input_path= "/workspace/data/dataset")

    # create parameter sweeping list
    params_list = setup_parameters(cfg=cfg)
    # Init control variables for multi gpu usage for multiple training in parallel
    gpu_ids = cfg["training_config"]["gpu_ids"]
    n_gpus = len(gpu_ids)
    gpu_cnt = 0
    param_cnt = 0
    n_params = len(params_list)
    training_processes = []

    # setup training
    while param_cnt < n_params:

        # create as much trainings in parallel as different gpu´s are available
        while (gpu_cnt < n_gpus) and (param_cnt < n_params):

            print("Param_set: ", str(param_cnt + 1), " of ", str(n_params))

            # create a single training with defined parameters
            p = tmp.Process(
                target=run_training,
                args=(cfg, params_list[param_cnt], param_cnt, gpu_ids[gpu_cnt]),
            )
            p.start()
            training_processes.append(p)
            gpu_cnt += 1
            param_cnt += 1
            time.sleep(10)

        # wait until a free gpu is available again
        if gpu_cnt == n_gpus:

            for p in training_processes:
                p.join()
            training_processes.clear()
            gpu_cnt = 0

    # if there is still one training process left, wait for it to finish
    if training_processes:

        for p in training_processes:
            p.join()
        training_processes.clear()

    return

if __name__ == "__main__":
    try:
        with open("twoD_pose_estimator/config/config.json", 'r') as f:
            cfg = json.load(f)
    except Exception as ex:
        sys.exit("provided cfg file path not valid")
    start_trainings(cfg=cfg)