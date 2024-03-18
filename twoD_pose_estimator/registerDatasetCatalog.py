from detectron2.data import DatasetCatalog, MetadataCatalog
import os
import glob
import json
import numpy as np

#from custom_mapper import custom_mapper, get_dataset_dicts
import sys
sys.path.append("/workspace/repos/")
from twoD_pose_estimator.model_config import KEYPOINT_NAMES, KEYPOINT_CONNECTION_RULES, KEYPOINT_FLIP_MAP


def get_dataset_dicts(path: str):
    files = glob.glob(path+"/*")
    dataset_dicts = []
    for json_file in files:
        with open(json_file) as f:
            record = json.load(f)
        dataset_dicts.append(record)
    
    return dataset_dicts


def register_data(input_path: str):
    global KEYPOINT_NAMES, KEYPOINT_CONNECTION_RULES, KEYPOINT_FLIP_MAP
    #input_path = "/workspace/data/dataset"
    
    for d in ["train", "test"]:
        DatasetCatalog.register("carla/" + d, lambda d=d: get_dataset_dicts(os.path.join(input_path, d, "labels")))
        MetadataCatalog.get("carla/" + d).set(thing_classes=["person"])
        MetadataCatalog.get("carla/" + d).set(keypoint_names = KEYPOINT_NAMES)
        MetadataCatalog.get("carla/" + d).set(keypoint_connection_rules = KEYPOINT_CONNECTION_RULES)
        MetadataCatalog.get("carla/" + d).set(keypoint_flip_map = KEYPOINT_FLIP_MAP)
    
    return

if __name__ == '__main__':
    import copy
    import cv2
    from detectron2.utils.visualizer import Visualizer
    from detectron2.utils.visualizer import ColorMode

    # try:
    #     with open("config/config.json", 'r') as f:
    #         cfg = json.load(f)
    # except Exception as ex:
    #     sys.exit("provided cfg file path not valid")
    
    # register_data(input_path= "/workspace/data/dataset")
    # params_list = setup_parameters(cfg=cfg)

    # keypoint_hflip_indices = np.arange(len(KEYPOINT_NAMES))
    # for src, dst in KEYPOINT_FLIP_MAP:
    #     src_idx = KEYPOINT_NAMES.index(src)
    #     dst_idx = KEYPOINT_NAMES.index(dst)
    #     keypoint_hflip_indices[src_idx] = dst_idx
    #     keypoint_hflip_indices[dst_idx] = src_idx

    # dataset_dicts_ = DatasetCatalog.get("carla/train")
    # import random
    # for d in random.sample(dataset_dicts_, 5):
    #     d_ = copy.deepcopy(d)
    #     d_ = custom_mapper(d_, cfg, keypoint_hflip_indices=keypoint_hflip_indices, is_train = True)
    #     img = np.uint8(255*d_["image"].numpy().transpose(1, 2, 0))
    #     img = cv2.applyColorMap(img[...,0],cv2.COLORMAP_BONE)
    #     #cv2.imshow("img",img)
    #     v = Visualizer(img, # draw_circle
    #                scale=0.8, 
    #                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    #     )   
    #     vis = v.draw_dataset_dict(d_)
    #     cv2.imshow('test', vis.get_image())
    #     cv2.waitKey(0)
        
    #     print("d")   
    # cv2.destroyAllWindows() 
    # #trainer = CustomTrainer(cfg) 
    # print()    
