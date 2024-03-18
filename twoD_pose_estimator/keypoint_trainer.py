import os
import numpy as np
import cv2
from tqdm import tqdm
import copy

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

import sys
sys.path.append("/workspace/repos")
from twoD_pose_estimator.custom_mapper import custom_mapper, evaluator_mapper, RandomHorizonZoom, Resize
from twoD_pose_estimator.eval.custom_coco_eval import COCOCustomEvaluator

#from old_train_files.hooks import LossEvalHook
from detectron2.data import (
    DatasetMapper,
    DatasetCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from torch.utils.data import Dataset
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T


# class FilteredDataset(Dataset):
#     def __init__(self, cfg, original_dataset):
#         self.cfg = cfg
#         self.filtered_dataset = []
#         self.total_filtered_instances = 0
#         print("Filtering dataset...")
#         for item in tqdm(original_dataset):
#             filtered_item = self.filter_instances(item)
#             if filtered_item is not None:
#                 self.filtered_dataset.append(filtered_item)

#     def filter_instances(self, item):
#         hha_img = cv2.imread(item["file_name"], cv2.IMREAD_UNCHANGED)
#         transform_list_shape = [
#             RandomHorizonZoom(self.cfg.INPUT.AUG_ZOOM_TRAIN),
#             Resize(short_edge_length=[self.cfg.INPUT.AUG_MIN_SIZE_TRAIN[0]],long_edge_length=[self.cfg.INPUT.AUG_MAX_SIZE_TRAIN[0]], sample_style="choice")
#         ]
#         image, transforms = T.apply_transform_gens(transform_list_shape, hha_img)

#         threshold_bbox_size = image.shape[0] * image.shape[1] * 0.0001 
#         annos_valid_idx = []
#         annotations_list = copy.deepcopy(item['annotations'])
#         for item_idx, annotation in enumerate(annotations_list):
#             if annotation.get("iscrowd", 0) == 0:
#                 # in-place operation (utils.transform_instance_annotations)
#                 transformed_annotation = utils.transform_instance_annotations(copy.deepcopy(annotation), [transforms], image.shape[:2], keypoint_hflip_indices=self.cfg.DATASETS.KEYPOINT_HFLIP_INDICES)
#                 bbox = transformed_annotation['bbox']
#                 area_box = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
#                 if area_box >= threshold_bbox_size:
#                     annos_valid_idx.append(item_idx)
#                 else:
#                     self.total_filtered_instances +=1
                
#         # If no valid annotations remain for this item, return None
#         if not annos_valid_idx:
#             return None
#         annos = [elem for i, elem in enumerate(item['annotations']) if i in annos_valid_idx]
#         # Replace the original item's annotations with the filtered annotations
#         item['annotations'] = annos
#         return item

#     def __getitem__(self, idx):
#         return self.filtered_dataset[idx]

#     def __len__(self):
#         return len(self.filtered_dataset)

'''
CustomTrainer instance .start()
train (/workspace/detectron2/detectron2/engine/defaults.py:484)
super(): train (/workspace/detectron2/detectron2/engine/train_loop.py:150)
after_step (/workspace/detectron2/detectron2/engine/train_loop.py:180)
after_step (/workspace/detectron2/detectron2/engine/hooks.py:552)
_do_eval (/workspace/detectron2/detectron2/engine/hooks.py:525)
test_and_save_results (/workspace/detectron2/detectron2/engine/defaults.py:453)
test (/workspace/detectron2/detectron2/engine/defaults.py:608)
inference_on_dataset (/workspace/detectron2/detectron2/evaluation/evaluator.py:165)

COCOCustomEvaluator instance evaluator:
evaluator.reset()
evaluator.process()
evaluator.evaluate()
'''


class CustomTrainer(DefaultTrainer):   
    """
    This class overloads the DefaultTrainer so that after each args.eval_period the evaluation with the "COCOEvaluator"
    is triggered.
    """
    @classmethod
    def build_train_loader(cls, cfg):
        #mapper = CustomDatasetMapper(cfg, is_train=True)
        original_dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        #filtered_dataset = FilteredDataset(cfg, original_dataset)           # len(original_dataset[0]['annotations'])
        #print(f"num of total_filtered_instances = {filtered_dataset.total_filtered_instances}")
        return build_detection_train_loader(cfg, dataset=original_dataset, mapper=lambda d: custom_mapper(d, cfg))
        #return custom_build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        #return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))
        return build_detection_test_loader(
            cfg, dataset_name, mapper=lambda d: custom_mapper(d, cfg, is_train=False)
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, tasks=cfg.TEST_TASKS, distributed=True, output_dir=output_folder, 
                    cfg=cfg, kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS)


    
if __name__ == "__main__":
    '''
    Visualization of DatasetCatalog in uint8, including bbox, segmentation
    '''
    import copy
    import numpy as np
    import cv2
    import json
    import sys

    from detectron2.utils.visualizer import Visualizer
    from detectron2.utils.visualizer import ColorMode

    from registerDatasetCatalog import register_data
    register_data(input_path= "/workspace/data/dataset")

    from detectron2.data import DatasetCatalog, MetadataCatalog
    splittype = "train"
    dataset_dicts_ = DatasetCatalog.get(f"carla/{splittype}")
    meta_data = MetadataCatalog.get(f"carla/{splittype}")

    try:
        with open("twoD_pose_estimator/config/config.json", 'r') as f:
            cfg = json.load(f)
    except Exception as ex:
        sys.exit("provided cfg file path not valid")

    from twoD_pose_estimator.start_training import setup_parameters, setup_config
    # create parameter sweeping list
    params_list = setup_parameters(cfg=cfg)
    # Setup detectron2 training config
    cfg = setup_config(cfg, params_list[0])

    import random
    #from custom_mapper import custom_test_dataset_transform
    for d in tqdm(dataset_dicts_[24:]):#random.sample(dataset_dicts_, 5):  # 109, 67
        d_ = copy.deepcopy(d)
        #dset = custom_test_dataset_transform(d_, cfg)
        d_ = custom_mapper(d_, cfg, is_train=True)
        # dset['image'] = d_ ['image']
        # d_ = dset
        img = np.uint8(255*d_["image"].numpy().transpose(1, 2, 0))
        img = cv2.applyColorMap(img[...,0],cv2.COLORMAP_BONE)
        #cv2.imshow("img",img)
        v = Visualizer(img, # draw_circle
            scale=0.8, 
            metadata = meta_data,
            instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
        vis = v.draw_dataset_dict(d_)
        cv2.imshow('test', vis.get_image())
        cv2.waitKey(0)