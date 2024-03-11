import os
import numpy as np
import cv2
from tqdm import tqdm
import copy

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from custom_mapper import custom_mapper
from eval.custom_coco_eval import COCOCustomEvaluator

from old_train_files.hooks import LossEvalHook
from detectron2.data import (
    DatasetMapper,
    DatasetCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from torch.utils.data import Dataset
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T

from custom_mapper import RandomHorizonZoom, Resize


class FilteredDataset(Dataset):
    def __init__(self, cfg, original_dataset):
        self.cfg = cfg
        self.filtered_dataset = []
        self.total_filtered_instances = 0
        print("Filtering dataset...")
        for item in tqdm(original_dataset):
            filtered_item = self.filter_instances(item)
            if filtered_item is not None:
                self.filtered_dataset.append(filtered_item)

    def filter_instances(self, item):
        hha_img = cv2.imread(item["file_name"], cv2.IMREAD_UNCHANGED)
        transform_list_shape = [
            RandomHorizonZoom(self.cfg.INPUT.AUG_ZOOM_TRAIN),
            Resize(short_edge_length=[self.cfg.INPUT.AUG_MIN_SIZE_TRAIN[0]],long_edge_length=[self.cfg.INPUT.AUG_MAX_SIZE_TRAIN[0]], sample_style="choice")
        ]
        image, transforms = T.apply_transform_gens(transform_list_shape, hha_img)

        threshold_bbox_size = image.shape[0] * image.shape[1] * 0.0001 
        annos_valid_idx = []
        annotations_list = copy.deepcopy(item['annotations'])
        for item_idx, annotation in enumerate(annotations_list):
            if annotation.get("iscrowd", 0) == 0:
                # in-place operation (utils.transform_instance_annotations)
                transformed_annotation = utils.transform_instance_annotations(copy.deepcopy(annotation), [transforms], image.shape[:2], keypoint_hflip_indices=self.cfg.DATASETS.KEYPOINT_HFLIP_INDICES)
                bbox = transformed_annotation['bbox']
                area_box = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
                if area_box >= threshold_bbox_size:
                    annos_valid_idx.append(item_idx)
                else:
                    self.total_filtered_instances +=1
                
        # If no valid annotations remain for this item, return None
        if not annos_valid_idx:
            return None
        annos = [elem for i, elem in enumerate(item['annotations']) if i in annos_valid_idx]
        # Replace the original item's annotations with the filtered annotations
        item['annotations'] = annos
        return item

    def __getitem__(self, idx):
        return self.filtered_dataset[idx]

    def __len__(self):
        return len(self.filtered_dataset)

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

# import torch
# import logging
# from detectron2.config import CfgNode
# #from detectron2.data.datasets.coco import convert_to_coco_json
# from pycocotools.coco import COCO
# from detectron2.utils.file_io import PathManager
# import contextlib
# import io

# import contextlib
# import datetime
# import io
# import json
# import logging
# import numpy as np
# import os
# import shutil
# import pycocotools.mask as mask_util
# from fvcore.common.timer import Timer
# from iopath.common.file_io import file_lock
# from PIL import Image

# from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
# from detectron2.utils.file_io import PathManager

# logger = logging.getLogger(__name__)

# def convert_to_coco_dict(dataset_name, cfg=None):
#     """
#     Convert an instance detection/segmentation or keypoint detection dataset
#     in detectron2's standard format into COCO json format.

#     Generic dataset description can be found here:
#     https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

#     COCO data format description can be found here:
#     http://cocodataset.org/#format-data

#     Args:
#         dataset_name (str):
#             name of the source dataset
#             Must be registered in DatastCatalog and in detectron2's standard format.
#             Must have corresponding metadata "thing_classes"
#     Returns:
#         coco_dict: serializable dict in COCO json format
#     """

#     dataset_dicts = DatasetCatalog.get(dataset_name)
#     metadata = MetadataCatalog.get(dataset_name)

#     # unmap the category mapping ids for COCO
#     if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
#         reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
#         reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
#     else:
#         reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

#     categories = [
#         {"id": reverse_id_mapper(id), "name": name}
#         for id, name in enumerate(metadata.thing_classes)
#     ]

#     logger.info("Converting dataset dicts into COCO format")
#     coco_images = []
#     coco_annotations = []

#     from custom_mapper import custom_test_dataset_transform
#     for image_id, image_dict in enumerate(dataset_dicts):
#         image_dict = custom_test_dataset_transform(image_dict, cfg) # convert to json serializable

#         coco_image = {
#             "id": image_dict.get("image_id", image_id),
#             "width": int(image_dict["width"]),
#             "height": int(image_dict["height"]),
#             "file_name": str(image_dict["file_name"]),
#         }
#         coco_images.append(coco_image)

#         anns_per_image = image_dict.get("annotations", [])
#         for annotation in anns_per_image:
#             # create a new dict with only COCO fields
#             coco_annotation = {}

#             # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
#             bbox = annotation["bbox"]
#             if isinstance(bbox, np.ndarray):
#                 if bbox.ndim != 1:
#                     raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
#                 bbox = bbox.tolist()
#             if len(bbox) not in [4, 5]:
#                 raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
#             from_bbox_mode = annotation["bbox_mode"]
#             to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
#             bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

#             # COCO requirement: instance area
#             if "segmentation" in annotation:
#                 # Computing areas for instances by counting the pixels
#                 segmentation = annotation["segmentation"]
#                 # TODO: check segmentation type: RLE, BinaryMask or Polygon
#                 if isinstance(segmentation, list):
#                     polygons = PolygonMasks([segmentation])
#                     area = polygons.area()[0].item()
#                 elif isinstance(segmentation, dict):  # RLE
#                     area = mask_util.area(segmentation).item()
#                 else:
#                     raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
#             else:
#                 # Computing areas using bounding boxes
#                 if to_bbox_mode == BoxMode.XYWH_ABS:
#                     bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
#                     area = Boxes([bbox_xy]).area()[0].item()
#                 else:
#                     area = RotatedBoxes([bbox]).area()[0].item()

#             if "keypoints" in annotation:
#                 keypoints = annotation["keypoints"]  # list[int]
#                 for idx, v in enumerate(keypoints):
#                     if idx % 3 != 2:
#                         # COCO's segmentation coordinates are floating points in [0, H or W],
#                         # but keypoint coordinates are integers in [0, H-1 or W-1]
#                         # For COCO format consistency we substract 0.5
#                         # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
#                         keypoints[idx] = v - 0.5
#                 if "num_keypoints" in annotation:
#                     num_keypoints = annotation["num_keypoints"]
#                 else:
#                     num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

#             # COCO requirement:
#             #   linking annotations to images
#             #   "id" field must start with 1
#             coco_annotation["id"] = len(coco_annotations) + 1
#             coco_annotation["image_id"] = coco_image["id"]
#             coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
#             coco_annotation["area"] = float(area)
#             coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
#             coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))

#             # Add optional fields
#             if "keypoints" in annotation:
#                 coco_annotation["keypoints"] = keypoints
#                 coco_annotation["num_keypoints"] = num_keypoints

#             if "segmentation" in annotation:
#                 seg = coco_annotation["segmentation"] = annotation["segmentation"]
#                 if isinstance(seg, dict):  # RLE
#                     counts = seg["counts"]
#                     if not isinstance(counts, str):
#                         # make it json-serializable
#                         seg["counts"] = counts.decode("ascii")

#             coco_annotations.append(coco_annotation)

#     logger.info(
#         "Conversion finished, "
#         f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
#     )

#     info = {
#         "date_created": str(datetime.datetime.now()),
#         "description": "Automatically generated COCO json file for Detectron2.",
#     }
#     coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
#     if len(coco_annotations) > 0:
#         coco_dict["annotations"] = coco_annotations
#     return coco_dict

# def convert_to_coco_json(dataset_name, output_file, allow_cached=True, cfg=None):
#     """
#     Converts dataset into COCO format and saves it to a json file.
#     dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

#     Args:
#         dataset_name:
#             reference from the config file to the catalogs
#             must be registered in DatasetCatalog and in detectron2's standard format
#         output_file: path of json file that will be saved to
#         allow_cached: if json file is already present then skip conversion
#     """

#     # TODO: The dataset or the conversion script *may* change,
#     # a checksum would be useful for validating the cached data

#     PathManager.mkdirs(os.path.dirname(output_file))
#     with file_lock(output_file):
#         if PathManager.exists(output_file) and allow_cached:
#             logger.warning(
#                 f"Using previously cached COCO format annotations at '{output_file}'. "
#                 "You need to clear the cache file if your dataset has been modified."
#             )
#         else:
#             logger.info(f"Converting annotations of dataset '{dataset_name}' to COCO format ...)")
#             coco_dict = convert_to_coco_dict(dataset_name, cfg=cfg)

#             logger.info(f"Caching COCO format annotations at '{output_file}' ...")
#             tmp_file = output_file + ".tmp"
#             with PathManager.open(tmp_file, "w") as f:
#                 json.dump(coco_dict, f)
#             shutil.move(tmp_file, output_file)

# class CustomCOCOEvaluator(COCOEvaluator):
#     """
#     Evaluate AR for object proposals, AP for instance detection/segmentation, AP
#     for keypoint detection outputs using COCO's metrics.
#     See http://cocodataset.org/#detection-eval and
#     http://cocodataset.org/#keypoints-eval to understand its metrics.
#     The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
#     the metric cannot be computed (e.g. due to no predictions made).

#     In addition to COCO, this evaluator is able to support any bounding box detection,
#     instance segmentation, or keypoint detection dataset.
#     """

#     def __init__(
#         self,
#         dataset_name,
#         tasks=None,
#         distributed=True,
#         output_dir=None,
#         cfg=None,
#         *,
#         max_dets_per_image=None,
#         use_fast_impl=True,
#         kpt_oks_sigmas=(),
#     ):
#         """
#         Args:
#             dataset_name (str): name of the dataset to be evaluated.
#                 It must have either the following corresponding metadata:

#                     "json_file": the path to the COCO format annotation

#                 Or it must be in detectron2's standard dataset format
#                 so it can be converted to COCO format automatically.
#             tasks (tuple[str]): tasks that can be evaluated under the given
#                 configuration. A task is one of "bbox", "segm", "keypoints".
#                 By default, will infer this automatically from predictions.
#             distributed (True): if True, will collect results from all ranks and run evaluation
#                 in the main process.
#                 Otherwise, will only evaluate the results in the current process.
#             output_dir (str): optional, an output directory to dump all
#                 results predicted on the dataset. The dump contains two files:

#                 1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
#                    contains all the results in the format they are produced by the model.
#                 2. "coco_instances_results.json" a json file in COCO's result format.
#             max_dets_per_image (int): limit on the maximum number of detections per image.
#                 By default in COCO, this limit is to 100, but this can be customized
#                 to be greater, as is needed in evaluation metrics AP fixed and AP pool
#                 (see https://arxiv.org/pdf/2102.01066.pdf)
#                 This doesn't affect keypoint evaluation.
#             use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
#                 Although the results should be very close to the official implementation in COCO
#                 API, it is still recommended to compute results with the official API for use in
#                 papers. The faster implementation also uses more RAM.
#             kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
#                 See http://cocodataset.org/#keypoints-eval
#                 When empty, it will use the defaults in COCO.
#                 Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
#         """
#         self._cfg = cfg
#         self._logger = logging.getLogger(__name__)
#         self._distributed = distributed
#         self._output_dir = output_dir
#         self._use_fast_impl = use_fast_impl

#         # COCOeval requires the limit on the number of detections per image (maxDets) to be a list
#         # with at least 3 elements. The default maxDets in COCOeval is [1, 10, 100], in which the
#         # 3rd element (100) is used as the limit on the number of detections per image when
#         # evaluating AP. COCOEvaluator expects an integer for max_dets_per_image, so for COCOeval,
#         # we reformat max_dets_per_image into [1, 10, max_dets_per_image], based on the defaults.
#         if max_dets_per_image is None:
#             max_dets_per_image = [1, 10, 100]
#         else:
#             max_dets_per_image = [1, 10, max_dets_per_image]
#         self._max_dets_per_image = max_dets_per_image

#         if tasks is not None and isinstance(tasks, CfgNode):
#             kpt_oks_sigmas = (
#                 tasks.TEST.KEYPOINT_OKS_SIGMAS if not kpt_oks_sigmas else kpt_oks_sigmas
#             )
#             self._logger.warn(
#                 "COCO Evaluator instantiated using config, this is deprecated behavior."
#                 " Please pass in explicit arguments instead."
#             )
#             self._tasks = None  # Infering it from predictions should be better
#         else:
#             self._tasks = tasks

#         self._cpu_device = torch.device("cpu")

#         self._metadata = MetadataCatalog.get(dataset_name)
#         if not hasattr(self._metadata, "json_file"):
#             if output_dir is None:
#                 raise ValueError(
#                     "output_dir must be provided to COCOEvaluator "
#                     "for datasets not in COCO format."
#                 )
#             self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

#             cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
#             self._metadata.json_file = cache_path
#             convert_to_coco_json(dataset_name, cache_path, cfg=self._cfg)

#         json_file = PathManager.get_local_path(self._metadata.json_file)
#         with contextlib.redirect_stdout(io.StringIO()):
#             self._coco_api = COCO(json_file)

#         # Test set json files do not contain annotations (evaluation must be
#         # performed using the COCO evaluation server).
#         self._do_evaluation = "annotations" in self._coco_api.dataset
#         if self._do_evaluation:
#             self._kpt_oks_sigmas = kpt_oks_sigmas 


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
        return COCOEvaluator(dataset_name, tasks=("bbox", "keypoints"), distributed=True, output_dir=output_folder, 
                             cfg=cfg, kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS)

    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #     if output_folder is None:
    #         output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
    #         os.makedirs(output_folder, exist_ok=True)
    #     evaluator = COCOCustomEvaluator(
    #         dataset_name,
    #         tasks=("bbox", ),
    #         #tasks=("bbox", "keypoints"),
    #         use_fast_impl=True,
    #         distributed=True,
    #         output_dir=output_folder,
    #         kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS#,
    #         #cfg = cfg
    #     )
    #     return evaluator
    
    # not working as desired
    # def build_hooks(self):
    #     hooks = super().build_hooks()   # 

    #     # pop the standard evaluation hook. Just want to evaluate the validation loss
    #     # hooks.pop(4)
    #     try:
    #         hooks.insert(
    #             -1,
    #             LossEvalHook(
    #                 self.cfg.TEST.LOSS_VALIDATION_PERIOD,
    #                 self.model,
    #                 build_detection_test_loader(
    #                     self.cfg,
    #                     self.cfg.DATASETS.TEST[0],
    #                     mapper=lambda d: custom_mapper(d, self.cfg, is_train=False)
    #                     #DatasetMapper(self.cfg, True, augmentations=[]),
    #                 ),
    #             ),
    #         )
    #         # hooks.insert(-1, LearningRateHook(self.cfg, 100))
    #     except IndexError as e:
    #         print(e)
    #     return hooks
    
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
        with open("config/config.json", 'r') as f:
            cfg = json.load(f)
    except Exception as ex:
        sys.exit("provided cfg file path not valid")

    from start_training import setup_parameters, setup_config
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