from detectron2.data.transforms.augmentation import Augmentation
import detectron2.data.transforms as T
from fvcore.transforms.transform import (
    CropTransform, Transform, PadTransform, TransformList
)
from detectron2.data.transforms.transform import ResizeTransform, NoOpTransform
from detectron2.data import detection_utils as utils

from model_config import KINTREE_TABLE

from itertools import combinations
from PIL import Image 
import torch
import numpy as np
import cv2
import copy
import glob
import json
import os
import copy
import base64
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
#from model_config import cfg

class Resize(Augmentation):
    """
    Resize the image based on a rang or choice of edge length
    """

    @torch.jit.unused
    def __init__(
        self, short_edge_length, long_edge_length, sample_style="range", interp=Image.NEAREST
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            long_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the longest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        self.interp = interp
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if isinstance(short_edge_length, int):
            long_edge_length = (long_edge_length, long_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
            assert len(long_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {long_edge_length}!"
            )
        self._init(locals())

    @torch.jit.unused
    def get_transform(self, image):
        h, w = image.shape[:2]
        if self.is_range:
            newh = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
            neww = np.random.randint(self.long_edge_length[0], self.long_edge_length[1] + 1)
        else:
            newh = np.random.choice(self.short_edge_length)
            neww = np.random.choice(self.long_edge_length)
        if newh == 0 or neww == 0:
            return NoOpTransform()
        return ResizeTransform(h, w, newh, neww, self.interp)

class RandomHorizonZoom(Augmentation):
    """
    Randomly crop a subimage out of an image, while maintaining the aspect ratio of the original.
    Example: original image (height: 1024, width: 2048), random crop size: single entry with a ratio of 1/8

    1. Calculate the crop size:
    The crop size is determined based on the random crop ratio (1/8, 1/4) and the original image dimensions.
    Since the height of the crop is specified as one-eighth of the original image height (1024 / 8 = 128), the crop height will be 128 pixels.

    2. Perform the random crop:
    The random crop is applied to the original image while maintaining the aspect ratio. Since the crop height is 128 pixels, 
    the crop width is calculated to maintain the original aspect ratio (2048 / 1024 = 2). 
    Therefore, the crop width will be 2 * 128 = 256 pixels.

    3. Select the crop region:
    The center point of the crop region aligns with the center point of the original image. 
    The crop region will have a height of 128 pixels and a width of 256 pixels.

    4. Generate the final image:
    The final image will be the cropped region extracted from the original image, 
    resulting in an image with a height of 128 pixels and a width of 256 pixels. 
    The pixel information within the cropped region remains the same, preserving the details of the selected portion of the original image.
    """

    def __init__(self, crop_size):
        """
        Args:
            crop_size (list[float]): the relative ratio
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        """
        calculates the starting coordinates for the crop based on the center of the image 
        and returns a CropTransform object with the calculated coordinates and crop size.
        """
        h, w = image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        
        if h >= croph and w >= cropw:
            w0 = int(w / 2.0 - cropw / 2.0)
            h0 = int(h / 2.0 - croph / 2.0)
            return CropTransform(w0, h0, cropw, croph)
        else:
            w0 = int(w / 2.0)
            h0 = int(h / 2.0)
            ph = int((croph-h)//2)
            return PadTransform(0, ph, 0, ph)

    def get_crop_size(self, image_size):
        """
        crops both the height and the width of the image. 
        The crop size is determined based on the specified crop ratio (crop_size) and the original image dimensions (h for height and w for width). 
        The resulting crop size will have a different aspect ratio than the original image due to the relative crop ratio.

        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        self.aspect = h/w
        # random sample a crop size
        self.crop_size_ = np.random.choice(self.crop_size)
        # build crop size while maintaining the aspect ratio
        #self.crop_size = (self.aspect*self.crop_size, self.crop_size)
        ch = self.crop_size_    # crop height
        return int(h * ch + 0.5), w

import sys
import warnings

def custom_mapper(dataset_dict, cfg, is_train=True):
    """[summary]

    Args:
        dataset_dict ([list]): a dict containing image informations and annotation data
        is_train (bool, optional): training flag

    Returns:
        [list]: a dict containing image informations, annotation data, and image data for training step
    """
    keypoint_hflip_indices = cfg.DATASETS.KEYPOINT_HFLIP_INDICES
    if cfg.INPUT.RANDOM_FLIP != 'none' and not keypoint_hflip_indices:
        sys.exit("Error: Flip prompted but no keypoint_hflip_indices provided")

    # is saved as BGR and at model_config set to RGB channels will be reversed
    hha_img = cv2.imread(dataset_dict["file_name"], cv2.IMREAD_UNCHANGED)
    
    if cfg.INPUT.USE_AUGMENTATION:
        vertical, horizontal = False, False
        if cfg.INPUT.RANDOM_FLIP == "horizontal":   # only horizontal
            horizontal = True
        if cfg.INPUT.RANDOM_FLIP == "vertical":
            vertical = True

        if isinstance(is_train, bool):
            if is_train:
                zoom_factor = float(np.random.choice(cfg.INPUT.AUG_ZOOM_TRAIN))
            else:   
                # deprecated, value of 0.125 and 0.25 corresponds to 22.5degree and 45degree fov, respectively, 
                # or more precisely: 180 * zoom_factor -> fov in degree, can be changed at data creation
                zoom_factor = 0.125
        if isinstance(is_train, tuple):
            assert len(is_train)==2, "is_train flag expects format tuple(bool, float)"
            assert isinstance(is_train[1], float), "second tuple item needs to be of type float"
            zoom_factor = is_train[1] # 0.125, 0.25
        transform_list_shape = [
            RandomHorizonZoom([zoom_factor]),
            ResizeTransform(int(zoom_factor*hha_img.shape[0]), int(hha_img.shape[1]), int(max(cfg.INPUT.AUG_ZOOM_TRAIN)*hha_img.shape[0]), int(hha_img.shape[1]), Image.NEAREST)
        ]
        image, transforms = T.apply_transform_gens(transform_list_shape, hha_img)

        

    else:
        image, transforms = T.apply_transform_gens([], hha_img)

    image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32")) # channel on first dimension

    # augmentation of the annotations
    annotations = copy.deepcopy(dataset_dict['annotations'])
    
    # Convert JSON serializable 'normal' string back to binary string to be usable with detectrons' functions
    for anno in annotations:
        anno['segmentation']['counts'] = base64.b64decode(anno['segmentation']['counts'])

    with warnings.catch_warnings():
        # transforms the annotations accodining to new augemntation rules
        # will only effect attributes of "bbox", "segmentation", "keypoints"
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        annos = [utils.transform_instance_annotations(copy.deepcopy(annotation), [transforms], image.shape[1:], keypoint_hflip_indices=keypoint_hflip_indices)
                for annotation in annotations if annotation.get("iscrowd", 0) == 0]
    
    # discard instances which doesn't meet the following requirements:
        # 1. keypoints which got visibility label "v=0": not labeled (in which case also x=y=0), only possible after augmentation, data are only v=1 OR v=2
        # 2. median pcl distance is larger than threshold, at 50km/h circa 40m Distance to fully stop (incl. reaction time) https://www.autobild.de/artikel/bremsweg-formel-13443369.html
    filtered_annos = [obj_dict for obj_dict in annos if (not np.any(obj_dict['keypoints'][:, -1] == 0)) \
                        and np.median(np.linalg.norm(np.array(obj_dict['keypoints_3d'], dtype=np.float32)[:, :-1], axis=1)) <= cfg.INPUT.MAX_OBJ_DISTANCE]
    
    annos = filtered_annos
    dataset_dict["image"] = image
    dataset_dict["height"], dataset_dict["width"] = image.shape[1:]
    instances = utils.annotations_to_instances(annos, image.shape[1:], mask_format = cfg.INPUT.MASK_FORMAT)
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    dataset_dict["annotations"] = annos
    
    return dataset_dict

from itertools import chain
def evaluator_mapper(dataset_dict, cfg, is_train=False):
    # json serializable annotations without image information, 
    # plain annotation used to save to coco when testing once and then accessed after each eval iter
    dset_mapped_org = custom_mapper(dataset_dict, cfg, is_train=is_train)
    dset_mapped = copy.deepcopy(dset_mapped_org)
    del dset_mapped['image']
    del dset_mapped['origin_dataset_id']
    del dset_mapped['origin_frame_id']
    del dset_mapped['instances']
    
    for anno_obj in dset_mapped['annotations']:
        del anno_obj['segmentation']
        # 'segm_pcl' and 'keypoints_3d' are optional not needed here
        del anno_obj['segm_pcl']
        del anno_obj['keypoints_3d']
        anno_obj['bbox'] = anno_obj['bbox'].tolist()
        #anno_obj['segmentation'] = anno_obj['segmentation'].tolist()
        # # # for idx_segm, segm in enumerate(anno_obj['segmentation']):
        # # #     anno_obj['segmentation'][idx_segm] = segm.tolist()
        anno_obj['keypoints'] = [float(elem) for elem in list(chain(*anno_obj['keypoints']))]

    '''
    TODO:
    File "/workspace/repos/keypoint_trainer.py", line 110, in <lambda>
    cfg, dataset_name, mapper=lambda d: custom_mapper(d, cfg, is_train=False)
    File "/workspace/repos/custom_mapper.py", line 187, in custom_mapper
    image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32")) # channel on first dimension
UnboundLocalError: local variable 'image' referenced before assignment

    ''' 

    return dset_mapped

if __name__ == "__main__":
    pass
    # run start_training.py script for testing the mapper function
