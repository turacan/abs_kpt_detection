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

    dataset_dict = copy.deepcopy(dataset_dict)
    hha_img = cv2.imread(dataset_dict["file_name"], cv2.IMREAD_UNCHANGED)
    
    if cfg.INPUT.USE_AUGMENTATION:
        vertical, horizontal = False, False
        if cfg.INPUT.RANDOM_FLIP == "horizontal":   # only horizontal
            horizontal = True
        if cfg.INPUT.RANDOM_FLIP == "vertical":
            vertical = True

        cropw = np.random.choice(np.linspace(1/3 * hha_img.shape[1], 7/8 * hha_img.shape[1], num=5, dtype=int))
        croph = int(hha_img.shape[0]*cfg.INPUT.AUG_ZOOM_TRAIN[0])        
        if is_train:
            transform_list_shape = [
                RandomHorizonZoom(cfg.INPUT.AUG_ZOOM_TRAIN),
                T.RandomCrop("absolute", (croph, cropw)),
                Resize(short_edge_length=[cfg.INPUT.AUG_MIN_SIZE_TRAIN[0]],long_edge_length=[cfg.INPUT.AUG_MAX_SIZE_TRAIN[0]], sample_style="choice"),
            ]
            # if vertical or horizontal:
            #     transform_list_shape.append(T.RandomFlip(prob=0.5, horizontal=horizontal, vertical=vertical))
                
            image, transforms = T.apply_transform_gens(transform_list_shape, hha_img)
        else:
            transform_list_shape = [
                RandomHorizonZoom(cfg.INPUT.AUG_ZOOM_TRAIN),
                Resize(short_edge_length=[cfg.INPUT.AUG_MIN_SIZE_TRAIN[0]],long_edge_length=[cfg.INPUT.AUG_MAX_SIZE_TRAIN[0]], sample_style="choice"),
            ]
            image, transforms = T.apply_transform_gens(transform_list_shape, hha_img)
    else:
        image, transforms = T.apply_transform_gens([], hha_img)

    # test = np.uint8(255*image)
    # cv2.imshow('test', test)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # for elem in annos:
    #     box = elem['bbox']
    #     test = cv2.rectangle(test, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 255, 255], 1)
         
    image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32")) # channel on first dimension

    # augmentation of the annotations
    annotations = copy.deepcopy(dataset_dict['annotations'])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        annos = [utils.transform_instance_annotations(copy.deepcopy(annotation), [transforms], image.shape[1:], keypoint_hflip_indices=keypoint_hflip_indices)
                for annotation in annotations if annotation.get("iscrowd", 0) == 0]
    
    annos_temp = copy.deepcopy(annos)

    '''
    TODO: keypoint correction logic, need origanl unaltered joint infor to calc direction vec
        how about adding padding pixel to the objetcs: 
    '''
    annos = []
    annos_ = copy.deepcopy(annos_temp)  
    dir_vectors = []
    threshold_bbox_size = 40
    for orig_dset_dict, augm_dset_dict in zip(annotations, annos_):
        box = augm_dset_dict['bbox']
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        if area_box < threshold_bbox_size or np.abs(box[2]-box[0])<2:  # skip if bbox size is too small
            continue

        kpts = np.array(orig_dset_dict['keypoints']).reshape(-1,3) 
        augm_kpts = augm_dset_dict['keypoints']
        compare_kpts = copy.deepcopy(augm_dset_dict['keypoints'])
    
        mask_out_of_bounds = np.zeros(shape=(kpts.shape[0],), dtype=bool )
        for idx_kpt, kpt in enumerate(augm_kpts):
            if not (kpt[0] > box[0] and kpt[0] <= box[2] and kpt[1] > box[1] and kpt[1] <= box[3]):
                box_width = box[2] - box[0]
                diff_abs_min = np.min(np.abs(np.array([box[0], box[2]]) - kpt[0]))
                if diff_abs_min < box_width*0.2:
                    continue
                else:
                    mask_out_of_bounds[idx_kpt] = True

        # calculate direction vectors for each joint connection
        temp_vec = []
        flag = True
        for connection in KINTREE_TABLE.T:
            
            dx = kpts[connection[1], 0] - kpts[connection[0], 0]    # direction vector x orig
            dy = kpts[connection[1], 1] - kpts[connection[0], 1]    # direction vector y orig
            
            magnitude = np.sqrt(dx**2 + dy**2)  # irrelevant in original (here), but in augmented relevant

            if flag:
                if magnitude > int(hha_img.shape[1]/2)-1:
                    altered_kpts = copy.deepcopy(kpts)
                    right_side = np.argwhere((altered_kpts[...,0]- int(hha_img.shape[1]/2) -1) > 0).squeeze()
                    if right_side.size == 1:
                        right_side = right_side[np.newaxis,...]
                    left_side = list(set(range(0,16)).difference(right_side))

                    min_on_right_side = np.min(altered_kpts[right_side,...,0])
                    offset = hha_img.shape[1]-1 - min_on_right_side
                    altered_kpts[right_side,...,0] = altered_kpts[right_side,...,0] - min_on_right_side
                    if len(left_side) > 0:
                        altered_kpts[left_side,...,0] = altered_kpts[left_side,...,0] + offset
                    
                    if altered_kpts.min() < 0:
                        print("KPT adjustment <0")
                    # altered_kpts[...,0] = altered_kpts[...,0] - offset
                    # altered_kpts[...,0] = altered_kpts[...,0] - np.min(altered_kpts[..., 0])   # kpts x value starts at 0
                    flag = False

            if flag:    # no instance which is on both image sides
                normalized_vector = (dx / magnitude, dy / magnitude)
                temp_vec.append([normalized_vector[0], normalized_vector[1], 0])
            if not flag:
                dx = altered_kpts[connection[1], 0] - altered_kpts[connection[0], 0]    # direction vector x orig
                dy = altered_kpts[connection[1], 1] - altered_kpts[connection[0], 1]    # direction vector y orig

                magnitude = np.sqrt(dx**2 + dy**2)  # irrelevant in original (here), but in augmented relevant
                normalized_vector = (dx / magnitude, dy / magnitude)

                temp_vec.append([normalized_vector[0], normalized_vector[1]])

        # look if some kpts (of augm) are marked as 'not labeled' because of out of bounds
        #dir_vectors = np.array(temp_vec) 

        augm_kpts[mask_out_of_bounds, -1] = 0
        

        # jnts_out_of_bounds_connection = np.argwhere(dir_vectors[...,2] == 1).squeeze()  # out of bounds are labled above with flag 1
        # if jnts_out_of_bounds_connection.size == 1:
        #     jnts_out_of_bounds_connection = jnts_out_of_bounds_connection[np.newaxis,...]
        # dir_vectors = dir_vectors[jnts_out_of_bounds_connection]

        # task_kpts = copy.deepcopy(augm_kpts)
        # if jnts_out_of_bounds_connection.size>0:
            
        #     kintree_connection = [KINTREE_TABLE.T[problematic_connection] for problematic_connection in jnts_out_of_bounds_connection]
        #     for connection, vec in zip(kintree_connection, dir_vectors[...,:2]):  
        #         unlabeled_pos = connection[1]
        #         pos_at_kintree = (np.argwhere(KINTREE_TABLE.T == unlabeled_pos)).tolist()
        #         pos_at_kintree = sorted(pos_at_kintree, key=lambda x: (KINTREE_TABLE.T[tuple(x)], -x[1]))  # [[2, 0], [1, 1]]

        #         for pos in copy.deepcopy(pos_at_kintree):   # could be more than 1 occurance, e.g. pelvis
        #             vec = np.array(temp_vec[pos[0]][:2]) 
        #             # if pos[1] == 1:
        #             start_point = task_kpts[KINTREE_TABLE.T[pos[0], 0]][:2]

        #             #start_point = task_kpts[connection[0]][:2]
        #             flag = False
        #             for length in np.linspace(10, 100, 50):
        #                 end_point = start_point + vec * length  # if u >= x_min and u <= x_max and v >= y_min and v <= y_max:
        #                 if end_point[0] > box[0] and end_point[0] <= box[2] and end_point[1] > box[1] and end_point[1] <= box[3]:
        #                     real_end_point = end_point
        #                     flag = True
        #                 else:
        #                     break
        #             if flag:
        #                 task_kpts[unlabeled_pos] = np.array([real_end_point[0], real_end_point[1], 1]) # KINTREE_TABLE.T[pos[0], 1]
        #                 pos_at_kintree.remove(pos)
        #                 #task_kpts[connection[1]] = np.array([real_end_point[0], real_end_point[1], 1]) # KINTREE_TABLE.T[pos[0], 1]
        #             else:
        #                 task_kpts[unlabeled_pos] = np.array([0., 0., -1]) 
        #                 pos_at_kintree.remove(pos)

        #     augm_dset_dict['keypoints'] = task_kpts   

        unlabeled_jnt_pos = np.argwhere(augm_dset_dict['keypoints'][...,2] == 0) # which joints are 'not labeled' or out of bounds
        orig_unlabeled_size = unlabeled_jnt_pos.size
        
        while unlabeled_jnt_pos.size>0:
            orig_unlabeled_size = unlabeled_jnt_pos.size
            
            task_kpts = copy.deepcopy(augm_dset_dict['keypoints'])
            for unlabeled_pos in unlabeled_jnt_pos:
                pos_at_kintree = (np.argwhere(KINTREE_TABLE.T == unlabeled_pos)).tolist()
                pos_at_kintree = sorted(pos_at_kintree, key=lambda x: (KINTREE_TABLE.T[tuple(x)], -x[1]))  # [[2, 0], [1, 1]]

                for pos in copy.deepcopy(pos_at_kintree):   # could be more than 1 occurance, e.g. pelvis
                    # if pos[1] != 0:
                    vec = np.array(temp_vec[pos[0]][:2])    # direction vector from starting point (first idx)
                    start_point = task_kpts[KINTREE_TABLE.T[pos[0], 0]][:2]

                    flag = False
                    for length in np.linspace(10, 100, 50):
                        end_point = start_point + vec * length  # if u >= x_min and u <= x_max and v >= y_min and v <= y_max:
                        if (end_point[0]<0) or (end_point[0]>(image.shape[-1]-1)):  # end_point out of image bounds
                            break

                        if end_point[0] >= box[0] and end_point[0] <= box[2] and end_point[1] >= box[1] and end_point[1] <= box[3]:
                            real_end_point = end_point
                            flag = True
                        else:
                            # box_width = box[2] - box[0]     # also must be in image frame
                            # if end_point[0] > box[2]: # point on right
                            #     diff = end_point[0] - box[2]
                            #     if diff < box_width*0.05:
                            #         flag = True

                            # elif end_point[0] < box[0]: # point on left
                            #     diff = box[0] - end_point[0]
                            #     if diff < box_width*0.05:
                            #         flag = True
                            break

                    if flag:
                        task_kpts[unlabeled_pos] = np.array([real_end_point[0], real_end_point[1], 1]) # KINTREE_TABLE.T[pos[0], 1]
                        pos_at_kintree.remove(pos)
                    else:   # desired jnt has no place
                        task_kpts[unlabeled_pos] = np.array([0., 0., -1]) 
                        pos_at_kintree.remove(pos)
              
            augm_dset_dict['keypoints'] = task_kpts 
            unlabeled_jnt_pos = np.argwhere(augm_dset_dict['keypoints'][...,2] == 0) # which joints are 'not labeld'
            if unlabeled_jnt_pos.size == orig_unlabeled_size: 
                break

            # task_kpts = task_kpts.astype(int)
            # img_test = np.zeros(shape=(image.shape[1:]), dtype=np.uint8)
            # for pnts in task_kpts[...,:2]:
            #     img_test[pnts[1], pnts[0]] = 255
            # cv2.imshow('test', img_test)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        augm_dset_dict['keypoints'][np.argwhere(augm_dset_dict['keypoints'][...,2] == -1).squeeze()] = np.array([0., 0., 0.]) 
        
        temp = np.argwhere(augm_dset_dict['keypoints'][...,-1]>0).squeeze()
        if temp.size == 0:  # probably object which is on both border edges and bbox is on opposite site
            continue
        kpt_min_x, kpt_max_x = augm_dset_dict['keypoints'][temp,0].min(), augm_dset_dict['keypoints'][temp,0].max()
        kpt_min_y, kpt_max_y = augm_dset_dict['keypoints'][temp,1].min(), augm_dset_dict['keypoints'][temp,1].max()

        # check if bboxes should be adjusted
        if kpt_min_x<augm_dset_dict['bbox'][0]:
            temp = int(np.floor(kpt_min_x)) 
            if temp>=0:
                augm_dset_dict['bbox'][0] = temp
        if kpt_max_x>augm_dset_dict['bbox'][2]:
            temp = int(np.ceil(kpt_max_x))
            if temp<image.shape[-1]:
                augm_dset_dict['bbox'][2] = temp
        if kpt_min_y<augm_dset_dict['bbox'][1]:
            temp = int(np.floor(kpt_min_y))
            if temp>=0:
                augm_dset_dict['bbox'][1] = temp
        if kpt_max_y>augm_dset_dict['bbox'][3]:
            temp = int(np.ceil(kpt_max_y))
            if temp<image.shape[1]:
                augm_dset_dict['bbox'][3] = temp
        annos.append(augm_dset_dict)
    
    # Filter instances which don't meet some criteria (here: bbox size)
    # threshold_bbox_size = 40 #cfg.INPUT.AUG_MIN_SIZE_TRAIN[0] * cfg.INPUT.AUG_MAX_SIZE_TRAIN[0] * 0.001 
    # annos = []
    # for obj in annos_temp:
    #     box = obj['bbox']
    #     area_box = (box[2] - box[0]) * (box[3] - box[1])
    #     if area_box >= threshold_bbox_size:
    #         annos.append(obj)   # excludes key value pairs of orig dict (total len may differ)


    # dataset_dict["xyz_img"] = xyz_img
    dataset_dict["image"] = image
    dataset_dict["height"], dataset_dict["width"] = image.shape[1:]
    instances = utils.annotations_to_instances(annos, image.shape[1:], mask_format = cfg.INPUT.MASK_FORMAT)
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    dataset_dict["annotations"] = annos
    
    return dataset_dict


def custom_mapper_org(dataset_dict, cfg, is_train=True):
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

    dataset_dict = copy.deepcopy(dataset_dict)
    hha_img = cv2.imread(dataset_dict["file_name"], cv2.IMREAD_UNCHANGED)
    
    if cfg.INPUT.USE_AUGMENTATION:
        vertical, horizontal = False, False
        if cfg.INPUT.RANDOM_FLIP == "horizontal":   # only horizontal
            horizontal = True
        if cfg.INPUT.RANDOM_FLIP == "vertical":
            vertical = True

        cropw = np.random.choice(np.linspace(1/3 * hha_img.shape[1], 7/8 * hha_img.shape[1], num=5, dtype=int))
        croph = int(hha_img.shape[0]*cfg.INPUT.AUG_ZOOM_TRAIN[0])        
        if is_train:
            transform_list_shape = [
                RandomHorizonZoom(cfg.INPUT.AUG_ZOOM_TRAIN),
                T.RandomCrop("absolute", (croph, cropw)),
                Resize(short_edge_length=[cfg.INPUT.AUG_MIN_SIZE_TRAIN[0]],long_edge_length=[cfg.INPUT.AUG_MAX_SIZE_TRAIN[0]], sample_style="choice"),
            ]
            # if vertical or horizontal:
            #     transform_list_shape.append(T.RandomFlip(prob=0.5, horizontal=horizontal, vertical=vertical))
                
            image, transforms = T.apply_transform_gens(transform_list_shape, hha_img)
        else:
            transform_list_shape = [
                RandomHorizonZoom(cfg.INPUT.AUG_ZOOM_TRAIN),
                Resize(short_edge_length=[cfg.INPUT.AUG_MIN_SIZE_TRAIN[0]],long_edge_length=[cfg.INPUT.AUG_MAX_SIZE_TRAIN[0]], sample_style="choice"),
            ]
            image, transforms = T.apply_transform_gens(transform_list_shape, hha_img)
    else:
        image, transforms = T.apply_transform_gens([], hha_img)

    # test = np.uint8(255*image)
    # cv2.imshow('test', test)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # for elem in annos:
    #     box = elem['bbox']
    #     test = cv2.rectangle(test, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 255, 255], 1)
         
    image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32")) # channel on first dimension

    # augmentation of the annotations
    annotations = copy.deepcopy(dataset_dict['annotations'])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        annos = [utils.transform_instance_annotations(copy.deepcopy(annotation), [transforms], image.shape[1:], keypoint_hflip_indices=keypoint_hflip_indices)
                for annotation in annotations if annotation.get("iscrowd", 0) == 0]
    
    annos_temp = copy.deepcopy(annos)

    '''
    TODO: keypoint correction logic, need origanl unaltered joint infor to calc direction vec
        how about adding padding pixel to the objetcs: 
    '''
    annos = []
    annos_ = copy.deepcopy(annos_temp)  
    dir_vectors = []
    threshold_bbox_size = 40
    for orig_dset_dict, augm_dset_dict in zip(annotations, annos_):
        box = augm_dset_dict['bbox']
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        if area_box < threshold_bbox_size or np.abs(box[2]-box[0])<2:  # skip if bbox size is too small
            continue

        kpts = np.array(orig_dset_dict['keypoints']).reshape(-1,3) 
        augm_kpts = augm_dset_dict['keypoints']
    
        
        temp_vec = []
        flag = True
        for connection in KINTREE_TABLE.T:
            
            dx = kpts[connection[1], 0] - kpts[connection[0], 0]    # direction vector x orig
            dy = kpts[connection[1], 1] - kpts[connection[0], 1]    # direction vector y orig
            
            magnitude = np.sqrt(dx**2 + dy**2)  # irrelevant in original (here), but in augmented relevant

            if flag:
                if magnitude > int(hha_img.shape[1]/2)-1:
                    altered_kpts = copy.deepcopy(kpts)
                    right_side = np.argwhere((altered_kpts[...,0]- int(hha_img.shape[1]/2) -1) > 0).squeeze()
                    if right_side.size == 1:
                        right_side = right_side[np.newaxis,...]
                    left_side = list(set(range(0,16)).difference(right_side))

                    min_on_right_side = np.min(altered_kpts[right_side,...,0])
                    offset = hha_img.shape[1]-1 - min_on_right_side
                    altered_kpts[right_side,...,0] = altered_kpts[right_side,...,0] - min_on_right_side
                    if len(left_side) > 0:
                        altered_kpts[left_side,...,0] = altered_kpts[left_side,...,0] + offset
                    
                    if altered_kpts.min() < 0:
                        print("KPT adjustment <0")
                    # altered_kpts[...,0] = altered_kpts[...,0] - offset
                    # altered_kpts[...,0] = altered_kpts[...,0] - np.min(altered_kpts[..., 0])   # kpts x value starts at 0
                    flag = False

            if flag:    # no instance which is on both image sides
                normalized_vector = (dx / magnitude, dy / magnitude)
                temp_vec.append([normalized_vector[0], normalized_vector[1], 0])
            if not flag:
                dx = altered_kpts[connection[1], 0] - altered_kpts[connection[0], 0]    # direction vector x orig
                dy = altered_kpts[connection[1], 1] - altered_kpts[connection[0], 1]    # direction vector y orig

                magnitude = np.sqrt(dx**2 + dy**2)  # irrelevant in original (here), but in augmented relevant
                normalized_vector = (dx / magnitude, dy / magnitude)

                
                temp_vec.append([normalized_vector[0], normalized_vector[1], 1])

        # look if some kpts (of augm) are marked as 'not labeled' because of out of bounds
        dir_vectors = np.array(temp_vec) 
        jnts_out_of_bounds_connection = np.argwhere(dir_vectors[...,2] == 1).squeeze()  # out of bounds are labled above with flag 1
        if jnts_out_of_bounds_connection.size == 1:
            jnts_out_of_bounds_connection = jnts_out_of_bounds_connection[np.newaxis,...]
        dir_vectors = dir_vectors[jnts_out_of_bounds_connection]

        task_kpts = copy.deepcopy(augm_kpts)
        if jnts_out_of_bounds_connection.size>0:
            
            kintree_connection = [KINTREE_TABLE.T[problematic_connection] for problematic_connection in jnts_out_of_bounds_connection]
            for connection, vec in zip(kintree_connection, dir_vectors[...,:2]):  
                unlabeled_pos = connection[1]
                pos_at_kintree = (np.argwhere(KINTREE_TABLE.T == unlabeled_pos)).tolist()
                pos_at_kintree = sorted(pos_at_kintree, key=lambda x: (KINTREE_TABLE.T[tuple(x)], -x[1]))  # [[2, 0], [1, 1]]

                for pos in copy.deepcopy(pos_at_kintree):   # could be more than 1 occurance, e.g. pelvis
                    vec = np.array(temp_vec[pos[0]][:2]) 
                    # if pos[1] == 1:
                    start_point = task_kpts[KINTREE_TABLE.T[pos[0], 0]][:2]

                    #start_point = task_kpts[connection[0]][:2]
                    flag = False
                    for length in np.linspace(10, 100, 50):
                        end_point = start_point + vec * length  # if u >= x_min and u <= x_max and v >= y_min and v <= y_max:
                        if end_point[0] > box[0] and end_point[0] <= box[2] and end_point[1] > box[1] and end_point[1] <= box[3]:
                            real_end_point = end_point
                            flag = True
                        else:
                            break
                    if flag:
                        task_kpts[unlabeled_pos] = np.array([real_end_point[0], real_end_point[1], 1]) # KINTREE_TABLE.T[pos[0], 1]
                        pos_at_kintree.remove(pos)
                        #task_kpts[connection[1]] = np.array([real_end_point[0], real_end_point[1], 1]) # KINTREE_TABLE.T[pos[0], 1]
                    else:
                        task_kpts[unlabeled_pos] = np.array([0., 0., -1]) 
                        pos_at_kintree.remove(pos)

            augm_dset_dict['keypoints'] = task_kpts   

        unlabeled_jnt_pos = np.argwhere(augm_dset_dict['keypoints'][...,2] == 0) # which joints are 'not labeled'
        orig_unlabeled_size = unlabeled_jnt_pos.size
        
        while unlabeled_jnt_pos.size>0:
            orig_unlabeled_size = unlabeled_jnt_pos.size
            
            task_kpts = copy.deepcopy(augm_dset_dict['keypoints'])
            for unlabeled_pos in unlabeled_jnt_pos:
                pos_at_kintree = (np.argwhere(KINTREE_TABLE.T == unlabeled_pos)).tolist()
                pos_at_kintree = sorted(pos_at_kintree, key=lambda x: (KINTREE_TABLE.T[tuple(x)], -x[1]))  # [[2, 0], [1, 1]]

                for pos in copy.deepcopy(pos_at_kintree):   # could be more than 1 occurance, e.g. pelvis
                    # if pos[1] != 0:
                    vec = np.array(temp_vec[pos[0]][:2])    # direction vector from starting point (first idx)
                    start_point = task_kpts[KINTREE_TABLE.T[pos[0], 0]][:2]

                    flag = False
                    for length in np.linspace(10, 100, 50):
                        end_point = start_point + vec * length  # if u >= x_min and u <= x_max and v >= y_min and v <= y_max:
                        if end_point[0] > box[0] and end_point[0] <= box[2] and end_point[1] > box[1] and end_point[1] <= box[3]:
                            real_end_point = end_point
                            flag = True
                        else:
                            break
                    if flag:
                        task_kpts[unlabeled_pos] = np.array([real_end_point[0], real_end_point[1], 1]) # KINTREE_TABLE.T[pos[0], 1]
                        pos_at_kintree.remove(pos)
                    else:   # desired jnt has no place
                        task_kpts[unlabeled_pos] = np.array([0., 0., -1]) 
                        pos_at_kintree.remove(pos)
              
            augm_dset_dict['keypoints'] = task_kpts
            unlabeled_jnt_pos = np.argwhere(augm_dset_dict['keypoints'][...,2] == 0) # which joints are 'not labeld'
            if unlabeled_jnt_pos.size == orig_unlabeled_size: 
                break

            # task_kpts = task_kpts.astype(int)
            # img_test = np.zeros(shape=(image.shape[1:]), dtype=np.uint8)
            # for pnts in task_kpts[...,:2]:
            #     img_test[pnts[1], pnts[0]] = 255
            # cv2.imshow('test', img_test)
            # cv2.waitKey(0)

        augm_dset_dict['keypoints'][np.argwhere(augm_dset_dict['keypoints'][...,2] == -1).squeeze()] = np.array([0., 0., 0.]) 
        annos.append(augm_dset_dict)
    
   
    # dataset_dict["xyz_img"] = xyz_img
    dataset_dict["image"] = image
    dataset_dict["height"], dataset_dict["width"] = image.shape[1:]
    instances = utils.annotations_to_instances(annos, image.shape[1:], mask_format = cfg.INPUT.MASK_FORMAT)
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    dataset_dict["annotations"] = annos
    
    return dataset_dict
    
#from fvcore.transforms.transform import TransformList, CropTransform, ResizeTransform
from itertools import chain
def custom_test_dataset_transform(dataset_dict, cfg):
    dataset_dict = copy.deepcopy(dataset_dict)
    transforms = TransformList([CropTransform(x0=0, y0=384, w=2048, h=256), ResizeTransform(h=256, w=2048, new_h=800, new_w=1333, interp=0)])
    
    annotations = copy.deepcopy(dataset_dict['annotations'], cfg)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        annos = [utils.transform_instance_annotations(copy.deepcopy(annotation), [transforms], (800, 1333), keypoint_hflip_indices=cfg.DATASETS.KEYPOINT_HFLIP_INDICES)
                for annotation in annotations if annotation.get("iscrowd", 0) == 0]
    
    annos_temp = copy.deepcopy(annos)

    # Filter instances which don't meet some criteria (here: bbox size)
    threshold_bbox_size = 40 
    annos = []
    for orig_dset_dict, augm_dset_dict in zip(annotations, annos_temp):
        box = augm_dset_dict['bbox']
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        if area_box < threshold_bbox_size or np.abs(box[2]-box[0])<2:  # skip if bbox size is too small
            continue
        kpts = np.array(orig_dset_dict['keypoints']).reshape(-1,3) 
        
        temp_vec = []
        for connection in KINTREE_TABLE.T:
            flag = True
            dx = kpts[connection[1], 0] - kpts[connection[0], 0]    # direction vector x orig
            dy = kpts[connection[1], 1] - kpts[connection[0], 1]    # direction vector y orig
            
            magnitude = np.sqrt(dx**2 + dy**2)  # irrelevant in original (here), but in augmented relevant

            if flag:
                if magnitude > int(1333/2):
                    altered_kpts = copy.deepcopy(kpts)
                    right_side = np.argwhere((altered_kpts[...,0]-int(1333/2) -1) > 0).squeeze()
                    if right_side.size == 1:
                        right_side = right_side[np.newaxis,...]
                    left_side = list(set(range(0,16)).difference(right_side))

                    min_on_right_side = np.min(altered_kpts[right_side,...,0])
                    offset = 1333 -1 - min_on_right_side
                    altered_kpts[right_side,...,0] = altered_kpts[right_side,...,0] - min_on_right_side
                    if len(left_side) > 0:
                        altered_kpts[left_side,...,0] = altered_kpts[left_side,...,0] + offset
                    
                    if altered_kpts.min() < 0:
                        print("KPT adjustment <0")
                    # altered_kpts[...,0] = altered_kpts[...,0] - offset
                    # altered_kpts[...,0] = altered_kpts[...,0] - np.min(altered_kpts[..., 0])   # kpts x value starts at 0
                    flag = False

            if flag:    # no instance which is on both image sides
                normalized_vector = (dx / magnitude, dy / magnitude)
                temp_vec.append([normalized_vector[0], normalized_vector[1], 0])
            if not flag:
                dx = altered_kpts[connection[1], 0] - altered_kpts[connection[0], 0]    # direction vector x orig
                dy = altered_kpts[connection[1], 1] - altered_kpts[connection[0], 1]    # direction vector y orig

                magnitude = np.sqrt(dx**2 + dy**2)  # irrelevant in original (here), but in augmented relevant
                normalized_vector = (dx / magnitude, dy / magnitude)

                temp_vec.append([normalized_vector[0], normalized_vector[1], 1])

        # look if some kpts (of augm) are marked as 'not labeled' because of out of bounds
        dir_vectors = np.array(temp_vec) 
        jnts_out_of_bounds_connection = np.argwhere(dir_vectors[...,2] == 1).squeeze()
        if jnts_out_of_bounds_connection.size == 1:
            jnts_out_of_bounds_connection = jnts_out_of_bounds_connection[np.newaxis,...]
        dir_vectors = dir_vectors[jnts_out_of_bounds_connection]

        task_kpts = copy.deepcopy(augm_dset_dict['keypoints'])
        if jnts_out_of_bounds_connection.size>0:
            
            kintree_connection = [KINTREE_TABLE.T[problematic_connection] for problematic_connection in jnts_out_of_bounds_connection]
            for connection, vec in zip(kintree_connection, dir_vectors[...,:2]):  
                unlabeled_pos = connection[1]
                pos_at_kintree = (np.argwhere(KINTREE_TABLE.T == unlabeled_pos)).tolist()
                pos_at_kintree = sorted(pos_at_kintree, key=lambda x: (KINTREE_TABLE.T[tuple(x)], -x[1]))  # [[2, 0], [1, 1]]

                for pos in copy.deepcopy(pos_at_kintree):   # could be more than 1 occurance, e.g. pelvis
                    vec = np.array(temp_vec[pos[0]][:2]) 
                    # if pos[1] == 1:
                    start_point = task_kpts[KINTREE_TABLE.T[pos[0], 0]][:2]

                    #start_point = task_kpts[connection[0]][:2]
                    flag = False
                    for length in np.linspace(10, 100, 50):
                        end_point = start_point + vec * length  # if u >= x_min and u <= x_max and v >= y_min and v <= y_max:
                        if end_point[0] > box[0] and end_point[0] <= box[2] and end_point[1] > box[1] and end_point[1] <= box[3]:
                            real_end_point = end_point
                            flag = True
                        else:
                            break
                    if flag:
                        task_kpts[unlabeled_pos] = np.array([real_end_point[0], real_end_point[1], 1]) # KINTREE_TABLE.T[pos[0], 1]
                        pos_at_kintree.remove(pos)
                        #task_kpts[connection[1]] = np.array([real_end_point[0], real_end_point[1], 1]) # KINTREE_TABLE.T[pos[0], 1]
                    else:
                        task_kpts[unlabeled_pos] = np.array([0., 0., -1]) 
                        pos_at_kintree.remove(pos)

            augm_dset_dict['keypoints'] = task_kpts   

        unlabeled_jnt_pos = np.argwhere(augm_dset_dict['keypoints'][...,2] == 0) # which joints are 'not labeld'
        orig_unlabeled_size = unlabeled_jnt_pos.size
        
        while unlabeled_jnt_pos.size>0:
            orig_unlabeled_size = unlabeled_jnt_pos.size
            
            task_kpts = copy.deepcopy(augm_dset_dict['keypoints'])
            for unlabeled_pos in unlabeled_jnt_pos:
                pos_at_kintree = (np.argwhere(KINTREE_TABLE.T == unlabeled_pos)).tolist()
                pos_at_kintree = sorted(pos_at_kintree, key=lambda x: (KINTREE_TABLE.T[tuple(x)], -x[1]))  # [[2, 0], [1, 1]]

                for pos in copy.deepcopy(pos_at_kintree):   # could be more than 1 occurance, e.g. pelvis
                    # if pos[1] != 0:
                    vec = np.array(temp_vec[pos[0]][:2])    # direction vector from starting point (first idx)
                    start_point = task_kpts[KINTREE_TABLE.T[pos[0], 0]][:2]

                    flag = False
                    for length in np.linspace(10, 100, 50):
                        end_point = start_point + vec * length  # if u >= x_min and u <= x_max and v >= y_min and v <= y_max:
                        if end_point[0] > box[0] and end_point[0] <= box[2] and end_point[1] > box[1] and end_point[1] <= box[3]:
                            real_end_point = end_point
                            flag = True
                        else:
                            break
                    if flag:
                        task_kpts[unlabeled_pos] = np.array([real_end_point[0], real_end_point[1], 1]) # KINTREE_TABLE.T[pos[0], 1]
                        pos_at_kintree.remove(pos)
                    else:   # desired jnt has no place
                        task_kpts[unlabeled_pos] = np.array([0., 0., -1]) 
                        pos_at_kintree.remove(pos)
              
            augm_dset_dict['keypoints'] = task_kpts
            unlabeled_jnt_pos = np.argwhere(augm_dset_dict['keypoints'][...,2] == 0) # which joints are 'not labeld'
            if unlabeled_jnt_pos.size == orig_unlabeled_size: 
                break

            # task_kpts = task_kpts.astype(int)
            # img_test = np.zeros(shape=(image.shape[1:]), dtype=np.uint8)
            # for pnts in task_kpts[...,:2]:
            #     img_test[pnts[1], pnts[0]] = 255
            # cv2.imshow('test', img_test)
            # cv2.waitKey(0)

        augm_dset_dict['keypoints'][np.argwhere(augm_dset_dict['keypoints'][...,2] == -1).squeeze()] = np.array([0., 0., 0.]) 

        # old
        # box = augm_dset_dict['bbox']
        # area_box = (box[2] - box[0]) * (box[3] - box[1])
        # if area_box >= threshold_bbox_size:
        augm_dset_dict['bbox'] = augm_dset_dict['bbox'].tolist()
        for idx_segm, segm in enumerate(augm_dset_dict['segmentation']):
            augm_dset_dict['segmentation'][idx_segm] = segm.tolist()
        #obj['keypoints'] = [float(elem) for elem in list(chain(*obj['keypoints']))]
        augm_dset_dict['keypoints'] = [float(elem) for elem in list(chain(*augm_dset_dict['keypoints']))]
        annos.append(augm_dset_dict)
    dataset_dict["height"], dataset_dict["width"] = (800, 1333)  
    dataset_dict["annotations"] = annos
    return dataset_dict


# from detectron2.data import DatasetMapper
# class CustomDatasetMapper(DatasetMapper):
#     def __init__(self, cfg, is_train=True):
#         super().__init__(cfg, is_train)
#         self.cfg = cfg
#         self.keypoint_hflip_indices = cfg.DATASETS.KEYPOINT_HFLIP_INDICES

#     def __call__(self, dataset_dict):
#         #dataset_dict = super().__call__(dataset_dict)
#         dataset_dict = custom_mapper(dataset_dict, self.cfg, self.keypoint_hflip_indices, self.is_train)
#         return dataset_dict

# from detectron2.data import DatasetCatalog, MetadataCatalog
# def get_dataset_dicts(path: str):
#     files = glob.glob(path+"/*")
#     dataset_dicts = []
#     for json_file in files:
#         with open(json_file) as f:
#             record = json.load(f)
#         dataset_dicts.append(record)
    
#     return dataset_dicts

# from detectron2.engine import DefaultTrainer
# from detectron2.data import build_detection_test_loader, build_detection_train_loader
# from detectron2.evaluation import DatasetEvaluators, COCOEvaluator, inference_on_dataset
# from detectron2.utils.comm import reduce_dict
# import datetime
# import logging


# class CustomTrainer(DefaultTrainer):
#     @classmethod
#     def build_train_loader(cls, cfg):
#         # Custom mapper for training
#         def train_mapper(dataset_dict):
#             return custom_mapper(dataset_dict, cfg, keypoint_hflip_indices=keypoint_hflip_indices, is_train=True)
        
#         return build_detection_train_loader(cfg, mapper=train_mapper)

#     @classmethod
#     def build_test_loader(cls, cfg, dataset, keypoint_hflip_indices=None, is_train=True):
#         return build_detection_test_loader(dataset, cfg, mapper = lambda dataset, cfg, keypoint_hflip_indices, is_train: custom_mapper(dataset, cfg, keypoint_hflip_indices=None, is_train=True))

#     @classmethod
#     def build_evaluator(cls, cfg):
#         return COCOEvaluator(cfg, distributed=False, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))



    
if __name__ == "__main__":
    from registerDatasetCatalog import register_data
    from detectron2.data import DatasetCatalog, MetadataCatalog
    register_data(input_path= "/workspace/data/dataset")
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer
    from detectron2.utils.visualizer import ColorMode
    
    from model_config import cfg, KEYPOINT_NAMES, KEYPOINT_FLIP_MAP

    # keypoint_hflip_indices = np.arange(len(KEYPOINT_NAMES))
    # for src, dst in KEYPOINT_FLIP_MAP:
    #     src_idx = KEYPOINT_NAMES.index(src)
    #     dst_idx = KEYPOINT_NAMES.index(dst)
    #     keypoint_hflip_indices[src_idx] = dst_idx
    #     keypoint_hflip_indices[dst_idx] = src_idx

    dataset_dicts_ = DatasetCatalog.get("carla/train")
    import random
    for d in random.sample(dataset_dicts_, 5):
        d_ = copy.deepcopy(d)
        d_ = custom_mapper(d_, cfg)
        img = np.uint8(255*d_["image"].numpy().transpose(1, 2, 0))
        img = cv2.applyColorMap(img[...,0],cv2.COLORMAP_BONE)
        #cv2.imshow("img",img)
        v = Visualizer(img, # draw_circle
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )   
        vis = v.draw_dataset_dict(d_)
        cv2.imshow('test', vis.get_image())
        cv2.waitKey(0)
        print("d")    
    trainer = CustomTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    # print()    
