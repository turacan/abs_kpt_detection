import json
import sys
import numpy as np
import copy
import cv2
import open3d as o3d

import torch
from torch import nn

from detectron2.data import (
    DatasetCatalog,
    build_detection_train_loader
)
from custom_mapper import custom_mapper
from tqdm import tqdm
from save_dataset_detectron_format import calculate_projection_matrix
from start_training import setup_parameters, setup_config
from utils.spherical import o3d_draw_skeleton


kintree_table = np.array([ 
    [0, 1],     # Pelvis -> R_Hip
    [1, 2],     # R_Hip -> R_Knee
    [2, 3],     # R_Knee -> R_Ankle
    [0, 4],     # Pelvis -> L_Hip
    [4, 5],     # L_Hip -> L_Knee
    [5, 6],     # L_Knee -> L_Ankle
    [0, 7],     # Pelvis -> Torso
    [7, 8],     # Torso -> Neck
    [8, 9],     # Neck -> Head
    [7, 10],    # Torso -> R_Shoulder
    [10, 11],   # R_Shoulder -> R_Elbow
    [11, 12],   # R_Elbow -> R_Wrist
    [7, 13],    # Torso -> L_Shoulder
    [13, 14],   # L_Shoulder -> L_Elbow
    [14, 15]    # L_Elbow -> L_Wrist
]).T


# def get_surrounding_pixels(im, v, u, offset=0.1):
#     # Get the dimensions of the image
#     height, width = im.shape

#     # Compute the range of indices to consider in the v (vertical/row) and u (horizontal/column) dimensions
#     v_indices = range(max(0, v - offset), min(height, v + offset + 1))
#     u_indices = range(max(0, u - offset), min(width, u + offset + 1))

#     surrounding_pixels = [(vi, ui) for vi in v_indices for ui in u_indices if (vi, ui) != (v, u)]
#     return surrounding_pixels

def get_surrounding_pixels(image, v, u, offset):
    surrounding_pixels = []
    for i in range(-offset, offset + 1):
        for j in range(-offset, offset + 1):
            if 0 <= v + i < image.shape[0] and 0 <= u + j < image.shape[1]:
                surrounding_pixels.append((v + i, u + j))
    return surrounding_pixels


class DepthEstimator(nn.Module):
    def __init__(self):
        super(DepthEstimator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 64),   # Input layer
            nn.ReLU(),   # Activation function
            nn.Linear(64, 64),   # Hidden layer
            nn.ReLU(),   # Activation function
            nn.Linear(64, 16),   # Output layer
        )

    def forward(self, x):
        return self.layers(x)

def build_train_loader(cfg):
    original_dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
    return build_detection_train_loader(cfg, dataset=original_dataset, mapper=lambda d: custom_mapper(d, cfg))
   
if __name__ == "__main__":    
    try:
        with open("config/config.json", 'r') as f:
            cfg = json.load(f)
    except Exception as ex:
        sys.exit("provided cfg file path not valid")

    # create parameter sweeping list
    params_list = setup_parameters(cfg=cfg)
    # Setup detectron2 training config
    cfg = setup_config(cfg, params_list[0])

    # unfreeze cfgNode
    cfg.defrost()

    # Change the config
    cfg.MODEL.WEIGHTS = "/workspace/data/model_output/workspace_pre__augs_2e-05_2023-06-26_22-08-55/model_0071999.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.MODEL.DEVICE = "CUDA"
    cfg.INPUT.AUG_ZOOM_TRAIN = [1/4]    # 1/4: 256 channels, 1/8: 128 channels
    cfg.freeze()


    # Get the metadata
    from registerDatasetCatalog import register_data
    register_data(input_path= "/workspace/data/dataset")


    K, _ = calculate_projection_matrix(height=800, width=1333, fov_degrees=45)
    
    data_loader = build_train_loader(cfg)
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_idx, batch_inputs in tqdm(enumerate(data_loader)):
            iterator = iter(batch_inputs)
            try:
                for dset_dict in iterator:
                    kpts_arr = dset_dict['instances'].get_fields().get('gt_keypoints', None)
                    box_arr = dset_dict['instances'].get_fields().get('gt_boxes', None)
                    polygon_raw = dset_dict['instances'].get_fields().get('gt_masks', None)
                    if kpts_arr == None or polygon_raw == None:
                        continue
                    
                    box_arr = box_arr.tensor.cpu().numpy()
                    polygon_raw = polygon_raw.polygons
                    polygon_list = []
                    for idx_polygon_arr, polygon in enumerate(polygon_raw):
                        polygon_list.append([sublist.astype(int).tolist() for sublist in polygon])
                        #polygon_list.append(np.unique(np.array([item for sublist in polygon for item in sublist], dtype=int).reshape(-1,2), axis=0))
                    
                    kpts_arr = kpts_arr.tensor.cpu().numpy()
                    kpts_arr = np.where(kpts_arr[...,-1][...,np.newaxis]!=0, kpts_arr, np.nan) # make kpts line nan if not visible 

                    # Add an extra dimension to kpts_arr and K
                    temp_kpt = np.concatenate((kpts_arr[...,:2], np.ones((kpts_arr.shape[0], kpts_arr.shape[1], 1))), axis=-1)
                    K_inv = np.linalg.inv(K)[np.newaxis, np.newaxis, :, :]

                    # Transpose temp_kpt to match the dimensions for np.matmul
                    temp_kpt = temp_kpt[np.newaxis,...].transpose((0, 1, 3, 2))

                    # Perform matrix multiplication using broadcasting
                    phi_theta = np.matmul(K_inv, temp_kpt)[..., :2, :]

                    # Transpose back to original shape
                    phi_theta = phi_theta.transpose((0, 1, 3, 2)).squeeze()

                    for instance_idx, polygon in enumerate(polygon_list):
                       
                        # polygon_x = polygon[:,0].min(), polygon[:,0].max()
                        # polygon_y = polygon[:,1].min(), polygon[:,1].max()
                        
                        depth_image = 250 * dset_dict['image'].cpu().numpy().transpose(1, 2, 0)[...,0]

                         # Create an empty mask to match the size of your image
                        mask = np.zeros_like(depth_image, dtype=np.uint8)  # Adjust the size as needed

                        # Fill the mask with the contour
                        for contour in polygon:
                            mask = cv2.fillPoly(mask, [np.array(contour).reshape(-1, 2)], 1)

                        # Get the points inside the contour
                        inside_points = np.where(mask)

                        polygon_depth_values = depth_image[inside_points[0], inside_points[1]]
                        polygon_depth_values = polygon_depth_values[polygon_depth_values>0]
                        median_depth = np.median(polygon_depth_values)
                        print(f"median_depth={median_depth}")
                        accepted_range = [median_depth - 1.5, median_depth + 1.5]
                        
                        kpts_loc = kpts_arr[instance_idx][...,:2].astype(np.uint16)
                        bbox = box_arr[instance_idx]
                        test = (copy.deepcopy(depth_image)).astype(np.uint8)
                        test[kpts_loc[:, 1], kpts_loc[:, 0]] = 255
                        test = cv2.rectangle(test, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 150, 1)
                        cv2.imshow('VERTICAL', test)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                        # Calculate the maximum offset in pixels
                        bbox_diagonal = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
                        max_offset = int(0.05 * bbox_diagonal)

                        # Get valid depth
                        result_range = []
                        for i, keypoint in enumerate(kpts_loc):
                            u, v = keypoint.astype(int)
                            if not accepted_range[0] <= depth_image[v, u] <= accepted_range[1]:
                                offset = 1
                                while offset <= max_offset:
                                    surrounding_pixels = get_surrounding_pixels(depth_image, v, u, offset)
                                    valid_pixels = [p for p in surrounding_pixels if accepted_range[0] <= depth_image[p[0], p[1]] <= accepted_range[1]]
                                    if valid_pixels:
                                        distances = [np.abs(depth_image[p[0], p[1]] - median_depth) for p in valid_pixels]
                                        nearest_pixel = valid_pixels[np.argmin(distances)]
                                        result_range.append(depth_image[nearest_pixel[0], nearest_pixel[1]])
                                        break
                                    offset += 1
                                else:
                                    result_range.append(median_depth)
                            else:
                                result_range.append(depth_image[v, u])

                        depth = np.array(result_range)

                        phi, theta = phi_theta[instance_idx, :, 0], phi_theta[instance_idx, :,1]
                        x = depth * np.sin(np.pi/2-theta) * np.cos(phi) 
                        y = depth * np.sin(np.pi/2-theta) * np.sin(phi)
                        z = depth * np.cos(np.pi/2-theta)

                        jnts_skeleton_gt = np.stack([x,y,z], axis=-1)
                        line_set_gt, joints_gt = o3d_draw_skeleton(jnts_skeleton_gt, kintree_table)
                        o3d.visualization.draw_geometries([line_set_gt, joints_gt])
                        print(result_range)

                    # phi_theta = []
                    # for kpts in kpts_arr:
                    #     temp_kpt = np.concatenate((copy.deepcopy(kpts[...,:2]).T, np.ones((1, kpts.shape[0]))), axis=0)  
                    #     phi_theta.append(np.matmul(np.linalg.inv(K), temp_kpt)[:2])
            except Exception as ex:
                print(ex)
                
                