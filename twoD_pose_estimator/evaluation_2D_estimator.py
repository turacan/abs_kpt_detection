from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

from custom_mapper import custom_mapper
from data_generation.save_dataset_detectron_format import calculate_projection_matrix

import cv2
import json
import sys
import copy
import numpy as np
import matplotlib
matplotlib.use('TKAgg')

from matplotlib import pyplot as plt

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch

import numpy as np
from scipy.optimize import linear_sum_assignment
from detectron2.structures import Boxes
import matplotlib.pyplot as plt


from matplotlib.legend_handler import HandlerBase
import matplotlib.lines as mlines
class MarkerHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                xdescent, ydescent, width, height, fontsize,
                trans):
        return [plt.Line2D([width/2], [height/2.], linestyle='None',
                    marker=orig_handle[0], color=orig_handle[1], markersize=orig_handle[2])]
    
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

def match_instances(pred_boxes, gt_boxes, iou_threshold=0.5):
    # Convert the predicted and ground truth bounding boxes to numpy arrays
    pred_boxes_np = pred_boxes.tensor.cpu().numpy()
    gt_boxes_np = gt_boxes.tensor.cpu().numpy()

    # Compute the IoU matrix between predicted and ground truth bounding boxes
    iou_matrix = compute_iou_matrix(pred_boxes_np, gt_boxes_np)

    # Apply the Hungarian algorithm to find the optimal instance matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    # Create a mask indicating the matched instances
    match_mask = iou_matrix[row_ind, col_ind] >= iou_threshold

    # Filter out unmatched instances
    matched_pred_idxs = row_ind[match_mask]
    matched_gt_idxs = col_ind[match_mask]

    matched_pred_boxes = pred_boxes[matched_pred_idxs]
    matched_gt_boxes = gt_boxes[matched_gt_idxs]

    return matched_pred_idxs, matched_gt_idxs, matched_pred_boxes, matched_gt_boxes

def compute_iou_matrix(boxes1, boxes2):
    # Compute the IoU matrix between two sets of bounding boxes
    iou_matrix = np.zeros((len(boxes1), len(boxes2)))
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou = compute_iou(box1, box2)
            iou_matrix[i, j] = iou
    return iou_matrix


def compute_iou(box1, box2):
    # Compute the IoU (Intersection over Union) between two bounding boxes
    intersection_width = min(box1[2], box2[2]) - max(box1[0], box2[0])
    intersection_height = min(box1[3], box2[3]) - max(box1[1], box2[1])
    
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0
    
    intersection_area = intersection_width * intersection_height
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou

# def compute_iou(box1, box2):
#     # Compute the IoU (Intersection over Union) between two bounding boxes
#     intersection = np.maximum(0, np.minimum(box1[2:], box2[2:]) - np.maximum(box1[:2], box2[:2]))
#     intersection_area = np.prod(intersection)
#     box1_area = np.prod(box1[2:] - box1[:2])
#     box2_area = np.prod(box2[2:] - box2[:2])
#     iou = intersection_area / (box1_area + box2_area - intersection_area)
#     return iou

# scale-invariant
def calculate_error_pck(pred: np.ndarray, gt: np.ndarray, gt_box_sizes: list, thresholds: np.ndarray = np.linspace(0, 0.2, num=11)) -> float:
    # Calculate per joint error (2D Euclidean distance)
    thresholds = np.arange(0,0.5,0.005)
    joint_errors = np.linalg.norm(pred - gt, axis=-1)

    # Compute PCK at each threshold and store in a dictionary
    pck_at_thresholds = {}
    for alpha in thresholds:
        pck_threshold = alpha * gt_box_sizes[:, None]
        pck = np.nanmean(joint_errors <= pck_threshold)
        pck_at_thresholds[alpha] = pck

    return pck_at_thresholds


# scale-invariant
def calculate_error_scale_inv_mpjpe(pred: np.ndarray, gt: np.ndarray, gt_box_sizes: list) -> float:
    # Calculate per joint error (2D Euclidean distance), and normalize by bounding box size
    joint_errors = np.linalg.norm(pred - gt, axis=-1) / gt_box_sizes[:, np.newaxis]
    if np.argwhere(joint_errors==np.inf).size>0:
        print("ERROR in mapper")

    return joint_errors


def gather_eval_input(gt: dict, pred: dict, iou_thresh: float = 0.5):
    pred_boxes = pred["instances"].get_fields()['pred_boxes']
    gt_boxes = gt["instances"].get_fields()['gt_boxes']

    pred_idxs, gt_idxs, *_ = match_instances(pred_boxes, gt_boxes, iou_threshold=iou_thresh)

    pred_boxes_np = pred_boxes.tensor.cpu()[pred_idxs].numpy()
    gt_boxes_np = gt_boxes.tensor.cpu()[gt_idxs].numpy()

    # diagonal
    #pred_box_sizes = np.sqrt((pred_boxes_np[:, 2] - pred_boxes_np[:, 0]) * (pred_boxes_np[:, 3] - pred_boxes_np[:, 1]))
    gt_box_sizes = np.sqrt((gt_boxes_np[:, 2] - gt_boxes_np[:, 0]) * (gt_boxes_np[:, 3] - gt_boxes_np[:, 1]))

    pred_kpts = pred["instances"].get_fields()['pred_keypoints'][pred_idxs].numpy()[...,:2]# key pred_keypoints already a tensor
    #pred_kpts, pred_flag = pred_kpts[...,:2], pred_kpts[...,-1]
 
    gt_kpts = gt["instances"].get_fields()['gt_keypoints'].tensor[gt_idxs].numpy()
    gt_kpts, visibility = gt_kpts[...,:2], np.where(gt_kpts[...,-1]>=1, 1, 0)[...,np.newaxis].astype(np.uint8)

    if np.argwhere(visibility==0).size>0:
        print()

    gt_kpts = np.where(visibility >= 1, gt_kpts, np.nan)
    pred_kpts = np.where(visibility >= 1, pred_kpts, np.nan)

    # pck = calculate_error_pck(pred_kpts, gt_kpts, gt_box_sizes) 
    # mpjpe = calculate_error_scale_inv_mpjpe(pred_kpts, gt_kpts, gt_box_sizes) 
    #MPJPE_batch = np.linalg.norm(pred_kpts[...,:2] - gt_kpts[...,:2], axis=-1)
    return pred_kpts, gt_kpts, gt_box_sizes, pred_boxes_np, gt_boxes_np
     

import numpy as np
from scipy.stats import mode
from skimage.measure import label, regionprops
def get_surrounding_pixels(im, v, u, offset):
    # Get the dimensions of the image
    height, width = im.shape

    # Compute the range of indices to consider in the v (vertical/row) and u (horizontal/column) dimensions
    v_indices = range(max(0, v - offset), min(height, v + offset + 1))
    u_indices = range(max(0, u - offset), min(width, u + offset + 1))

    surrounding_pixels = [(vi, ui) for vi in v_indices for ui in u_indices if (vi, ui) != (v, u)]
    return surrounding_pixels

def bbox_points_in_limits(im, bbox, keypoints, max_offset_pct, default_value):
    # Calculate statistics for the bbox
    im_range = im[...,0] * 250
    im_angle_suface = im[...,1]
    
    instance_image = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    range, angle_suface = (instance_image[...,0] * 250).flatten(), instance_image[...,1].flatten()
    mode_value_range, count = mode(range)
    mode_value_angle, count = mode(angle_suface)

    print(f"mode={mode_value_range}, std={np.std(range)}, accepted_range={mode_value_range-np.std(range)}, {mode_value_range+np.std(range)}")
    # test = (copy.deepcopy(im_range)).astype(np.uint8)
    # test = cv2.rectangle(test, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 150, 1)
    # cv2.imshow('VERTICAL', test)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # We are using 1.5 * IQR rule to detect outliers here
    # q75, q25 = np.percentile(range_values, [75 ,25])
    # iqr = q75 - q25
    # mask = (range_values >= q25 - 1.5 * iqr) & (range_values <= q75 + 1.5 * iqr)
    x = 0.2  # change x to any value that suits your need
    #accepted_range = [mode_value_range * (1 - x), mode_value_range * (1 + x)]
    accepted_range = [(mode_value_range-np.std(range)).squeeze(), (mode_value_range+np.std(range)).squeeze()]

    # Calculate the maximum offset in pixels
    bbox_diagonal = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    max_offset = int(max_offset_pct * bbox_diagonal)

    # Adjust the keypoints
    result_range = []
    for i, keypoint in enumerate(keypoints):
        u, v = keypoint.astype(int)
        if not accepted_range[0] <= im_range[v, u] <= accepted_range[1]:
            offset = 1
            while offset <= max_offset:
                surrounding_pixels = get_surrounding_pixels(im_range, v, u, offset)
                for vp, up in surrounding_pixels:
                    if bbox[0] <= vp <= bbox[2] and bbox[1] <= up <= bbox[3]:
                        if accepted_range[0] <= im_range[vp, up] <= accepted_range[1]:
                            result_range.append(im_range[vp, up])
                            #keypoints[i] = [vp, up]
                            break
                else:
                    offset += 1
                    continue
                break
            else:
                result_range.append(mode_value_range)
                #im_range[v, u] = default_value if default_value is not None else mode_value
        else:
            result_range.append(im_range[v, u])

    return result_range


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

# unfreeze cfgNode
cfg.defrost()

# Change the config
cfg.MODEL.WEIGHTS = "/workspace/data/model_output/workspace_pre__augs_0.0001_2023-12-20_14-26-53/model_final.pth"
#/workspace/data/model_output/workspace_pre__augs_2e-05_2023-06-26_22-08-55/model_0071999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
cfg.MODEL.DEVICE = "cpu"    # cuda
#cfg.INPUT.AUG_ZOOM_TRAIN = [1/4]    # 1/4: 256 channels, 1/8: 128 channels
cfg.freeze()

# Create a predictor
predictor = DefaultPredictor(cfg)

# Get the metadata
from registerDatasetCatalog import register_data
register_data(input_path= "/workspace/data/dataset")

IOU_THRESH = 0.75
visu_flag = False
splittype = "test"
val_dicts = DatasetCatalog.get(f"carla/{splittype}")
val_metadata = MetadataCatalog.get(f"carla/{splittype}")

MPJPE_list = []
pred_kpt_list = []
gt_kpt_list = []
box_size_list = []

from tqdm import tqdm
for d in tqdm(val_dicts[:]):
    #im = cv2.imread(d["file_name"], cv2.IMREAD_UNCHANGED)

    d_ = copy.deepcopy(d)
    d_ = custom_mapper(d_, cfg, is_train=True)
    if len(d_['annotations']) == 0: # skip frames if no instance is present
        continue
    im = d_["image"].numpy().transpose(1, 2, 0)
    
    # make a prediction
    outputs = predictor(im[...,::-1]) # in __call__ channels not be changed, because input format="RGB", but cv2.unchanged doesn't convert to BGR

    # make custom evaluation
    pred_kpts, gt_kpts, gt_box_sizes, pred_box, gt_box = gather_eval_input(gt=d_, pred=outputs, iou_thresh=IOU_THRESH)
    pred_kpt_list.extend(pred_kpts)
    gt_kpt_list.extend(gt_kpts)
    box_size_list.extend(gt_box_sizes)

    # Visualize the predicted bounding boxes
    if visu_flag:
        # Keep scale=1 in order to associate the predicted coords with the image below.
        # If the scale is different the predicted coords won't match the bbox shown on
        # the image below.
        outputs_altered = copy.deepcopy(outputs)

        # make all joints visible and visualized
        for idx, elem in enumerate(outputs_altered["instances"]._fields['pred_keypoints']):
            elem[:,2] = 2
            outputs_altered["instances"].get_fields()['pred_keypoints'][idx]= elem #elem.type(torch.int16) #elem[:,2] = 2

        # Visualization
        im = np.uint8(255*im)
        im = cv2.applyColorMap(im[...,0],cv2.COLORMAP_MAGMA) # cv2.COLORMAP_HSV
        visual = Visualizer(im[:, :, ::-1], scale=1, metadata=val_metadata) 
        #visual = Visualizer(im[:, :, ::-1], scale=1) 
        v = copy.deepcopy(visual).draw_instance_predictions(outputs_altered["instances"].to("cpu"))

        v_gt = copy.deepcopy(visual).draw_dataset_dict(d_)
        # for kpt in d_['annotations']:
        #     print(len(kpt['keypoints']))
        # concatenate image Horizontally
        Hori = np.concatenate((v.get_image(), v_gt.get_image()), axis=1)
        
        # concatenate image Vertically
        Verti = np.concatenate((v.get_image(), v_gt.get_image()), axis=0)
        Verti = cv2.resize(Verti, (2048, 1024), interpolation = cv2.INTER_AREA)

        cv2.namedWindow('VERTICAL', cv2.WINDOW_AUTOSIZE)
        imS = cv2.resize(Verti, (1500, 750))   
        #cv2.resizeWindow("VERTICAL", 1000, 1000)
        # Naming a window
        cv2.imshow('VERTICAL', imS)
        cv2.waitKey(0)

if visu_flag:
    cv2.destroyAllWindows()

pred_kpt_np = np.array(pred_kpt_list, dtype=np.float16)
gt_kpt_np = np.array(gt_kpt_list, dtype=np.float16)
box_size_np = np.array(box_size_list, dtype=np.float16)

# Calculate mean per joint position error (MPJPE)
MPJPE_list = calculate_error_scale_inv_mpjpe(pred_kpt_np, gt_kpt_np, box_size_np)
mpjpe = np.nanmean(MPJPE_list, axis=0)
MPJPE_skalar = np.nanmean(mpjpe)

# Calculate Percentage of correct keypoints
pck = calculate_error_pck(pred_kpt_np, gt_kpt_np, box_size_np)

'''
The error is calculated as the pixel distance divided by the length of the BBOX diagonal, 
resulting in a scale-invariant measurement. 
This pixel error is relative to the BBOX diagonal length, providing a standardized metric 
for evaluating the accuracy of the object detection or localization algorithm
'''
import seaborn as sns
fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8))

# Major ticks
major_ticks_x = np.arange(0, 0.5+0.05, 0.05)
major_ticks_y = np.arange(0, 101, 10)
ax.set_xticks(major_ticks_x)
ax.set_yticks(major_ticks_y)
ax.grid()

# Set the limits of the plot
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 100)

x, y = pck.keys(), np.array(list(pck.values()))*100
# Set tick sizes
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

sns.lineplot(x=x, y=y, linewidth=5)
ax.lines[0].set_linestyle("--")
ax.set_title(f'PCK total, IOU\u2265{IOU_THRESH}', fontsize=25)
ax.set_xlabel('Pixel Error Threshold', fontsize=19)
ax.set_ylabel('Detection rate, %', fontsize=19)


import pandas as pd
joint_list={0:'Pelvis', 
            1:'R_Hip', 2:'R_Knee', 3:'R_Ankle', 
            4:'L_Hip', 5:'L_Knee', 6:'L_Ankle', 
            7:'Torso', 8:'Neck', 9:'Head', 
            10:'R_Shoulder', 11:'R_Elbow', 12:'R_Wrist', 
            13:'L_Shoulder', 14:'L_Elbow', 15:'L_Wrist'
}

name=[]
for k, v in joint_list.items():
    name.append(f'{k}: {v}')

df_MPJPE_boxplot = pd.DataFrame(
    MPJPE_list,
    columns=name
)

df_MPJPE = pd.DataFrame({
    'Joint': list(range(16)),
    'Error': mpjpe
})

fig2 = plt.figure()
fig2.set_size_inches(18.5, 10.5)
ax = fig2.add_subplot()

#increase font size of all elements
sns.boxplot(data=df_MPJPE_boxplot, fliersize=2) 
ax.set_title(f"Per Joint Position Error: Scale-Invariant Pixel Error Relative to BBOX Diagonal, IOU\u2265{IOU_THRESH}", fontsize=20)
plt.xticks(fontsize=12, rotation=45)
ax.set_ylabel('Error', fontsize=16)
plt.yticks(fontsize=18)
plt.xticks(fontsize=13.5)
ax.set_ylim(0, 1)
# Enable grid with horizontal lines at each 0.1 interval
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.yticks([i * 0.1 for i in range(11)])

ax.scatter(x=range(0,16), y=df_MPJPE['Error'], marker="x", linewidths=2, facecolor='red')
ax.axhline(y=MPJPE_skalar, linewidth=2, color='b')


list_marker    = ["_", "x"]
list_marker_color  = ["b", "r"]
list_markersize = [12] * 2
list_label    = ['Mean MPJPE','MPJPE']

ax.legend(list(zip(list_marker, list_marker_color, list_markersize)), list_label, fontsize=12,
        handler_map={tuple:MarkerHandler()}, loc="upper left") 

ax.text(0.99, 0.99, f'Mean MPJPE={MPJPE_skalar:.3f}',
    verticalalignment='top', horizontalalignment='right',
    transform=ax.transAxes, fontsize=16, color='b')

print(len(gt_kpt_list))
plt.show()

# IOU=0.5, total of 1107 person objects are evaluated 

# IOU=0.75, total of 611 person objects are evaluated 



    # # # # make keypoint depth logic
    # # # max_offset_pct = 0.1
    # # # default_value = None
    # # # depth_values_list = []
    # # # for bbox, keypoints in zip(gt_box.astype(int), pred_kpt_list):
    # # #     depth_values = bbox_points_in_limits(im[...,[0,2]], bbox, keypoints, max_offset_pct, default_value)
    # # #     depth_values_list.append(depth_values)

    # # # depth_values_arr = np.array(depth_values_list)

    # # # convert u,v-coord in phi theta
    # # K, _ = calculate_projection_matrix(height=im.shape[0], width=im.shape[1], fov_degrees=45)
    # # u,v = gt_kpts[0,0].astype(int)
    # # # temp = gt_kpts[0].astype(int)
    # # # u = temp[:, 0]
    # # # v = temp[:, 1]
    # # depth = 1#im[v,u,0] * 250
    # # #depth = depth_values_arr[0]
    # # temp_kpt = np.hstack((copy.deepcopy(gt_kpts[0]).squeeze(), np.ones((16,1))))
    # # phi_theta = np.matmul(np.linalg.inv(K) , temp_kpt.T)
    # # phi, theta = phi_theta[0], phi_theta[1]
    # # x = depth * np.sin(np.pi/2-theta) * np.cos(phi) 
    # # y = depth * np.sin(np.pi/2-theta) * np.sin(phi)
    # # z = depth * np.cos(np.pi/2-theta)

    # # jnts_skeleton_gt = np.stack([x,y,z], axis=-1)

    # # temp_kpt = np.hstack((copy.deepcopy(pred_kpts[0]).squeeze(), np.ones((16,1))))
    # # u,v = pred_kpts[0,0].astype(int)
    # # # temp = pred_kpts[0].astype(int)
    # # # u = temp[:, 0]
    # # # v = temp[:, 1]
    # # depth = 1#im[v,u,0] * 250
    # # phi_theta = np.matmul(np.linalg.inv(K) , temp_kpt.T)
    # # phi, theta = phi_theta[0], phi_theta[1]
    # # x = depth * np.sin(np.pi/2-theta) * np.cos(phi) 
    # # y = depth * np.sin(np.pi/2-theta) * np.sin(phi)
    # # z = depth * np.cos(np.pi/2-theta)

    # # jnts_skeleton_pred= np.stack([x,y,z], axis=-1)
    # # np.rad2deg(phi)
    # # import open3d as o3d
    # # from utils.spherical import o3d_draw_skeleton

    # # # fig = plt.figure(figsize=(12, 12))
    # # # ax = fig.add_subplot(projection='3d')
    # # # ax.scatter(jnts_skeleton[:,0], jnts_skeleton[:,1], jnts_skeleton[:,2])
    # # # plt.show()


    # # line_set_gt, joints_gt = o3d_draw_skeleton(jnts_skeleton_gt, kintree_table)
    # # line_set_pred, joints_pred = o3d_draw_skeleton(jnts_skeleton_pred, kintree_table, color_set="CMY")
    # # coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # # #o3d.visualization.draw_geometries([line_set_gt, joints_gt, line_set_pred, joints_pred])
    # # #o3d.visualization.draw_geometries([line_set_gt, joints_gt])#, coord_mesh])
