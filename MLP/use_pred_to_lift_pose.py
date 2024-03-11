import json
import sys
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
sys.path.append("/workspace/repos/")
from custom_mapper import custom_mapper
from save_dataset_detectron_format import calculate_projection_matrix, MAX_SENSOR_RANGE, KINTREE_TABLE
from utils.spherical import o3d_draw_skeleton
from utils.fc_plots_eval import plot_mpjpe_boxplot
import open3d as o3d

from MLP.root_relative_distance_network import RelativeDistancePredictionNetwork, RootDistancePredictionNetwork, CorrectionNetwork, IntegratedPoseEstimationModel

try:
    with open("config/config.json", 'r') as f:
        cfg = json.load(f)
except Exception as ex:
    sys.exit("provided cfg file path not valid")

from start_training import setup_parameters, setup_config


DEVICE = "cpu"  # cuda:0
input_w_depth = True

inference = True
visu_flag_3d= False  # will only get to conditional if inference
visu_flag_2d = False # will only get to conditional if inference

save_dset = False
IOU_THRESH=0.5#0.5

splittype = "test"
save_path= "/workspace/data/dataset/" + f"{splittype}/" + "pred_2d" 


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
cfg.MODEL.DEVICE = DEVICE    # cuda
#cfg.INPUT.AUG_ZOOM_TRAIN = [1/4]    # 1/4: 256 channels, 1/8: 128 channels
cfg.freeze()

# Create a predictor
predictor = DefaultPredictor(cfg)

# Get the metadata
from registerDatasetCatalog import register_data
register_data(input_path= "/workspace/data/dataset")


val_dicts = DatasetCatalog.get(f"carla/{splittype}")
val_metadata = MetadataCatalog.get(f"carla/{splittype}")

# define layer sizes, per instance not batch
if inference:
    input_size = 3 * 16
    if input_w_depth:
        input_size +=1

    output_size_relative = 16   # relative distance of the joints to the root joint (pelvis, first index)
    hidden_size_relative  = 1024    # 1024

    output_size_root = 1    # scalar which is the distance to the root joint
    hidden_size_root  = 256         # 256

    input_size_correction = 3 * 16
    output_size_correction = 3 * 16 
    hidden_size_correction = 1024   # 2048

    relative_network = RelativeDistancePredictionNetwork(input_size=input_size, 
                                            hidden_size=hidden_size_relative, 
                                            output_size=output_size_relative)
    relative_network.to(DEVICE)

    root_network = RootDistancePredictionNetwork(input_size=input_size, 
                                            hidden_size=hidden_size_root, 
                                            output_size=output_size_root)
    root_network.to(DEVICE)

    correction_network = CorrectionNetwork(input_size=input_size_correction, 
                                            hidden_size=output_size_correction, 
                                            output_size=output_size_correction)
    correction_network.to(DEVICE)

    absolute_3d_joint_estimation_network = IntegratedPoseEstimationModel(relative_network, root_network, correction_network)
    absolute_3d_joint_estimation_network.load_state_dict(torch.load('/workspace/repos/MLP/weights/model_weights-240111_122503-finetuned_wd.pth', map_location=DEVICE))
    absolute_3d_joint_estimation_network.to(DEVICE)
    # '/workspace/repos/MLP/weights/model_weights-231206_152641.pth'
    absolute_3d_joint_estimation_network.eval()

from utils.fc_instance_matching import match_instances

if inference:
    PJPE_list=[]
    count_outliers=0
    num_total=0
for d in tqdm(val_dicts[:]):
    mapped_anno = custom_mapper(d, cfg, is_train=(False, 0.125)) # current fov needs to be changed accordingly
    im = mapped_anno["image"].numpy().transpose(1, 2, 0)
    
    # make a prediction
    outputs = predictor(im[...,::-1]) # in __call__ channels not be changed, because input format="RGB", but cv2.unchanged doesn't convert to BGR
    num_objs = len(outputs['instances'].get_fields()["pred_classes"])

    pred_boxes = outputs["instances"].get_fields()['pred_boxes']
    gt_boxes = mapped_anno["instances"].get_fields()['gt_boxes']

    pred_idxs, gt_idxs, *_ = match_instances(pred_boxes, gt_boxes, iou_threshold=IOU_THRESH)

    pred_boxes = outputs['instances'].get_fields()["pred_boxes"].tensor
    pred_masks = outputs['instances'].get_fields()["pred_masks"]
    pred_keypoints = outputs['instances'].get_fields()["pred_keypoints"][..., :-1]

    if save_dset:
        file_id = mapped_anno['image_id']
        gt_idx_list = []
        pred_kpts_2d_list = []
        gt_kpts_2d_list = []
        gt_kpts_3d_list = []

    for pred_idx, gt_idx in zip(pred_idxs, gt_idxs):
        if save_dset:
            # type str cooresponds to image name name .exr and json annotation file
            pred_kpts_2d = pred_keypoints[pred_idx].numpy().astype(np.float32)
            gt_kpts_2d = mapped_anno['annotations'][gt_idx]['keypoints'][:, :-1].astype(np.float32)
            gt_kpts_3d = np.array(mapped_anno['annotations'][gt_idx]['keypoints_3d'], dtype=np.float32)[:, :-1]
            
            # ADD TO LIST
            gt_idx_list.append(gt_idx)
            pred_kpts_2d_list.append(pred_kpts_2d)
            gt_kpts_2d_list.append(gt_kpts_2d)
            gt_kpts_3d_list.append(gt_kpts_3d)

        elif inference:
            K, _ = calculate_projection_matrix(height=im.shape[0], width=im.shape[1], fov_degrees=22.5)
            temp_kpt = np.concatenate((pred_keypoints[pred_idx], np.ones((16,1))), axis=1)
            phi_theta = np.matmul(np.linalg.inv(K) , temp_kpt.T)[:-1].T
            phi = phi_theta[:, 0]
            theta = phi_theta[:, 1]

            ux = np.sin(np.pi/2-theta) * np.cos(phi) 
            uy = np.sin(np.pi/2-theta) * np.sin(phi)
            uz = np.cos(np.pi/2-theta)

            # Stack the coordinates to form the 3D points
            udvs = np.column_stack((ux, uy, uz))

            # Normalize the vectors to ensure they are unit vectors, range -1 to 1
            udvs /= np.linalg.norm(udvs, axis=1, keepdims=True)

            udvs = torch.tensor(udvs, dtype=torch.float32).view(16*3)

            range_img = im[...,0]*MAX_SENSOR_RANGE
            median_distance_obj = np.median(range_img[pred_masks[pred_idx]])

            if input_w_depth:
                model_input = torch.cat([udvs, torch.tensor(median_distance_obj, dtype=torch.float32).to(DEVICE).view(1)], dim=0) 

            with torch.no_grad():
                output_pose, predicted_root_distance = absolute_3d_joint_estimation_network(model_input.view(1, -1)) # new axis at dim0 to imitate batch
                pred_pose_np = output_pose.detach().cpu().numpy().reshape(16,3)
                gt_jnts_np = np.array(mapped_anno['annotations'][gt_idx]['keypoints_3d'], dtype=np.float32)[:, :-1]

                num_total +=1  
                
                # calc MPJPE
                PJPE = np.linalg.norm(pred_pose_np - gt_jnts_np, axis=-1)
                MPJPE = np.mean(PJPE)
                if MPJPE > 1:   # if greater than threshold of 1m error will be discarded
                    count_outliers +=1
                    print(count_outliers, num_total, f"error={MPJPE}")
                    continue    # continue if not wanted to include in mpjpe error calculation
                # else:
                #     continue  
                PJPE_list.append(PJPE)
                #print(f"MPJPE= {MPJPE}", end='\r', flush=True)
            if visu_flag_2d:
                # [[maybe_unused]] gt_boxes = mapped_anno['annotations'][obj_idx]['bbox']
                # [[maybe_unused]] gt_masks = mapped_anno['annotations'][obj_idx]['segmentation']
                gt_keypoints = mapped_anno['annotations'][gt_idx]['keypoints'][:, :-1].astype(int)
                img_black = np.zeros(shape=(256, 2048, 3), dtype=np.uint8)
                img_black = np.stack([im[..., 0], im[..., 0], im[..., 0]], axis=-1)  # depth image
                img_black = cv2.rectangle(img_black, 
                        (int(pred_boxes[pred_idx][0]),int(pred_boxes[pred_idx][1])), 
                        (int(pred_boxes[pred_idx][2]), int(pred_boxes[pred_idx][3])), 
                        color=(0,255,0), thickness=1) 
                
                img_black[gt_keypoints[:, 1], gt_keypoints[:, 0]] = (255, 0,0)  # gt red
                temp_gt_kpts = pred_keypoints[pred_idx].numpy().astype(int)      
                img_black[temp_gt_kpts[:, 1], temp_gt_kpts[:, 0]] = (0, 0,255)  # pred blue
                cv2.imshow("test", img_black[..., ::-1])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            if visu_flag_3d:
                print(f"error={MPJPE}", f"obj distance from sensor: {median_distance_obj}")
                line_set_pred, joints_pred = o3d_draw_skeleton(pred_pose_np, KINTREE_TABLE, color_set="CMY") 

                line_set_gt, joints_gt = o3d_draw_skeleton(gt_jnts_np, KINTREE_TABLE)
                coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1) # x: red, y: green, z: blue
                #o3d.visualization.draw_geometries([coord_mesh, line_set_2D, joints_2D])
                o3d.visualization.draw_geometries([coord_mesh, line_set_gt, joints_gt, line_set_pred, joints_pred])
    
    if save_dset:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        np.savez(os.path.join(save_path, file_id), \
                gt_idx=np.array(gt_idx_list, dtype=np.int32), \
                pred_kpts_2d=np.array(pred_kpts_2d_list, dtype=np.float32), \
                gt_kpts_2d=np.array(gt_kpts_2d_list, dtype=np.float32), \
                gt_kpts_3d=np.array(gt_kpts_3d_list, dtype=np.float32))

# convert to numpy array and make the error be measured in millimeters (mm)
PJPE_arr = np.array(PJPE_list, dtype=np.float32) * 1000 
print(f"Total number of objects: {num_total}")
plot_mpjpe_boxplot(PJPE_arr, IOU_THRESH)
print("Program exit 0")


# PJPE_total = np.nanmean(, axis=0)
# MPJPE_skalar = np.nanmean(PJPE_total)


# final_mpjpe = np.median(MPJPE_list)
# print(f"final_mpjpe= {final_mpjpe}")


# currently there are some dominant outliers which worsen the MPJPE calc, 
# so future work has to watch out for these outliers and implement some countermeasures