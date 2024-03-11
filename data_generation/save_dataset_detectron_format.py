'''
generation of ground-truth labels and the Spherical Projection image 
from the CARLA simulation data (data_generation/open3d_lidar_cuboids_skeleton.py)
'''

from detectron2.data import DatasetCatalog
#from mydataset import load_mydataset_json
#from model_config import cfg
import detectron2.data.transforms as T
from detectron2.data import (
    DatasetMapper, # the default mapper
    build_detection_train_loader
)

import os
import sys
import glob
import json
import numpy as np
from utils.spherical import to_deflection_coordinates, spherical_projection, o3d_draw_skeleton
import cv2
import matplotlib.pyplot as plt
import time 
import copy
import pycocotools
import base64
import open3d as o3d
from detectron2.structures.boxes import BoxMode
import uuid
from tqdm import tqdm
import pandas as pd
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# parameters derived from the carla data aquisition step
SENSOR_OFFSET_Z = 1.8   # mount height, defined as vehicle_offset = carla.Location(x=-0.5, z=1.8)  
MAX_SENSOR_RANGE = 250  # parameter of run-config

# desired output image dimension and granilarity, all are dependend from each other
'''
equirectangular image is employed to represent a spherical view at 360° of longitude, spanning from -180° to +180°
and 180° of latitude, starting from -90° (South
Pole) and extending to +90° (North Pole).

H_eq = Fov_eq / (Fov_sensor / #channels_sensor)

Inspiration of Ouster OS2: Fov_sensor = 22.5°, #channels_sensor = 128
'''
IMG_HEIGHT = int(1024)  
IMG_WIDTH = int(2048)
FOV_DEGREES = int(180) 
#

LABEL_COLORS = np.array([
    (255, 255, 255), # None             # 0
    (70, 70, 70),    # Building         # 1
    (100, 40, 40),   # Fences           # 2
    (55, 90, 80),    # Other            # 3
    (220, 20, 60),   # Pedestrian       # 4
    (153, 153, 153), # Pole             # 5
    (157, 234, 50),  # RoadLines        # 6
    (128, 64, 128),  # Road             # 7
    (244, 35, 232),  # Sidewalk         # 8
    (107, 142, 35),  # Vegetation       # 9
    (0, 0, 142),     # Vehicle          # 10
    (102, 102, 156), # Wall             # 11
    (220, 220, 0),   # TrafficSign      # 12
    (70, 130, 180),  # Sky              # 13
    (81, 0, 81),     # Ground           # 14
    (150, 100, 100), # Bridge           # 15
    (230, 150, 140), # RailTrack        # 16
    (180, 165, 180), # GuardRail        # 17
    (250, 170, 30),  # TrafficLight     # 18
    (110, 190, 160), # Static           # 19
    (170, 120, 50),  # Dynamic          # 20
    (45, 60, 150),   # Water            # 21
    (145, 170, 100), # Terrain          # 22
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

# carla has fixed joint names, only the desired joints are preserved
BONE_KEYS = [
        'crl_hips__C',        # Pelvis              # 0
        'crl_thigh__R',       # R_Hip               # 1
        'crl_leg__R',         # R_Knee              # 2
        'crl_foot__R',        # R_Ankle             # 3
        'crl_thigh__L',       # L_Hip               # 4
        'crl_leg__L',         # L_Knee              # 5
        'crl_foot__L',        # L_Ankle             # 6
        'crl_spine01__C',     # Torso               # 7
        'crl_neck__C',        # Neck                # 8
        'crl_Head__C',        # Head                # 9
        'crl_arm__R',         # R_Shoulder          # 10
        'crl_foreArm__R',     # R_Elbow             # 11
        'crl_hand__R',        # R_Wrist             # 12
        'crl_arm__L',         # L_Shoulder          # 13
        'crl_foreArm__L',     # L_Elbow             # 14
        'crl_hand__L'         # L_Wrist             # 15
]

# connection rules of above defined skeleton, indluding the filtered joints
KINTREE_TABLE = np.array([ 
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


# projection matrix to be used toconvert spherical coodinates into pixel coordinates
def calculate_projection_matrix(height=128, width=2048, fov_degrees=180):
    # Convert FOV to radians
    fov_radians = np.deg2rad(fov_degrees)
    # Calculate discretization and center coordinates
    # "-1" to ensure that the maximum pixel coordinates after transformation do not exceed the actual size of the image
    delta_phi = (width-1) / (2 * np.pi)
    delta_theta = (height-1) / (fov_radians)
    c_phi = (width-1) / 2   # shift to the right
    c_theta = (height-1) / 2    # shift downwards

    K = np.array([[delta_phi, 0, c_phi],
              [0, -delta_theta, c_theta],
              [0, 0, 1]])

    return K, fov_radians

K, FOV_RADIANS = calculate_projection_matrix(IMG_HEIGHT, IMG_WIDTH, FOV_DEGREES)


# transforms carla world coordinates to sensor coordinates
def to_ego(pose, world2sensor):
    pose_cloud = o3d.geometry.PointCloud()
    pose_cloud.points = o3d.utility.Vector3dVector(pose.reshape(-1,3).astype(np.float64))
    pose_cloud.transform(world2sensor)
    
    ego_pose_trajectory = np.asarray(pose_cloud.points).reshape(pose.shape)
    return ego_pose_trajectory


# transforms unordered 2D-list into ordered image-like structure using spherical projection
# Utilizes projection matrix to obtain the points which meet the granularity definitions, defined in fc:calculate_projection_matrix
def spherical_projection_v2(point_cloud, height=128, width=2048, fov_degrees=45):
    global K 
    global FOV_RADIANS
    fov_radians = FOV_RADIANS

    epsilon = 1e-6
    theta_range = [-fov_radians / 2 + epsilon, fov_radians / 2 - epsilon]
    
    point_cloud = np.column_stack((point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 3:])) 
    
    phi, theta = to_deflection_coordinates(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])

    inside_fov = np.argwhere((theta >= theta_range[0]) & (theta <= theta_range[1])).squeeze()
    phi = phi[inside_fov]
    theta = theta[inside_fov]
    xyz_things = point_cloud[inside_fov]

    spherical_coords = np.column_stack((phi, theta, np.ones_like(phi)))

    uv_coords = (np.matmul(K, spherical_coords.T)).T
    uv_coords = np.round(uv_coords[:, :2]).astype(int)

    valid_uv = np.argwhere((uv_coords[:, 0] < width) & (uv_coords[:, 1] < height)).squeeze()
    uv_coords = uv_coords[valid_uv]
    xyz_things = xyz_things[valid_uv]

    depth_values_sorted_descending = np.argsort(-np.linalg.norm(xyz_things[..., 0:3], axis=1))
    uv_coords = uv_coords[depth_values_sorted_descending]
    xyz_things = xyz_things[depth_values_sorted_descending]
    I = np.zeros((height, width, xyz_things.shape[1])).astype(np.float32)
    I[uv_coords[:, 1], uv_coords[:, 0], :] = xyz_things

    return I


# calculates the Direction Cosine of the local Surface Normals
def build_normal_xyz(xyz, norm_factor=2):
    '''
    @param xyz: ndarray with shape (h,w,3) containing a stagged point cloud
    @param norm_factor: int for the smoothing in Schaar filter
    '''
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]

    Sxx = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0/norm_factor)    
    Sxy = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0/norm_factor)

    Syx = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0/norm_factor)    
    Syy = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0/norm_factor)

    Szx = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0/norm_factor)    
    Szy = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0/norm_factor)

    gradient_magnitude = np.dstack((np.sqrt(Sxx**2 + Sxy**2), np.sqrt(Syx**2 + Syy**2), np.sqrt(Szx**2 + Szy**2)))
    #build cross product, without negation, for validation use
    '''
    normal += 1
    normal = normal/2
    plt.figure(1)
    plt.imshow(cv2.cvtColor(normal, cv2.COLOR_BGR2RGB))
    plt.show()
    '''
    normal_x = Syx*Szy - Szx*Syy
    normal_y = Szx*Sxy - Szy*Sxx
    normal_z = Sxx*Syy - Syx*Sxy
    
    normal = np.dstack((normal_x, normal_y, normal_z))

    # normalize cross product
    n = np.linalg.norm(normal, axis=2)

    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    
    #build direction cosine: this is the targetted output
    cos_theta = normal[..., 2]

    return normal, gradient_magnitude, cos_theta


def compute_iou(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate the area of the intersection rectangle
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the IoU
    try:
        iou = inter_area / float(box1_area + box2_area - inter_area)
    except ZeroDivisionError as er:
        iou = 0 # no continue in for loop needed, iou below threshold discards it automatically

    return iou


def keypoint_coco_and_visibility(semantic_instance_arr: np.ndarray, uv_skeleton: np.ndarray, instance_id: int):
    # label joint visibility 
    # v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible.

    keypoints_list = [] # needs to be JSON serializable 
    uv_set = set()
    for u, v in uv_skeleton:
        u, v = float(u), float(v)
        if any(np.isnan([u, v])):
            keypoints_list.extend([0, 0, 0]) # not labeled or obj on both image sides --> some parts will be discarded and interpreted as not labeled 
            continue

        keypoints_list.extend([u, v]) 
        if (u, v) not in uv_set:
            uv_set.add((int(u), int(v)))           
            # check if semantic tag is human (=4) and only the instance if keypoint location is not occluded by other human (crowd)         
            if semantic_instance_arr[int(v), int(u), 0] == 4 and semantic_instance_arr[int(v), int(u), 1] == instance_id:
                keypoints_list.extend([2])      # labeled and visible
            else: 
                keypoints_list.extend([1])      # labeled but not visible ==> occluded by some obj
        else:
            keypoints_list.extend([1])  

    return keypoints_list


def calc_bbox(box1: list, box2: list, img_shape: list, offset: int=1): 
    '''calc final bbox out of skeleton and obj instance mask

    Args: 2x list or np.dnarray format (x1,y2,x2,y2)
    Returns: list (x1,y2,x2,y2)
    '''
    
    box_final = np.round(np.array([np.minimum(box1[0], box2[0]), # left
            np.minimum(box1[1], box2[1]), # top
            np.maximum(box1[2], box2[2]), # right
            np.maximum(box1[3], box2[3])  # bottom
            ])).astype(int)
    
    # include offset that obj is inside bbox and no parts touches (/are part of) bbox extents
    if box_final[0]-offset >= 0: # pt1 u
        box_final[0] = box_final[0]-offset
    if box_final[1]-offset >= 0: # pt1 v
        box_final[1] = box_final[1]-offset
    if box_final[2]+offset < img_shape[1]: # pt2 u
        box_final[2] = box_final[2]+offset
    if box_final[3]+offset < img_shape[0]: # pt2 v
        box_final[3] = box_final[3]+offset
    
    box_final = box_final.tolist()  # converts np.ndarray to python list to be json serializable
    return box_final


# some visulaization functions
def visu_fov_cutout(hha_img: np.ndarray): 
    test = np.uint8(255*hha_img[...,:])

    # Get the dimensions of the image
    height, width, _ = test.shape
    # Define the number of lines and the spacing between them

    # 128 height, 22.5 fov
    # y0=448
    # h=128

    # 256 height, 45 fov
    y0=384
    h=256

    # Define the thickness and color of the hatched lines
    thickness = 2  # You can adjust this value
    color = (0, 0, 0)  # Black color in BGR format

    # Create a copy of the image to draw the lines on
    output_image = test.copy()

    # Draw hatched lines at the top of the image
    for i in np.linspace(0, y0, 75):
        y = int(i)
        cv2.line(output_image, (0, y), (width-1, y), color, thickness)

    for i in np.linspace(y0+h, 1024, 75):
        y = int(i)
        cv2.line(output_image, (0, y), (width, y), color, thickness)

    # Naming a window
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("test", 300, 700)
    cv2.imshow('test', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visu_angle_localSurfaceNormal(range_img: np.ndarray, normal: np.ndarray, gradient_magnitude: np.ndarray, cos_theta: np.ndarray):
    # Visalization of depth image, the gradients and normal and the angle of the local surface normal
    fig, axs = plt.subplots(4,1, figsize=(18, 18))

    axs[0].imshow(range_img[450:670, 689:1530], cmap='gray')
    axs[0].set_title('Depth Image')

    axs[1].imshow(gradient_magnitude[450:670, 689:1530, 0])
    axs[1].set_title('Output of step 1: Gradient Magnitude of x-channel')

    
    axs[2].imshow(normal[450:670, 689:1530], cmap='gray')
    axs[2].set_title('Output of step 3: Surface Normal Magnitude')

    cos_theta = np.where(np.isnan(cos_theta), -1, cos_theta)
    axs[3].imshow(cos_theta[450:670, 689:1530], cmap='gray')
    axs[3].set_title('Output of step 5: Normalized Direction Cosine')

    for ax in axs:
        ax.axis('off')

    plt.show()


root_dir = "/workspace/data/dataset/CARLA_HIGH_RES_LIDAR"

def save_custom_dataset(label_list: list, save_path: str = None, img_ext: str = ".exr", save_flag: bool = False, visu_flag: bool = True) -> None:
    global root_dir
    if not save_path:
        sys.exit("Exit.\nNo save path provided!")
    if not isinstance(label_list, list):
        label_list = [label_list]
    
    origin_dataset_id_list = []
    custom_start_idx = 0   # 6, 10, 17  , 394
    #avg_visible_jnts = [] 
    for file_number, file in enumerate(tqdm(label_list[custom_start_idx:]), start=custom_start_idx):
        # if file_number == 50:
        #     print(f'mean count visible joints:  {np.mean(np.array(avg_visible_jnts), axis=0)}')
        #     print(f'count occluded joints > 5:  {np.sum(np.array(avg_visible_jnts)[:, 0], axis=0)}')
        print(file)
        dataset_dict = {}
        
        image_id = str(uuid.uuid4())
        "/workspace/data/dataset/temp/train"
        real_save_path = save_path.split(os.path.sep)
        del real_save_path[-2]
        real_save_path = (os.path.sep).join(real_save_path)
        while(os.path.exists(os.path.join(real_save_path, "img", image_id+img_ext))):
            image_id = str(uuid.uuid4())

        origin_dataset_id = (os.path.sep).join(file.split(os.path.sep)[-4:-2])
        dataset_dict['origin_dataset_id']: str = origin_dataset_id
        dataset_dict['origin_frame_id']: str = os.path.splitext(file.split(os.path.sep)[-1])[0]
        dataset_dict['file_name']: str = os.path.join(real_save_path, "img", image_id+img_ext)
        dataset_dict['height']: int = IMG_HEIGHT
        dataset_dict['width']: int = IMG_WIDTH
        dataset_dict['image_id']: str = image_id

        #print(file)
        with open(file, 'r') as f:
            metadata = json.load(f) # annotation file, attributes of carla world state

        pcd_file = file.replace("labels","raw").replace(".json",".npy")
        raw_data = np.load(pcd_file)

        calib_file = file.replace("labels","calib")
        with open(calib_file, 'r') as f:
            calib = json.load(f)

        # Colorize the pointcloud based on the CityScapes color palette
        ObjTag = raw_data['ObjTag']   # semantic tag
        int_color = LABEL_COLORS[ObjTag]   
        ObjTag = ObjTag[...,np.newaxis] 
        ObjIdx = raw_data['ObjIdx'][...,np.newaxis] 

        # Obtain [x,y,z]-coordiates, NOTE: All y-coordinates will be negated if used, to match right-handed coordinate-system
        # convert left-handed coordinate system to right-handed (format: x, -y , z)
        '''
        The Unreal Engine uses a left-handed coordinate system with the following characteristics:

        X-axis points right
        Y-axis points forward
        Z-axis points upward

        To convert this to a right-handed coordinate system, you can negate the Y-axis. This will result in:

        X-axis points right (unchanged)
        Y-axis points backward (negated)
        Z-axis points upward (unchanged)
        '''
        points = np.stack([raw_data['x'], -raw_data['y'],raw_data['z']],axis=-1)

        # concatenate all important data types
        xyz_rgb_tag_id = np.concatenate([points, int_color, ObjTag, ObjIdx], axis=-1)

        start = time.perf_counter()
        xyz_things = spherical_projection_v2(xyz_rgb_tag_id, height=IMG_HEIGHT, width=IMG_WIDTH, fov_degrees=FOV_DEGREES)
        print(f"time taken: {time.perf_counter()- start}")
        # range_img = np.linalg.norm(xyz_things[...,0:3],axis=-1)
        # idx_img = np.full(range_img.shape, -1, dtype=np.int16) 

        # xyz-values of point cloud but already in structured, image-like format
        xyz_arr = copy.deepcopy(xyz_things[..., 0:3])
        ## point with no measurment, bc too far away or at near image boundaries, are defined as xyz=[0,0,0] (default carla behaviour).  
        ## to correctly calculate range_img, those points need to be handled correctly
        ## here defined equal to MAX_SENSOR_RANGE, because MAX_SENSOR_RANGE is not a critical distance, not of interst anyways 
        ## in the following thse points will be called exception values
 
        mask = np.all(xyz_arr == [0, 0, 0], axis=-1)
        xyz_arr[mask] = 3*[MAX_SENSOR_RANGE]
        
        # create HHA image encoding

        # 1.ch: euclidean distance point to origin, normalized against MAX_SENSOR_RANGE
        ## MAX_SENSOR_RANGE is very rare case so  are redefined equal to MAX_SENSOR_RANGE
        range_img = np.linalg.norm(xyz_arr, axis=-1)
        range_img[mask] = MAX_SENSOR_RANGE
        range_normalized_float = ((range_img/MAX_SENSOR_RANGE)).astype(np.float32)  

        idx_img = np.full(range_img.shape, -1, dtype=np.int16)
        # fig = plt.figure()
        # plt.imshow(np.uint8(255 * range_normalized_float))
        # plt.show()

        # 2.ch: height_above_ground in gravitational pos. z-direction
        ## points below sensor height should not be negative. minumum value starts at 0
        if (xyz_arr[..., 2]).min() < 0: 
            height_above_ground = xyz_arr[..., 2]+np.abs((xyz_arr[..., 2]).min())   # shift all z-values upwards (abs(sensor height - minimum)))
        else:   # rare scenario if no points are below sensor
            height_above_ground = xyz_arr[..., 2]

        ## make a limit of height, values greater the threshold are less relevant, 
        ## already considers exception values, were defined as 3*[MAX_SENSOR_RANGE] so asigned to threshold value 50
        height_above_ground = np.where(height_above_ground>=50, 50, height_above_ground)    

        ## normalize against height limit
        height_above_ground_normalized_float = (height_above_ground/50).astype(np.float32) 


        # 3.ch: angle of local surface normal
        normal, gradient_magnitude, cos_theta = build_normal_xyz(xyz_arr)
        
        # display last image channel: pixel local surface normal (cos theta)
        # visu_angle_localSurfaceNormal(range_img, normal, gradient_magnitude, cos_theta)
   
        angle_suface_normal_float = cos_theta.astype(np.float32)
        ## Normalization, that values lie in range [0, 1]
        ## NaN-values are considered as the surface normal pointing downwards (e.g. ceiling)
        angle_suface_normal_float = np.where((np.isnan(angle_suface_normal_float)), -1, angle_suface_normal_float)
        angle_suface_normal_float = (angle_suface_normal_float + 1) /2
        
        # concatenate channels to form final HHA image representation
        hha_img = np.stack([range_normalized_float, height_above_ground_normalized_float, angle_suface_normal_float], axis=-1)
        
        # save output image for testing (good cutout range: [450:670, 689:1530,::-1]) 
        # NOTE: cv2 saves images in BGR format, but also reads it again in BGR. so the final read will without changing channel order be as intended 
        #cv2.imwrite("hha_image.exr", hha_img[..., ::1].astype(np.float32)) 
        
        # display the fov cutout (22.5 and 45degrees) of original image
        #visu_fov_cutout(hha_img)

        # extract pedestrians only
        #mask_1 = (xyz_things[...,6] == 4)  # filter only walker (==4) in the fov, idx 6: semantic tag
        #temp = [str(x) for x in data.keys() if len(x) <=5]  # filter static objects (ids very high)
        anno_dict_iter = iter(metadata.items())
        walker_keys = []
        for key, anno_data in anno_dict_iter: # Tuple(str, dict)
            # check if all person specific entries are present
            ## anno_data keys: dict_keys(['bones', 'motion_state', 'velocity', 'acceleration', 'extent', 'location', 'rotation', 'semantic_tag', 'type_id'])
            if "bones" in anno_data.keys() and isinstance(anno_data['bones'], dict) and len(anno_data['bones'])>0 \
                    and anno_data['motion_state'] == 'dynamic' and "walker" in anno_data['type_id']:
                walker_keys.append(int(key))
        
        # check which objects (id) are present in both: in raw data point cloud (i.e. in fov of sensor) and in world state annotations contained
        mask_walker_in_scene = np.isin(xyz_things[...,-1], walker_keys) 
        #filter_walker = np.logical_and(mask_1, mask_walker_in_scene)

        # skip image if no walker is inside sensors' fov in that scene
        if not np.any(mask_walker_in_scene):   
            continue

        temp_pid_pixels = (xyz_things[mask_walker_in_scene, -1]).astype(int)  # unordered 1D list of ids in scene
        unique_temp_pid_pixels = np.unique(temp_pid_pixels)
        valid_pid = []
        for elem in unique_temp_pid_pixels:
            count_pid = len(temp_pid_pixels[temp_pid_pixels==elem]) # alternatively: np.count_nonzero(temp_pid_pixels == elem)
            if count_pid > 10:  # object should cover a certain pixel amount (#pixels is choosen as 10)
                valid_pid.append(elem)

        if len(valid_pid) == 0:   # skip frame, no valid, big enough walkers
            continue

        mask_valid_walker = np.isin(xyz_things[...,-1], valid_pid)  # get mask of valid objects
        valid_walker_img = np.where(mask_valid_walker, xyz_things[...,-1], -1) # on True mask positions assign its corresponding pid, else =-1
        
        # only for visu
        final_img = np.where(mask_valid_walker[..., np.newaxis], [255, 255, 0], cv2.cvtColor(range_img, cv2.COLOR_GRAY2RGB)).astype(np.uint8)
        
        # person_data = metadata[f'{int(key)}']
        # center_x, center_y, center_z = person_data['location']
        # x,y,z = to_ego(np.array([center_x, center_y, center_z]),W2S)
        # # Convert the point cloud to spherical coordinates (format: x, -y , -z)
        # phi, theta = to_deflection_coordinates(x, -y, z)
        # spherical_coords = np.column_stack((phi, theta, np.ones_like(phi)))
        # # Project the spherical coordinates to image coordinates (u, v)
        # uv_coords = (np.matmul(K, spherical_coords.T)).T

        ##########
        # Ground Truth Annotations
        ##########

        # Transformation matrix: world to sensor coordinates (from annotation file, metadata measured in world coordinates)
        W2S = np.linalg.inv(np.array(calib["world2sensor"]))
        # Initialize annotation list 
        annotations = []
        # Initialize a list to store bounding boxes
        boxes = {}
        
        # K, _ = calculate_projection_matrix(height=IMG_HEIGHT, width=IMG_WIDTH, fov_degrees=FOV_DEGREES)
        counter_misplaced = 0
        vrus = {}
        visible_jnts = {'1': 0, '2': 0} 
        try:    # https://www.kaggle.com/code/linrds/convert-rle-to-polygons
            for pos_valid_pid, key in enumerate(valid_pid): # iterate over all valid pids
                maskedArr = np.where(valid_walker_img == key, 1, 0).astype(np.uint8)
            
                # xyz-coordinates of segmented object. Format: (u-coord, v-coord, x, y, z)
                argwhere_masked = np.array(np.nonzero(maskedArr))
                segm_pcl = np.column_stack( (argwhere_masked[::-1].T, xyz_things[argwhere_masked[0], argwhere_masked[1], :3])).tolist() 
                
                '''
                If dict, it represents the per-pixel segmentation mask in COCO’s compressed RLE format. 
                The dict should have keys “size” and “counts”. You can convert a ! uint8 ! segmentation mask of 0s and 1s
                    into such dict by pycocotools.mask.encode(np.asarray(mask, order="F")). 
                    cfg.INPUT.MASK_FORMAT must be set to bitmask if using the default data loader with such format.

                examples can be found on: https://www.kaggle.com/code/linrds/convert-rle-to-polygons
                '''

                # pixel coverage (area) of the mask already checked for being interpreted as valid entity, which is set to 10px
                rle = pycocotools.mask.encode(np.asarray(maskedArr, order="F"))  # segmentation mask in COCO’s compressed RLE format, order: fortran array

                # Convert the byte string to a regular string, to avoid TypeError: Object of type bytes is not JSON serializable
                rle['counts'] = base64.b64encode(rle['counts']).decode('utf-8')     # optional: latin1
                # if visu_flag:   # decoded coco rle mask vs original mask
                    # # restore original mask of rle
                    # loaded_rle = copy.deepcopy(rle)
                    # loaded_rle['counts'] = base64.b64decode(loaded_rle['counts'])
                    # binary_mask = pycocotools.mask.decode(loaded_rle)
                    
                    # _, ax = plt.subplots(1, 2, figsize=(20, 16))
                    # ax[0].imshow(binary_mask, cmap="gray")
                    # ax[1].imshow(maskedArr, cmap="gray")
                    # ax[0].set_title("restored")
                    # ax[1].set_title("ori")
                    # plt.show()
                
                '''
                handle objects that wrap around the image boundaries 
                check if person is on both vertical edges, bc of 360 degree horiziontal fov caption
                '''
                # along x-axis          
                x_axis = np.sort(np.unique(np.nonzero(maskedArr)[1]))  # get all unique x coordinates, sorted
                diff_x_axis = np.diff(x_axis)   # calc difference between each pixel coodinate in x direction
                split_idx = np.argwhere(diff_x_axis>hha_img.shape[1]/2).squeeze()   # if diff is larger than threshold, entity needs to be adjusted
                assert split_idx.size < 2, "problematic split_idx size, needs to be checked"
                flag_boundary_obj = False
                try:
                    if split_idx.size == 0: # normal: obj does NOT wrap around the image boundaries
                        x1 = x_axis[0]
                        x2 = x_axis[-1]
                    elif split_idx.size == 1: # wrap around the image boundaries
                        flag_boundary_obj = True    # will be handled as two objs
                        split_1  = x_axis[:split_idx+1] # left image half
                        split_2 = x_axis[split_idx+1:]  # right image half
                        
                        # split instance into two
                        ## left side
                        maskedArr_left = np.zeros_like(maskedArr, dtype=np.uint8)
                        maskedArr_left[:, split_1[0]:(split_1[-1]+1)] = maskedArr[:, split_1[0]:(split_1[-1]+1)]

                        # xyz-coordinates of segmented object. Format: (u-coord, v-coord, x, y, z)
                        argwhere_masked_left = np.array(np.nonzero(maskedArr_left))
                        segm_pcl_left = np.column_stack( (argwhere_masked_left[::-1].T, xyz_things[argwhere_masked_left[0], argwhere_masked_left[1], :3])).tolist() 

                        # segmentation mask in COCO’s compressed RLE format, order: fortran array
                        rle_left = pycocotools.mask.encode(np.asarray(maskedArr_left, order="F"))  
                        rle_left['counts'] = base64.b64encode(rle_left['counts']).decode('utf-8') 
                        # xyz-coordinates of object
                        #pcl_left = xyz_things[:, split_1[0]:(split_1[-1]+1), :3]

                        ## right side
                        maskedArr_right = np.zeros_like(maskedArr, dtype=np.uint8)
                        maskedArr_right[:, split_2[0]:(split_2[-1]+1)] = maskedArr[:, split_2[0]:(split_2[-1]+1)]

                        # xyz-coordinates of segmented object. Format: (u-coord, v-coord, x, y, z)
                        argwhere_masked_right = np.array(np.nonzero(maskedArr_right))
                        segm_pcl_right = np.column_stack( (argwhere_masked_right[::-1].T, xyz_things[argwhere_masked_right[0], argwhere_masked_right[1], :3])).tolist()  

                        # segmentation mask in COCO’s compressed RLE format, order: fortran array
                        rle_right = pycocotools.mask.encode(np.asarray(maskedArr_right, order="F"))  
                        rle_right['counts'] = base64.b64encode(rle_right['counts']).decode('utf-8') 
                        # xyz-coordinates of object
                        #pcl_right = xyz_things[:, split_2[0]:(split_2[-1]+1), :3]

                        # preleminary bbox of instance id mask
                        box_left_x = (split_1[0], split_1[-1])
                        box_right_x = (split_2[0], split_2[-1])
                        x1 = split_2[0]
                        x2 = split_1[-1]
                except ValueError as ve:
                    print(ve)
                except TypeError as te:
                    print(te)
                # along y-axis
                y_axis = np.sort(np.unique(np.nonzero(maskedArr)[0]))  
                y1 = y_axis[0]
                y2 = y_axis[-1]

                # bbox extent of raw data obj tags, needs to be checked against metadata, specifically while observing the joint locations
                # NOTE: sometimes writing inconsistencies, because metadata and raw data are frame inconsistent (CARLA related problem when using lidar) 
                # format: (x1, y1, x2, y2); extents have maximas included, no offset currently       
                if flag_boundary_obj:   # handled as 2 objs
                    box_left = [box_left_x[0], y1, box_left_x[1], y2]
                    box_right = [box_right_x[0], y1, box_right_x[1], y2]
                else:
                    box = [x1, y1, x2, y2]  # handled as 1 obj

                # get CARLA bones in 3D [x,y,z]-coordinates and only obtain the joints of the desired output skeleton structure
                # filtered bones names are enlisted in BONE_KEYS global var at beginning of script                                                                                                  
                bones_list = [metadata[str(key)]["bones"][bone_key]["world"] for bone_key in BONE_KEYS] # format: list[list]

                # transform world joint coordinates to sensor coordinates 
                # joints of each instances are saved, format: dict[ndarray], 
                vrus[key] = to_ego(np.array(bones_list), W2S)
                
                # draw skeleton
                # if visu_flag:
                #     jnts_skeleton = vrus[key]["joints_world"].squeeze()
                #     line_set, joints = o3d_draw_skeleton(jnts_skeleton, KINTREE_TABLE)
                #     coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
                #     o3d.visualization.draw_geometries([line_set, joints])#, coord_mesh])

                skeleton_xyz = copy.deepcopy(vrus[key])
                skeleton_xyz[:,1] = -skeleton_xyz[:,1]  # left to right handed-coordsys --> y-coord inverted

                # spherial projection
                phi, theta = to_deflection_coordinates(skeleton_xyz[:,0], skeleton_xyz[:,1], skeleton_xyz[:,2])
                spherical_coords = np.column_stack((phi, theta, np.ones_like(phi)))
                uv_coords = ((np.matmul(K, spherical_coords.T)).T)
                # Round the u and v coordinates to float32 and 3 decimal places
                uv_skeleton = np.round(uv_coords[:, :2], 3).astype(np.float32).squeeze()
                # Round the u and v coordinates to integers
                uv_skeleton_int = np.round(uv_coords[:, :2]).astype(int).squeeze()
                crl_hips__C_u, crl_hips__C_v = uv_skeleton[0]

                ##### 
                # check if bbox center of metadata is inside raw data object extent
                #####
                center_x_world, center_y_world, center_z_world = metadata[f'{int(key)}']['location']    # cuboid center
                center_x_sensor, center_y_sensor, center_z_sensor = to_ego(np.array([center_x_world, center_y_world, center_z_world]),W2S)
                # Note: coordinates in left-handed UE-coords (needs to be coverted to format: x, -y , z)
                phi, theta = to_deflection_coordinates(center_x_sensor, -center_y_sensor, center_z_sensor)
                spherical_coords = np.column_stack((phi, theta, np.ones_like(phi)))
                # Project the spherical coordinates to image coordinates (u, v)
                uv_coords = (np.matmul(K, spherical_coords.T)).T.squeeze()    # needs to be squeezed, bc only one coordinate
                # Round the u and v coordinates to integers
                uv_cuboid_center_int = np.round(uv_coords[:2]).astype(int)
                uv_cuboid_center = np.round(uv_coords[:2], 3).astype(np.float32)
                #####

                if False:   # visu_flag
                    # display binary object mask and the joints
                    # Project the spherical coordinates to image coordinates (u, v)
                    
                    binary_mask = maskedArr
                    binary_mask = (np.stack([binary_mask, binary_mask, binary_mask], axis=-1) * 255).astype(np.uint8)
                    
                    u, v = uv_skeleton_int[...,0], uv_skeleton_int[...,1]
                    binary_mask[v, u] = [255, 0, 255]
                    binary_mask[uv_cuboid_center_int[1], uv_cuboid_center_int[0]] = [0, 255, 0]

                    # display bbox
                    if not flag_boundary_obj:
                        pt1 = (int(box[0]), int(box[1]))
                        pt2 = (int(box[2]), int(box[3]))
                        color = (150, 150, 150)
                        binary_mask_umat = cv2.UMat(copy.deepcopy(binary_mask))
                        binary_mask_umat = cv2.rectangle(binary_mask_umat, pt1, pt2, color, 1)
                    elif flag_boundary_obj:
                        pt11 = (int(box_left[0]), int(box_left[1]))
                        pt12 = (int(box_left[2]), int(box_left[3]))
                        pt21 = (int(box_right[0]), int(box_right[1]))
                        pt22 = (int(box_right[2]), int(box_right[3]))
                        color = (150, 150, 10)
                        binary_mask_umat = cv2.UMat(copy.deepcopy(binary_mask))
                        binary_mask_umat = cv2.rectangle(binary_mask_umat, pt11, pt12, color, 1)
                        binary_mask_umat = cv2.rectangle(binary_mask_umat, pt21, pt22, color, 1)

                        # Naming a window 
                        cv2.namedWindow("binary_mask", cv2.WINDOW_NORMAL) 
                        cv2.resizeWindow("binary_mask", 1920, 1080) 
                        cv2.imshow('binary_mask', binary_mask_umat)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()    

                # correct joint locations that they are not out of legal image boundaries
                # e.g. if image width is 100 and point is at x=100, then need to adjust the point that it is x_=1 
                # ==> result: this joint is now handled to be positioned on left image side, rather than falsely on the right side
                uv_skeleton_correctly_shifted = copy.deepcopy(uv_skeleton)
                for jnt_idx, (u_float, v_float) in enumerate(uv_skeleton):
                    # along u-coord 
                    if u_float > hha_img.shape[1]-1: 
                        uv_skeleton_correctly_shifted[jnt_idx, 0] = u_float - (hha_img.shape[1]-1)
                    if u_float < 0: # TESTING, should not be <0
                        print()
                        pass
                    # along v-coord 
                    if v_float > hha_img.shape[0]-1:
                        uv_skeleton_correctly_shifted[jnt_idx, 1] = v_float - (hha_img.shape[0]-1)
                    if v_float < 0: # TESTING, should not be <0
                        print()
                        pass

                # check if skeleton is on both image sides, to be handled as two objs
                u_coord_skeleton = copy.deepcopy(uv_skeleton_correctly_shifted[:, 0])
                u_axis_argsort = np.argsort(u_coord_skeleton)
                u_axis_sorted = u_coord_skeleton[u_axis_argsort]
                diff_u_axis = np.diff(u_axis_sorted)   # calc difference between each pixel coodinate in x direction
                split_idx = np.argwhere(diff_u_axis>hha_img.shape[1]/2).squeeze()   # if diff is larger than threshold, entity needs to be adjusted
                flag_boundary_skeleton = False
                try:
                    if split_idx.size == 0: # normal: obj does NOT wrap around the image boundaries
                        pass # no split necesarry
                    elif split_idx.size == 1: # jnts wrap around the image boundaries     
                        flag_boundary_skeleton = True         
                        skeleton_idxs_left = u_axis_argsort[:split_idx+1]
                        skeleton_idxs_right = u_axis_argsort[split_idx+1:]
                except ValueError as ve:
                    print(ve)
                except TypeError as te:
                    print(te)
                    
                if flag_boundary_skeleton:
                    # assign joints to either left or right obj. at current sekelton, joints on the opposite image half are assigned with (u=np.nan, v=np.nan)
                    skeleton_left = np.full(shape=uv_skeleton_correctly_shifted.shape, fill_value=np.nan, dtype=uv_skeleton_correctly_shifted.dtype)
                    skeleton_right = np.full(shape=uv_skeleton_correctly_shifted.shape, fill_value=np.nan, dtype=uv_skeleton_correctly_shifted.dtype)
                    skeleton_left[skeleton_idxs_left] = uv_skeleton_correctly_shifted[skeleton_idxs_left]
                    box_skeleton_left = (np.nanmin(skeleton_left[:, 0]), np.nanmin(skeleton_left[:, 1]), \
                                    np.nanmax(skeleton_left[:, 0]), np.nanmax(skeleton_left[:, 1]))
                    
                    skeleton_right[skeleton_idxs_right] = uv_skeleton_correctly_shifted[skeleton_idxs_right]
                    box_skeleton_right = (np.nanmin(skeleton_right[:, 0]), np.nanmin(skeleton_right[:, 1]), \
                                    np.nanmax(skeleton_right[:, 0]), np.nanmax(skeleton_right[:, 1]))
                else:
                    box_skeleton = (np.min(uv_skeleton_correctly_shifted[:, 0]), np.min(uv_skeleton_correctly_shifted[:, 1]), \
                                    np.max(uv_skeleton_correctly_shifted[:, 0]), np.max(uv_skeleton_correctly_shifted[:, 1]))
                

                #####
                #  match obj bbox gathered from instance id (raw data) with bbox of skeleton (metadata).
                #####

                # normal/one instance, NOT present on both image sides
                if not any([flag_boundary_obj, flag_boundary_skeleton]):    # 
                    iou = compute_iou(box, box_skeleton)
                    if iou>=0.5:    # here use higher iou threshold than below with splitted instance, 
                                    # not too large bc skeleton height will be smaller than obj coverage
                        keypoints_list = keypoint_coco_and_visibility(xyz_things[..., -2:], uv_skeleton_correctly_shifted, key)
                        
                        # gather xyz-coords of joints with visibility label
                        temp_skeleton = np.array(keypoints_list).reshape(-1, 3)
                        keypoints_3d = np.column_stack([skeleton_xyz, temp_skeleton[:, -1]]).tolist()

                        # calc final bbox
                        box_final = calc_bbox(box, box_skeleton, rle['size'])
                        obj = { 
                            "bbox": box_final,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": 0,   
                            "segmentation": rle,
                            "iscrowd": 0,
                            "keypoints": keypoints_list,
                            "segm_pcl": segm_pcl,
                            "keypoints_3d": keypoints_3d}
                        annotations.append(obj) 

                    else:    # too small intersect, discard/skip object
                        continue
                else:
                    if all([flag_boundary_obj, flag_boundary_skeleton]):  
                        # both, instance id mask and skeleton result in two instances, \
                        # obj is present on both image sides
                        iou_left = compute_iou(box_left, box_skeleton_left)
                        iou_right = compute_iou(box_right, box_skeleton_right)

                        # thresholding: iou<10% is considered FP and has minimum of 2 joints on either side (less is not of interest)
                        if any([iou_left>=0.1 and (len(BONE_KEYS)- np.sum(np.isnan(skeleton_left))/2 > 1), \
                                iou_right>=0.1 and (len(BONE_KEYS) - np.sum(np.isnan(skeleton_right))/2 > 1)]):  
                            # check left side
                            if iou_left>=0.1 and (len(BONE_KEYS) - np.sum(np.isnan(skeleton_left))/2 > 1):
                                keypoints_list_left = keypoint_coco_and_visibility(xyz_things[..., -2:], skeleton_left, key)
                                box_final_left = calc_bbox(box_left, box_skeleton_left, rle['size'])
                                
                                # gather xyz-coords of joints with visibility label
                                temp_skeleton = np.array(keypoints_list_left).reshape(-1, 3)
                                keypoints_3d = np.column_stack([skeleton_xyz, temp_skeleton[:, -1]]).tolist()

                                obj = { 
                                    "bbox": box_final_left,
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "category_id": 0,   
                                    "segmentation": rle_left,
                                    "iscrowd": 0,
                                    "keypoints": keypoints_list_left,
                                    "segm_pcl": segm_pcl_left,
                                    "keypoints_3d": keypoints_3d}
                                annotations.append(obj) 

                            # check right side
                            if iou_right>=0.1 and (len(BONE_KEYS) - np.sum(np.isnan(skeleton_right))/2 > 1):
                                keypoints_list_right = keypoint_coco_and_visibility(xyz_things[..., -2:], skeleton_right, key)
                                box_final_right = calc_bbox(box_right, box_skeleton_right, rle['size'])

                                # gather xyz-coords of joints with visibility label
                                temp_skeleton = np.array(keypoints_list_right).reshape(-1, 3)
                                keypoints_3d = np.column_stack([skeleton_xyz, temp_skeleton[:, -1]]).tolist()

                                obj = { 
                                    "bbox": box_final_right,
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "category_id": 0,   
                                    "segmentation": rle_right,
                                    "iscrowd": 0,
                                    "keypoints": keypoints_list_right,
                                    "segm_pcl": segm_pcl_right,
                                    "keypoints_3d": keypoints_3d}
                                annotations.append(obj)

                        else:   # too small intersect, discard/skip object
                            continue

                    else:   # skeleton and instance mask extract different number of objs
                        if flag_boundary_obj and not flag_boundary_skeleton:
                            # assumes two instances from raw data side, but only one from skeleton 
                            # ==> will only be ONE final instance
                            iou_left = compute_iou(box_left, box_skeleton)
                            iou_right = compute_iou(box_right, box_skeleton)
                            
                            keypoints_list = keypoint_coco_and_visibility(xyz_things[..., -2:], uv_skeleton_correctly_shifted, key)

                            # gather xyz-coords of joints with visibility label
                            temp_skeleton = np.array(keypoints_list).reshape(-1, 3)
                            keypoints_3d = np.column_stack([skeleton_xyz, temp_skeleton[:, -1]]).tolist()
                            
                            if iou_left>=0.4:
                                box_final = calc_bbox(box_left, box_skeleton, rle['size'])
                                obj = { 
                                    "bbox": box_final,
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "category_id": 0,   
                                    "segmentation": rle_left,
                                    "iscrowd": 0,
                                    "keypoints": keypoints_list,
                                    "segm_pcl": segm_pcl_left,
                                    "keypoints_3d": keypoints_3d}
                                annotations.append(obj)
                            elif iou_right>=0.4:
                                box_final = calc_bbox(box_right, box_skeleton, rle['size'])
                                obj = { 
                                    "bbox": box_final,
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "category_id": 0,   
                                    "segmentation": rle_right,
                                    "iscrowd": 0,
                                    "keypoints": keypoints_list,
                                    "segm_pcl": segm_pcl_right,
                                    "keypoints_3d": keypoints_3d}
                                annotations.append(obj)
                            else:
                                continue    # too small intersect, discard/skip object 
                                       
                        elif not flag_boundary_obj and flag_boundary_skeleton:
                            # assumes two instances from metadata skeleton side, but only one from instance id raw data
                            # ==> will only be ONE final instance
                            iou_left = compute_iou(box, box_skeleton_left)
                            iou_right = compute_iou(box, box_skeleton_right)

                        if not any([iou_left>=0.4 and (16- np.sum(np.isnan(skeleton_left))/2 > 1), \
                                iou_right>=0.4 and (16- np.sum(np.isnan(skeleton_right))/2 > 1)]):  
                            continue    # too small intersect, discard/skip object 
                        else:
                            # handled as one object, because object not visible in output image and therfore joints area on occluded side are neglectible
                            if iou_left >= 0.4 and (16- np.sum(np.isnan(skeleton_left))/2 > 1) > 1:
                                keypoints_list = keypoint_coco_and_visibility(xyz_things[..., -2:], skeleton_left, key)
                                box_final = calc_bbox(box, box_skeleton_left, rle['size'])

                                # gather xyz-coords of joints with visibility label
                                temp_skeleton = np.array(keypoints_list).reshape(-1, 3)
                                keypoints_3d = np.column_stack([skeleton_xyz, temp_skeleton[:, -1]]).tolist()

                                obj = { 
                                    "bbox": box_final,
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "category_id": 0,   
                                    "segmentation": rle,
                                    "iscrowd": 0,
                                    "keypoints": keypoints_list,
                                    "segm_pcl": segm_pcl,
                                    "keypoints_3d": keypoints_3d}
                                annotations.append(obj)
                            if iou_right >= 0.4 and (16- np.sum(np.isnan(skeleton_right))/2 > 1):
                                keypoints_list = keypoint_coco_and_visibility(xyz_things[..., -2:], skeleton_right, key)
                                box_final = calc_bbox(box, box_skeleton_right, rle['size'])

                                # gather xyz-coords of joints with visibility label
                                temp_skeleton = np.array(keypoints_list).reshape(-1, 3)
                                keypoints_3d = np.column_stack([skeleton_xyz, temp_skeleton[:, -1]]).tolist()

                                obj = { 
                                    "bbox": box_final,
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "category_id": 0,   
                                    "segmentation": rle,
                                    "iscrowd": 0,
                                    "keypoints": keypoints_list,
                                    "segm_pcl": segm_pcl,
                                    "keypoints_3d": keypoints_3d}
                                annotations.append(obj)

        # object iteration in frame completed
        except KeyError as ke:
            print(ke)
            print("Error with file", file)  # no bones in label data
            continue
        except ValueError as ve:
            print(ve)
            print("Error with file", file)  

        # visualize results 
        if visu_flag: 
            for obj_dict in annotations[:]:

                # display point cloud with skeleton
                pose_cloud = o3d.geometry.PointCloud()
                pose_cloud.points = o3d.utility.Vector3dVector(np.array(obj_dict['segm_pcl'])[..., 2:])
                color = [0, 0, 0] # Assigning black color to all points
                colors = np.tile(color, (len(np.array(obj_dict['segm_pcl'])[..., 2:]), 1))
                pose_cloud.colors = o3d.utility.Vector3dVector(colors)

                line_set, joints = o3d_draw_skeleton(np.array(obj_dict['keypoints_3d'])[...,:-1], KINTREE_TABLE)
                o3d.visualization.draw_geometries([pose_cloud, line_set, joints])

                # obj segmentation + bbox + keypoints
                loaded_rle = copy.deepcopy(obj_dict['segmentation'])
                loaded_rle['counts'] = base64.b64decode(loaded_rle['counts'])
                binary_mask = pycocotools.mask.decode(loaded_rle)
                binary_mask = (np.stack([binary_mask, binary_mask, binary_mask], axis=-1) * 255).astype(np.uint8)
                
                #  keypoints encoded as u-coord, v-coord, visibility (only interest in: flag v=2: labeled and visible)
                keypoints = np.array(obj_dict['keypoints'], dtype=np.int64).reshape(-1, 3) 
                keypoints = keypoints[keypoints[..., -1] == 2, :-1]
                if keypoints.shape[0]>0:
                    binary_mask[keypoints[:, 1], keypoints[:, 0]] = [255, 0, 255]

                pt1 = (int(obj_dict['bbox'][0]), int(obj_dict['bbox'][1]))
                pt2 = (int(obj_dict['bbox'][2]), int(obj_dict['bbox'][3]))
                color = (150, 150, 150)
                binary_mask_umat = cv2.UMat(copy.deepcopy(binary_mask))
                binary_mask_umat = cv2.rectangle(binary_mask_umat, pt1, pt2, color, 1)

                cv2.namedWindow("binary mask /w obj segmentation + bbox + keypoints", cv2.WINDOW_NORMAL) 
                cv2.resizeWindow("binary mask /w obj segmentation + bbox + keypoints", 1920, 1080) 
                cv2.imshow("binary mask /w obj segmentation + bbox + keypoints", binary_mask_umat)
                
                range_img = copy.deepcopy((255* hha_img).astype(np.uint8))[...,0]
                hha_img_umat = cv2.UMat( range_img)
                hha_img_umat = cv2.rectangle(hha_img_umat, pt1, pt2, 100, 1)
                cv2.namedWindow("hha image", cv2.WINDOW_NORMAL) 
                cv2.resizeWindow("hha image", 1920, 1080)
                cv2.imshow("hha image", hha_img_umat)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print(pos_valid_pid, counter_misplaced)
        # skip frame if no large enough walker obj is in scene/ sensor fov
        if len(annotations) == 0:
            continue

        print(f'annotation length: {len(annotations)} of {pos_valid_pid} total')
                    
        dataset_dict['annotations'] = annotations
        num_instances = len(dataset_dict['annotations'])
        #avg_visible_jnts.append(((visible_jnts['1']), (visible_jnts['2']/num_instances)))
        #avg_visible_jnts.append(((visible_jnts['1']/num_instances), (visible_jnts['2']/num_instances)))
        if save_flag:
            # save image file
            os.makedirs(os.path.join(save_path, "img"), exist_ok=True)
            cv2.imwrite(os.path.join(save_path, "img", dataset_dict['image_id'] + img_ext), hha_img.astype(np.float32))  

            # save label file
            os.makedirs(os.path.join(save_path, "labels"), exist_ok=True)
            try:
                with open(os.path.join(save_path, "labels", dataset_dict['image_id'] + ".json"), 'w') as outfile:
                    json.dump(dataset_dict, outfile)
                    os.chmod(os.path.join(save_path, "labels", image_id + ".json"), mode=0o666)
            except TypeError as tErr:
                print(tErr)
                print(annotations)
        # checkout: add origin_dataset_id_set to the .txt-file of registered origin_datasets
        # if save_flag:
        #     if origin_dataset_id not in origin_dataset_id_list:
        #         with open(os.path.join(root_dir, 'registered_dataset_summary.txt'), 'a+') as f:
        #             f.write(origin_dataset_id + '\n')
        #         origin_dataset_id_list.append(origin_dataset_id)


if __name__ == "__main__":
    overlook_registered_dataset = True

    final_dirs = [file for file in glob.glob(root_dir+'/**/*') if 'Town' in file.split(os.path.sep)[-1]]
    if not overlook_registered_dataset:
        if os.path.exists(os.path.join(root_dir, 'registered_dataset_summary.txt')):
            with open(os.path.join(root_dir, 'registered_dataset_summary.txt'), 'r') as f:
                registered_datasets = f.readlines()
            registered_datasets = [temp.rstrip('\n') for temp in registered_datasets]

            for registered_dataset in registered_datasets:
                temp = os.path.join(root_dir,registered_dataset)
                if temp in final_dirs:
                    final_dirs.remove(temp)
        # 
        # for idx, dataset_path in enumerate(final_dirs):
        #     dataset_name = os.path.sep.join(dataset_path.split(os.path.sep)[-2:])
        #     if dataset_name in registered_datasets:
        #         final_dirs.pop(idx)
    
    total_entry_files = []
    for dataset_dir in final_dirs:
        total_entry_files.extend(glob.glob(os.path.join(dataset_dir, 'labels/*')))

    if len(total_entry_files) == 0:
        sys.exit("Provided Dataset already registered\nExiting.")
    
    # input_path = "/workspace/data/dataset/CARLA_HIGH_RES_LIDAR/2023-05-30_18-45-59/Town10HD_Opt"
    # total_entry_files = glob.glob(input_path + "/labels/*")   
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(total_entry_files, test_size=0.2, random_state=42)
    
    print(f"train set size: {len(train_data)},\t test set size: {len(test_data)}")
    # train split,  80%
    print("saving train set")
    save_custom_dataset(train_data, save_path="/workspace/data/dataset/temp/train") 
    
    # test split,   20%
    print("saving test set")
    time.sleep(5)
    save_custom_dataset(test_data, save_path="/workspace/data/dataset/temp/test") 

    print("save successful, marking datasets as registered")

    with open(os.path.join(root_dir, 'registered_dataset_summary.txt'), 'a+') as f:
        for datasets in final_dirs:
            temp = (os.path.sep).join(datasets.split(os.path.sep)[-2:])
            f.write(temp + '\n')
        


    