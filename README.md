# Implementation of the absolute HPE pipeline in the master thesis of Kerim Turacan

This include thes dataset creation and the 2step approach of detecing the 2D pose in an spherical projected LiDAR image and lifting into 3D.
The CARLA simulator is used for synthetic data generation (see Sec. 1), where AV scences are created within the realistic UE4 simulation environments.
The most important actors are spawnable and configureable pedestrians and cars. On one car in each scene a (semenatic) LiDAR is attached, from which the dataset used in this work was created.
The autopilot feature of actors in CARLA make them move autonomously at (mostly) high realsism.
However, with the used CARLA version 0.9.13 the poses of pedestrians were often very similar with the arms pointing straight downwards. Natural mutual interaction could not be observed, as the goal of the autopilot feature is just to move to the target distination; other interactions are not implemented by default.
It could also be observed that some persons form T-poses.
Additional problems has been encoutered in the synchronisation of the CARLA world state data and the measurments of the utilized LiDAR sensor. This resulted in joint locations of pedestrians not being consistent with the sensor's point cloud. When projected to an spherical image, the keypoints were sometimes not part of the target instance. Unfortunately, this problem seemed to be persistent with other sensor types. Later version may solve this issues as this drastically worsens the quality of the ground-truth keypoint annotations. To counter-act, a thresholding was implemented and frames discarded if this problem occured.

After creating the labeled LiDAR dataset, the LiDAR data gets projected onto an image plane using Spherical Projection (see utils/fc_matrix_and_projection_calculation.py). The Spherical Projection Image is dependent on the projection matrix, which defines the granularity of the projection and the employed FOV. Ideally a high FOV can be used first, from which a subimage can be then cropped out, which yields the desired FOV.

The Spherical Projection Image is used as input of a 2D estimator. Here, the Detectron2 keypoint_rcnn_R_50_FPN is used.
Feature extraction over the input image was performed using a ResNet-50-FPN backbone that yields a set of convolutional feature maps to extract the ROIs. A box head performs object classification and bounding box regression using the feature maps provided by the backbone. An optional mask head yields instance segmentation masks, and a keypoint detection head predicts specific key points on the objects detected by the box head network. This keypoint head is designed to predict the 16 keypoints delineated by the adapted custom CARLA skeleton structure.

The 2D ouputs are then passed to a lfting network, which aims to lift the 2D ouputs of the 2D estimator into 3D. The network consists of 2 regression networks, which predict the scalar root distance r_Root and the relative distance pose r_rel. the 3D pose can be recovered by:
```math
 (1_N \cdot r_{Root} + r_{rel}) \cdot \text{UDV}
 ```
UDV is the unit direction vector including the normalized cartesian componenents of the 2D joints. This is acchieved by converting the 2D pixel locations to spherical coordinates by inverse spherical projection and then the unit direction vectors are computed an concatenated to build UDV.

With a smaller dataset available and faster convergence an better absolute translation prediction, onto the UDV inputs the median depth of the instance segmented point cloud can be appended. the instance segmention information comes from the mask head of the 2D estimator output.

The loss function combines pose errors, distance errors and joint relation constraints.

## 1. Data Generation
        
While running the carla simulator server in a docker container (see https://carla.readthedocs.io/en/latest/build_docker/), generate the sensor and metadata. File data_generation/open3d_lidar_cuboids_skeleton.py is ideally located locally in carla repo pythonapi/examples --> two carla repos needed: 1. docker container as the carla-server, 2.repo including the carla python api, from which the data generation script is executed. Make sure the carla versions of the docker container and the local one matches.

Following files are being created:
PCL (.PLY) from the xyz-values of rawdata, calib file with W2S matrix, metdadata from the CARLA world state (most importantly the CARLA skeleton for HPE)

raw data consists of ```(data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)])) )```

## 2. Custom CARLA skeleton
CARLA natively supports only its distinct skeleton representation, making it incompatible with direct transformation to popular skeleton representations like Human3.6M or SMPL. This work utilizes selected CARLA joints to construct a custom skeleton structure to avoid the complexities and potential errors associated with converting to another joint representation.

The specific CARLA joint names extracted are:
```BONE_KEYS = [
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
```

These joints are used to construct the connected pose represntation. The underlying joint connectons foming the "bones" is defined by:
```
KINTREE_TABLE = np.array([ 
    [0, 1],     # Pelvis -> R_Hip               # 0
    [1, 2],     # R_Hip -> R_Knee               # 1
    [2, 3],     # R_Knee -> R_Ankle             # 2
    [0, 4],     # Pelvis -> L_Hip               # 3
    [4, 5],     # L_Hip -> L_Knee               # 4
    [5, 6],     # L_Knee -> L_Ankle             # 5
    [0, 7],     # Pelvis -> Torso               # 6
    [7, 8],     # Torso -> Neck                 # 7
    [8, 9],     # Neck -> Head                  # 8
    [7, 10],    # Torso -> R_Shoulder           # 9       
    [10, 11],   # R_Shoulder -> R_Elbow         # 10
    [11, 12],   # R_Elbow -> R_Wrist            # 11
    [7, 13],    # Torso -> L_Shoulder           # 12
    [13, 14],   # L_Shoulder -> L_Elbow         # 13
    [14, 15]    # L_Elbow -> L_Wrist            # 14
]).T
```