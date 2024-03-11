import glob
import numpy as np
import open3d as o3d
import copy
import json
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('/workspace/repos/')

from utils.spherical import o3d_draw_skeleton
from utils.fc_matrix_and_projection_calculation import calculate_projection_matrix, transform_kpts_to_unit_ray


# connection rules of above defined skeleton, indluding the filtered joints
KINTREE_TABLE = np.array([ 
    [0, 1],     # Pelvis -> R_Hip           b0
    [1, 2],     # R_Hip -> R_Knee           b1
    [2, 3],     # R_Knee -> R_Ankle         b2  
    [0, 4],     # Pelvis -> L_Hip           b3
    [4, 5],     # L_Hip -> L_Knee           b4
    [5, 6],     # L_Knee -> L_Ankle         b5
    [0, 7],     # Pelvis -> Torso           b6
    [7, 8],     # Torso -> Neck             b7
    [8, 9],     # Neck -> Head              b8
    [7, 10],    # Torso -> R_Shoulder       b10
    [10, 11],   # R_Shoulder -> R_Elbow     b11
    [11, 12],   # R_Elbow -> R_Wrist        b12
    [7, 13],    # Torso -> L_Shoulder       b13
    [13, 14],   # L_Shoulder -> L_Elbow     b14
    [14, 15]    # L_Elbow -> L_Wrist        b15
]).T
HIGH_DOF_BONES = [[[0, 1, 2],      # right leg
                    [3, 4, 5]],      # left leg
                    [[9, 10, 11],    # right arm
                    [12, 13, 14]]    # left arm
]


class ResidualBlock_old(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.25):
        super(ResidualBlock_old, self).__init__()

        # Modules with Trainable Parameters
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.BatchNorm1d(hidden_size)

        # Parameter-Free Module, instance can be used multiple times in forward pass
        # Needs to be assined here to apply customizable drop rate at instance creation
        self.dropout = nn.Dropout(p=dropout_rate)  
    
    def forward(self, x):
        identity = x

        # Apply the first set of layers: Linear -> BatchNorm -> ReLU -> Dropout
        x = F.relu(self.norm1(self.fc1(x)))
        x = self.dropout(x)

        # Apply the second set of layers: Linear -> BatchNorm -> ReLU -> Dropout
        x = F.relu(self.norm2(self.fc2(x)))
        x = self.dropout(x)

        # Add the identity (skip connection)
        x += identity

        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()

        # Modules with Trainable Parameters
        
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout_rate)  
        self.norm2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout_rate)  

        # Parameter-Free Module, instance can be used multiple times in forward pass
        # Needs to be assined here to apply customizable drop rate at instance creation
        #self.dropout = nn.Dropout(p=dropout_rate)  
    
    def forward(self, x):
        identity = x

        # Apply the first set of layers (full pre-activation): BatchNorm -> ReLU -> Dropout -> Linear
        x = F.relu(self.norm1(x))
        x = self.dropout1(x)
        x = self.fc1(x)

        # Apply the second set of layers (full pre-activation): BatchNorm -> ReLU -> Dropout -> Linear
        x = F.relu(self.norm2(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        # Add the identity (skip connection)
        x += identity

        return x


class LinearBNReLUDropout(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.5):
        super(LinearBNReLUDropout, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        return self.seq(x)


# Define the Relative Distance Prediction Network
class RelativeDistancePredictionNetwork(nn.Module):
    '''
    Predict the root-relative distance, i.e., relative distances. 
    Input: flattened unit direction vectors (UDVs). 
    GT: centered and flattened 3D pose 
    '''
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(RelativeDistancePredictionNetwork, self).__init__()
        # input linear layer
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Residual Block
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size)
        )
        self.block2 = nn.Sequential(
            LinearBNReLUDropout(hidden_size, hidden_size),
            LinearBNReLUDropout(hidden_size, hidden_size//2),
            LinearBNReLUDropout(hidden_size//2, hidden_size//(2**2)),
            LinearBNReLUDropout(hidden_size//(2**2), hidden_size//(2**3))
        )
        self.fc2 = nn.Linear(hidden_size//(2**3), hidden_size//(2**3))  # optional
        self.output_layer = nn.Linear(hidden_size//(2**3), output_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.res_blocks(x)
        x = self.block2(x)
        x = F.relu(self.fc2(x))             # optional
        x = self.output_layer(x)
        return x


class RootDistancePredictionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(RootDistancePredictionNetwork, self).__init__()
        # input linear layer
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Residual Block
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size)
        )

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.output_layer = nn.Linear(hidden_size//2, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # double check if input is flattened
        x = F.relu(self.fc1(x))
        x = self.res_blocks(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.elu(self.output_layer(x))+1
        return x


# Define the Correction Network
class CorrectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CorrectionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.residual_block = nn.Sequential(
            ResidualBlock(hidden_size, dropout_rate=0.2),
            ResidualBlock(hidden_size, dropout_rate=0.2)
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # double check if input is flattened
        x = F.relu(self.fc1(x))
        residual = x
        x = self.residual_block(x)
        x += residual
        x = self.output_layer(x)
        return x


class IntegratedPoseEstimationModel(nn.Module):
    def __init__(self, relative_model, root_network, correction_model):
        super(IntegratedPoseEstimationModel, self).__init__()
        self.relative_model = relative_model
        self.root_network = root_network
        #self.correction_model = correction_model

    def forward(self, x):
        x = x.view(x.size(0), -1)
        relative_distances = self.relative_model(x)     # input size: (batch_size, #joints * 3); output size (batch_size, #joints)
        root_distance = self.root_network(x)            # input size: (batch_size, #joints * 3); output size(batch_size, 1)
        #root_distance = torch.abs(root_distance)
        rho_i = torch.ones_like(relative_distances) * root_distance + relative_distances
        if x.size(1) %2 == 0: # NO depth in input
            combined_output = torch.column_stack([rho_i, rho_i, rho_i]) * x    
        else: # with depth information
            combined_output = torch.column_stack([rho_i, rho_i, rho_i]) * x[:, :-1]
            
        # Get the corrected pose from the correction network
        #corrected_pose = self.correction_model(combined_output)
        
        return combined_output, root_distance#corrected_pose


class CustomDataset(Dataset):
    def __init__(self, annotation_files, transform=None, pred_input_files=None, input_w_depth=True, w_gt_Data=True):
        self.annotation_files = annotation_files
        self.transform = transform  # converts 2D pose from image coordinates to unit direction vectors (UDV)
        self.pred_input_files = pred_input_files
        self.input_w_depth = input_w_depth
        self.w_gt_Data = w_gt_Data
        
        self.index = self._index_annotations()

    def _index_annotations(self):
        # Create an index for each annotation within each file
        index = []
        flag_anno_type = 0  # indicates if anno data is gt==0, or origins form prediction -> ==1
        if self.w_gt_Data:
            for file_idx, annotation_file in enumerate(self.annotation_files):
                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)['annotations']
                    for ann_idx in range(len(annotations)):
                        index.append((file_idx, ann_idx, flag_anno_type))    

        if self.pred_input_files:
            flag_anno_type = 1
            for file_idx, pred_input_file in enumerate(self.pred_input_files): 
                data = np.load(pred_input_file)
                for ann_idx in range(data['gt_idx'].shape[0]):
                    index.append((file_idx, ann_idx, flag_anno_type))    
                    
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if isinstance(self.index[idx], tuple):
            file_idx, ann_idx, flag_anno_type = self.index[idx]

        if not flag_anno_type:  # evaluates to normal annotation NOT pred labels
            annotation_file = self.annotation_files[file_idx]
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)['annotations'][ann_idx]

            # Handle 2D joints, calculate unit direction vectors --> input of the models
            keypoints_2d = annotation["keypoints"]
            kpts_2d = np.array(keypoints_2d, dtype=np.float32).reshape(16, -1)[:, :-1] # remove visibility flag
            keypoints_3d = annotation["keypoints_3d"]
            absolute_pose = np.array(keypoints_3d, dtype=np.float32).reshape(16, -1)[:, :-1] # remove visibility flag

            if self.input_w_depth:
                segm_pcl_distance = np.median(np.linalg.norm(np.array(annotation["segm_pcl"], dtype=np.float32)[:, 2:], axis=1))
            
            proj_matrix, _ = calculate_projection_matrix(height=1024, width=2048, fov_degrees=180)

        else:   # evaluates to pred_files
            data = np.load(self.pred_input_files[file_idx])
            kpts_2d = data['pred_kpts_2d'][ann_idx]
            absolute_pose = data['gt_kpts_3d'][ann_idx] # 3d pose

            if self.input_w_depth:
                segm_pcl_distance = np.median(np.linalg.norm(data["gt_kpts_3d"][ann_idx], axis=1))
                
            proj_matrix, _ = calculate_projection_matrix(height=256, width=2048, fov_degrees=22.5)

        if self.transform:
            udvs = self.transform(kpts_2d, proj_matrix) 
        else:
            udvs = kpts_2d
        # Handle 3D joints, gather scalar root distance and relative distances of the joints to root --> labels
        
        root_distance = np.linalg.norm(absolute_pose[0])    # absolute distance of root joint (pelvis) to sensor
        centralized_kpts_3d = absolute_pose - absolute_pose[0]  # zero-centering of the pose (pelvis is at origin)
        relative_distance_vector = np.linalg.norm(centralized_kpts_3d - centralized_kpts_3d[0], axis=1) # here uses the zero-centered pose

        if not self.input_w_depth:
            segm_pcl_distance=None
        ## comparing assempled parts with gt_3d pose
        # rho_i = np.ones(len(absolute_pose)) * root_distance + relative_distance_vector
        # P_i = np.column_stack([rho_i, rho_i, rho_i]) * udvs
        
        # line_set_gt, joints_gt = o3d_draw_skeleton(P_i, KINTREE_TABLE, color_set="CMY") 
        # coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1) # x: red, y: green, z: blue
        # o3d.visualization.draw_geometries([coord_mesh, line_set_gt, joints_gt])

        # returns dict of np.ndarrays, will be transformed to torch tensors with collate_fn
        return {
            "udvs": udvs,
            "relative_distances_gt": relative_distance_vector,
            "root_distance_gt": root_distance,
            "absolute_pose_gt": absolute_pose,
            "segm_pcl_distance": segm_pcl_distance
        }

from time import perf_counter_ns 
def collate_fn(batch):
    # Convert numpy arrays to PyTorch tensors and stack them flattened, here torch.from_numpy is not used to prevent dtype inconsistencies
    udvs_batch = torch.stack([torch.tensor(sample['udvs'], dtype=torch.float32).view(16*3) for sample in batch])
    relative_distances_gt_batch = torch.stack([torch.tensor(sample['relative_distances_gt'], dtype=torch.float32).view(16) for sample in batch])
    root_distance_gt_batch = torch.stack([torch.tensor(sample['root_distance_gt'], dtype=torch.float32).view(1) for sample in batch])
    absolute_pose_gt_batch = torch.stack([torch.tensor(sample['absolute_pose_gt'], dtype=torch.float32).view(16*3) for sample in batch])
    if batch[0]['segm_pcl_distance']:
        pcl_distance_batch = torch.stack([torch.tensor(sample['segm_pcl_distance'], dtype=torch.float32).view(1) for sample in batch])
    else:
        pcl_distance_batch = None

    # Return a dictionary of batches
    return {
        'udvs': udvs_batch,
        'relative_distances_gt': relative_distances_gt_batch,
        'root_distance_gt': root_distance_gt_batch,
        'absolute_pose_gt': absolute_pose_gt_batch,
        'pcl_distance': pcl_distance_batch
    }

def custom_loss_function(preds, targets, pred_root, lambda_d=None, lambda_p=None, lambda_c=None):
    '''
    Args: 
    preds and targets of shape [batch_size, num_keypoints, 3],
            where the last dimension corresponds to the (x, y, z) coordinates
    lambda coefficients indicate the control parameters  
    
    Output:
    loss function that combines the pose errors L_p, distance errors L_d, joint relation constraints L_c
    '''
    
    global KINTREE_TABLE
    global HIGH_DOF_BONES
    
    # Joints Distance Loss
        # measures the errors between the predicted joint distances (Euclidean) rho^ and label distances rho, i.e. MAE
    distance_loss = torch.nn.L1Loss(reduction='mean')(torch.linalg.norm(preds, ord=2, dim=-1), torch.linalg.norm(targets, ord=2, dim=-1))

    # Pose Loss
        #  measures errors between the predicted 3D poses P^ and the label 3D poses P (Euclidean distances), i.e. MAE
    pose_loss = torch.nn.L1Loss(reduction='mean')(preds, targets)
    #pose_loss = torch.nn.L1Loss(reduction='mean')(preds-preds[:,0].unsqueeze(1), targets-targets[:,0].unsqueeze(1))

    # Joint relation constraints
    idxs_right_first = KINTREE_TABLE[0, (np.transpose(HIGH_DOF_BONES, (0, 2, 1)))[..., 0].flatten()]    # right body half, first joint
    idxs_right_second = KINTREE_TABLE[1, (np.transpose(HIGH_DOF_BONES, (0, 2, 1)))[..., 0].flatten()]   # right body half, second joint

    idxs_left_first = KINTREE_TABLE[0, (np.transpose(HIGH_DOF_BONES, (0, 2, 1)))[..., 1].flatten()]     # left body half, first joint
    idxs_left_second = KINTREE_TABLE[1, (np.transpose(HIGH_DOF_BONES, (0, 2, 1)))[..., 1].flatten()]    # left body half, second joint
    
        # Body/Bone length symmetry
    loss_bone_length_sym = torch.nn.L1Loss(reduction='mean') \
                                (torch.linalg.norm((preds[:, idxs_right_second] - preds[:, idxs_right_first]), ord=2, dim=-1),
                                torch.linalg.norm((preds[:, idxs_left_second] - preds[:, idxs_left_first]), ord=2, dim=-1))

    pred_udv_right = preds[:, list(idxs_right_second)+list(idxs_left_second)] / torch.linalg.norm(preds[:, list(idxs_right_second)+list(idxs_left_second)], ord=2, dim=-1, keepdims=True) \
        - preds[:, list(idxs_right_first)+list(idxs_left_first)] / torch.linalg.norm(preds[:, list(idxs_right_first)+list(idxs_left_first)], ord=2, dim=-1, keepdims=True)
    
    gt_udv_right = targets[:, list(idxs_right_second)+list(idxs_left_second)] / torch.linalg.norm(targets[:, list(idxs_right_second)+list(idxs_left_second)], ord=2, dim=-1, keepdims=True) \
        - targets[:, list(idxs_right_first)+list(idxs_left_first)] / torch.linalg.norm(targets[:, list(idxs_right_first)+list(idxs_left_first)], ord=2, dim=-1, keepdims=True)
    
        # Directional constraint of adjacent joints
    loss_directional_sym = torch.nn.L1Loss(reduction='mean')(pred_udv_right, gt_udv_right)

    loss_constraints = loss_bone_length_sym + loss_directional_sym

    # Feature consistency Loss when using predicted 2D inputs to become similar to those of the labeled inputs.
    root_distance_loss = torch.nn.L1Loss(reduction='mean')(pred_root, torch.linalg.norm(targets[:, 0], ord=2, dim=-1).unsqueeze(dim=1))
        # calculate accumaulated weighted su
    total_loss = (lambda_d * distance_loss + 1 * root_distance_loss ) + lambda_p * pose_loss + lambda_c * loss_constraints
    #total_loss = total_loss#.requires_grad_(True)
    return total_loss

import os
if __name__ == "__main__":
    save_weights = False
    pre_trained = True
    train_model = False
    if pre_trained:
        pre_trained_weights = "/workspace/repos/MLP/weights/model_weights-240117_143312-wdfinetuned.pth" #/workspace/repos/MLP/weights/model_weights-240111_121437-wd.pth'
        # "/workspace/repos/MLP/weights/model_weights-240111_122503-finetuned_wd.pth"

    input_w_depth = True
    w_pred_input = True 
    w_gt_Data = False

    # Define NUM_EPOCHS
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if train_model:
        NUM_EPOCHS = 10
    else:
        NUM_EPOCHS = 1

    # loss function weights
    LAMBDA_D, LAMBDA_P, LAMBDA_C = 1, 1, 1   # 1, 1, 0.25   # if pretrained? 1, 1, 1

    # define layer sizes, per instance not batch
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
    absolute_3d_joint_estimation_network.to(DEVICE)

    if pre_trained:
        pretrained_state_dict  = (torch.load(pre_trained_weights, map_location=DEVICE))

        for name, param in absolute_3d_joint_estimation_network.named_parameters():
            # If the parameter exists in the pre-trained model, load the weights
            param.data = pretrained_state_dict[name]

    # Define the optimizer (using SGD) and scheduler
    optimizer = torch.optim.Adam(absolute_3d_joint_estimation_network.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      # Minimize the monitored quantity
        factor=0.5,      # Reduce to 10% of current LR
        patience=1,     # Number of epochs with no improvement after which learning rate will be reduced
        threshold=0.0001,  # Threshold for measuring the new optimum
        cooldown=1,      # Number of epochs to wait before resuming normal operation
        min_lr=1e-6,     # Minimum allowed learning rate
        eps=1e-8,        # Minimal decay to avoid updates that are too small
        verbose=True     # Print message to stdout on LR update
    )


    # Load datasets' filenames
    if w_gt_Data:
        dataset_path = "/workspace/data/dataset/train/labels/*"
        annotation_files = glob.glob(dataset_path)
    else:
        annotation_files = None

    # with predicted files
    if w_pred_input:
        dataset_path = "/workspace/data/dataset/test/pred_2d/*"
        pred_input_files = glob.glob(dataset_path)
    else:
        pred_input_files = None

    dataset = CustomDataset(annotation_files, transform=transform_kpts_to_unit_ray, pred_input_files=pred_input_files, input_w_depth=input_w_depth, w_gt_Data=w_gt_Data)
    # important to use shuffle, bc if 2D predected keypoints are also considered, they will be appended to end as a block
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=4) # num_wprkers @ = 1GB VRAM
    
    training_start_time = datetime.datetime.now()
    training_start_time = training_start_time.strftime("%y%m%d_%H%M%S")
    if input_w_depth:
        suffix_name = "-wd"
    else:
        suffix_name = ''
    if w_pred_input and not w_gt_Data:
        suffix_name += "finetuned"
    if save_weights:
        os.makedirs("/workspace/repos/MLP/weights", exist_ok=True)
        out_filename_modelWeights = f'/workspace/repos/MLP/weights/model_weights-{training_start_time}{suffix_name}.pth'

    # Initialize a writer
    tensorboard_dir = '/workspace/repos/MLP/runs/root_relative_network/'
    if os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir, exist_ok=True)
    tensorboard_dir = os.path.join(tensorboard_dir, f"{training_start_time}{suffix_name}")
    writer = SummaryWriter(tensorboard_dir)

    # Training loop
    if not train_model:
        with torch.no_grad():
            for epoch in range(NUM_EPOCHS):
                # set main network to train mode; subnetworks will automatically inherit the train mode 
                if train_model:
                    absolute_3d_joint_estimation_network.train()
                else:
                    absolute_3d_joint_estimation_network.eval()
                for batch_idx, batch in enumerate(dataloader):
                    # Reset gradients for this batch
                    optimizer.zero_grad()

                    model_input = batch['udvs'].to(DEVICE)  # UDVs
                    if input_w_depth:
                        model_input = torch.cat([model_input, batch['pcl_distance'].to(DEVICE)], dim=1)    # udvs and L2-distance
                    predicted_pose, predicted_root_distance = absolute_3d_joint_estimation_network(model_input)   
                    labels = batch['absolute_pose_gt'].clone().to(DEVICE).reshape(-1, 16, 3)

                    total_loss = custom_loss_function(predicted_pose.reshape(-1, 16, 3), labels, predicted_root_distance,
                                                        lambda_d=LAMBDA_D, lambda_p=LAMBDA_P, lambda_c=LAMBDA_C)

                    outputs_np = predicted_pose.clone().detach().to('cpu').numpy().reshape(-1, 16, 3)
                    gt_np = labels.clone().detach().to('cpu').numpy().reshape(-1, 16, 3)

                    MPJPE = np.mean(np.mean(np.linalg.norm(outputs_np - gt_np, axis=-1), axis=1))   # outer was axis=0

                    # Log the loss value to TensorBoard
                    writer.add_scalar('Loss/train/Custom', round(float(total_loss), 4), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar('MPJPE/train', MPJPE, epoch * len(dataloader) + batch_idx)
                    writer.add_scalar('Pelvis_dislocation/train', np.mean(np.linalg.norm(outputs_np - gt_np, axis=-1)[0]), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch * len(dataloader) + batch_idx)

                    # print current data on console
                    if batch_idx%1== 0:
                        print(f"Epoch: {epoch+1}/{NUM_EPOCHS} | LR: {optimizer.param_groups[0]['lr']} | Batch: {batch_idx+1} | Loss: {total_loss.item():.4f} | MPJPE: {MPJPE}" , end='\r', flush=True)

                    # Backward pass: compute gradient of the loss with respect to model parameters
                    if train_model:
                        total_loss.backward()

                    # Perform a single optimization step (parameter update)
                    # updates the parameters of the model. It applies the gradients that were computed during loss.backward() to the parameters. 
                    # This is the step where the optimizer's current learning rate is used to adjust the weights.
                    if train_model:
                        optimizer.step()

                # Update the learning rate of the optimizer
                if train_model:        
                    scheduler.step(total_loss)
    
    if save_weights:
        torch.save(absolute_3d_joint_estimation_network.state_dict(), out_filename_modelWeights)

    # Close the writer when you're done using it
    writer.close()

    line_set_gt, joints_gt = o3d_draw_skeleton(outputs_np[0], KINTREE_TABLE, color_set="CMY") 
    line_set_2D, joints_2D = o3d_draw_skeleton(gt_np[0], KINTREE_TABLE)
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1) # x: red, y: green, z: blue
    #o3d.visualization.draw_geometries([coord_mesh, line_set_2D, joints_2D])
    o3d.visualization.draw_geometries([coord_mesh, line_set_gt, joints_gt, line_set_2D, joints_2D])
    print("Program exit 0")


# # Define NUM_EPOCHS
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Load Dataset filenames
# dataset_path = "/workspace/data/dataset/temp/train/labels/*"
# annotation_files = glob.glob(dataset_path)


# for keypoints_2d, keypoints_3d, segm_pcl in get_gt_annoations(annotation_files):
#         kpts_3d = np.array(keypoints_3d, dtype=np.float32).reshape(-1, 4)[..., :-1] # exclude visibility flag
#         kpts_2d = np.array(keypoints_2d, dtype=np.float32).reshape(-1, 3)[..., :-1] # exclude visibility flag
        
#         kpts_ray_unit = transform_kpts_to_unit_ray(kpts_2d)
#         root_distance = np.linalg.norm(kpts_3d[0])

#         # Assuming kpts_3d is an array of shape (N, 3) where N is the number of joints
#         root_joint = kpts_3d[0]
#         relative_distance_vector = np.linalg.norm(kpts_3d - root_joint, axis=1)

#         rho_i = np.ones(len(kpts_3d)) * root_distance + relative_distance_vector
        
#         P_i = np.column_stack([rho_i, rho_i, rho_i]) * kpts_ray_unit