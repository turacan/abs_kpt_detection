r'''
Model Inputs:
    -   Unit Direction Vectors (UDVs): These are derived from the 2D keypoints using the camera's projection matrix to calculate
        the angles theta and phi. They represent the direction from the camera to the point in 3D space.
    -   Median Range Value: This is the median distance of the segmented object's point cloud from the camera. 
        It provides a reference distance that can be used to normalize the ranges you're trying to predict, 
        potentially making the learning task easier.

Model Outputs:
Range Residuals: Instead of predicting the absolute range, the model could predict a residual value that, 
    when added to the median range value, gives the final estimated range to the joint. 
    This could help the model focus on learning the variation in range relative to a known baseline (the median range), 
    which might be easier than learning the absolute range from scratch.


  Using the median or mean range value of the point cloud to center your keypoints could be a valid approach  

  at inference to recover true 3D joint locations
predicted_output_denorm = (output_model* std) + mean
predicted_output_recentered = predicted_output_denorm + median_pcl_loc

some entries of ˜x are not observed due to
joint occlusions or mis-detections. In order to not to change
the dimension of ˜x, the entries corresponding to these non-
observed joints will be set to zero.

# total of 8555 poses
'''

from typing import Any
import numpy as np
import sys
sys.path.append('/workspace/repos/')
import glob
import json
import open3d as o3d
import cv2
import datetime
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from save_dataset_detectron_format import calculate_projection_matrix, to_deflection_coordinates
from utils.spherical import o3d_draw_skeleton


# Constants definition 

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

def get_activation(layer, input, output):
    global activation
    activation = output.detach()

class SkeletonNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SkeletonNetwork, self).__init__()

        # Flatten the input and output dimensions
        input_dim = input_dim[0] * input_dim[1]
        output_dim = output_dim[0] * output_dim[1]
        
        # identity connection
        self.project_identity = nn.Linear(input_dim, hidden_dim)

        # First Linear layer with BatchNorm, SELU and Dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.selu1 = nn.SELU()
        #self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second Linear layer with BatchNorm, SELU and Dropout
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.selu2 = nn.SELU()
        #self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.25)
        
        # Third block (optional expansion)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim * 2)  # Example of making the network wider
        # self.norm3 = nn.LayerNorm(hidden_dim * 2)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.dropout3 = nn.Dropout(0.25)

        # # Third Linear layer with SELU and Dropout
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.selu3 = nn.SELU()
        self.dropout3 = nn.Dropout(0.25)
        
        # Fourth Linear layer for output
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # First block
        x = x.view(x.size(0), -1)  # Flatten the input
        # First block
        identity1 = self.project_identity(x)
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.selu1(out)
        out = self.dropout1(out)
        #out += identity1    # with identity connection
        
        # Second block
        identity2 = out
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.selu2(out)
        out = self.dropout2(out)
        #out += identity2    # with identity connection
        out += identity1    # with identity connection from the 2D input
        
        # Third block
        out = self.fc3(out)
        #out = self.norm3(out)
        out = self.selu3(out)
        out = self.dropout3(out)
        
        # Output layer
        out = self.fc4(out)
        return out
    

IMG_HEIGHT = int(1024)  
IMG_WIDTH = int(2048)
FOV_DEGREES = int(180) 
K, FOV_RADIANS = calculate_projection_matrix(IMG_HEIGHT, IMG_WIDTH, FOV_DEGREES)


def transform_kpts_to_unit_ray(kpts_uv: np.ndarray, proj_mat: np.ndarray=K):
    # Add an extra dimension to kpts_arr and K
    temp_kpt = np.column_stack((kpts_uv, np.ones(shape=(kpts_uv.shape[0],))))#.T
    
    K_inv = np.linalg.inv(proj_mat)

    phi_theta = np.matmul(K_inv, temp_kpt.T).T[:, :-1]
    phi = phi_theta[:, 0]
    theta = phi_theta[:, 1]

    ux = np.sin(np.pi/2-theta) * np.cos(phi) 
    uy = np.sin(np.pi/2-theta) * np.sin(phi)
    uz = np.cos(np.pi/2-theta)

    # Stack the coordinates to form the 3D points
    udvs = np.column_stack((ux, uy, uz))

    # Normalize the vectors to ensure they are unit vectors, range -1 to 1
    udvs /= np.linalg.norm(udvs, axis=1, keepdims=True)

    return udvs.astype(np.float32)

class Update_normalization_parameters():
    def __init__(self, fc) -> None:
        self.get_annotations_fc = fc
        self.aggregate = (0, 0.0, 0.0)

    def update(self, existingAggregate, newValue):
        (count, mean, M2) = existingAggregate
        count += 1 
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2

        return (count, mean, M2)

    def finalize(self, existingAggregate):
        (count, mean, M2) = existingAggregate
        if count < 2:
            return float('nan')
        else:
            (mean, variance) = (mean, M2 / count)
            return (mean, np.sqrt(variance))

    def __call__(self, outfile: str = None) -> Any:
        assert outfile != None, "Missing output file path"
        for (_, keypoints_3d, segm_pcl) in self.get_annotations_fc(annotation_files):
            kpts_3d = np.array(keypoints_3d, dtype=np.float32).reshape(-1, 4)[..., :-1]
            segm_pcl_np = np.array(segm_pcl, dtype=np.float32).reshape(-1, 5)[..., 2:]
            
            # Step 1: Centering pelvis to be at median pcl location
            median_pcl_loc = np.median(segm_pcl_np, axis=0)
            centralized_kpts_3d = kpts_3d - median_pcl_loc

            # Step 2: Update mean and std incrementally
            for value in centralized_kpts_3d:
                self.aggregate = self.update(self.aggregate, value)

        # Finalize to get the mean and std deviation
        (mean, std) = self.finalize(self.aggregate)
        
        np.savez(outfile, mean=mean.astype(np.float32), std=std.astype(np.float32))


def get_gt_annoations(annotation_files: list):
    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)['annotations']
            for annotation in annotations:
                yield annotation["keypoints"], annotation["keypoints_3d"], annotation['segm_pcl']

def batch_generator(annotation_files: list, batch_size: int, dset_mean: np.ndarray= None, dset_std: np.ndarray = None, DEVICE: str = 'cpu'):
    """Generator that yields batches of keypoints and keypoints_3d."""
    keypoints_batch, keypoints_3d_batch = [], []
    for keypoints_2d, keypoints_3d, segm_pcl in get_gt_annoations(annotation_files):
        # form label/ gt outputs
        kpts_3d = np.array(keypoints_3d, dtype=np.float32).reshape(-1, 4)[..., :-1]

        segm_pcl_np = np.array(segm_pcl, dtype=np.float32).reshape(-1, 5)[..., 2:]
        median_pcl_loc = np.median(segm_pcl_np, axis=0)
        centralized_kpts_3d = kpts_3d - kpts_3d[0] # median_pcl_loc
        # TESTING: don't use scaling
        #normalized_kpts = (centralized_kpts_3d - dset_mean) / dset_std 
        normalized_kpts = centralized_kpts_3d

        # form input
        kpts_uv = np.array(keypoints_2d, dtype=np.float32).reshape(-1, 3)[..., :-1]
        kpts_ray_unit = transform_kpts_to_unit_ray(kpts_uv)
        
        keypoints_batch.append(kpts_ray_unit)   # input
        keypoints_3d_batch.append(normalized_kpts)  # output label
        if len(keypoints_batch) == batch_size:
            yield torch.tensor(keypoints_batch, dtype=torch.float32).view(len(keypoints_batch), -1).to(DEVICE), torch.tensor(keypoints_3d_batch, dtype=torch.float32).view(len(keypoints_batch), -1).to(DEVICE)
            keypoints_batch, keypoints_3d_batch = [], []
    if keypoints_batch:  # handle the last batch which might be smaller than batch_size
        yield torch.tensor(keypoints_batch, dtype=torch.float32).view(len(keypoints_batch), -1).to(DEVICE), torch.tensor(keypoints_3d_batch, dtype=torch.float32).view(len(keypoints_batch), -1).to(DEVICE)


def custom_loss_function(preds, targets, distance_metric: str= 'mae', lambda_d=1.0, lambda_c=1.0):
    # Assuming preds and targets are of shape [batch_size, num_keypoints, 3]
    # where the last dimension corresponds to the (x, y, z) coordinates

    # Mean Squared Error for magnitude
    if distance_metric == 'mse':
        distance_loss = torch.nn.functional.mse_loss(preds, targets, reduction='mean')    # L2-loss, MSE
    elif distance_metric == 'mae':
        distance_loss = torch.nn.L1Loss()(preds, targets)                 # L1-loss, MAE

    # Normalize the predictions and the targets to unit vectors
    preds_normalized = torch.nn.functional.normalize(preds, p=2, dim=-1)
    targets_normalized = torch.nn.functional.normalize(targets, p=2, dim=-1)
    # Use mean to aggregate over all dimensions
    cosine_similarity_loss = (1 - torch.nn.functional.cosine_similarity(preds_normalized, targets_normalized, dim=-1)).mean()

    # Combine the losses, with a possible weighting scheme
    loss = lambda_d * distance_loss + lambda_c * cosine_similarity_loss

    return loss


def closest_point_on_ray(point_cloud, direction_vector):
    # Ensure the direction vector is normalized
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Project each point in the point cloud onto the ray
    projections = (np.dot(point_cloud, direction_vector)[:, np.newaxis]) * direction_vector

    # Calculate distances from original points to their projections
    distances = np.linalg.norm(point_cloud - projections, axis=1)

    # Find the index of the closest point
    closest_point_index = np.argmin(distances)

    # Return the closest point on the ray
    return projections[closest_point_index]

## Global variable definitions
BATCH_SIZE = 32
NUM_EPOCHS = 200
# loss fucntion lambda factors: tune wether focus is on localization (lamda_d) or rotation error (lamda_c)
LAMBDA_D, LAMBDA_C = 1.0, 0.2

def boxplot_hips_distance(annotation_files: str):
    distance_hips = []
    distance_pelvis_R_Hip = []
    distance_pelvis_L_Hip = []
    for keypoints_2d, keypoints_3d, segm_pcl in get_gt_annoations(annotation_files):
        kpts_3d = np.array(keypoints_3d, dtype=np.float32).reshape(-1, 4)[..., :-1] # exclude visibility flag
        kpts_2d = np.array(keypoints_2d, dtype=np.float32).reshape(-1, 3)[..., :-1] # exclude visibility flag
        
        kpts_ray_unit = transform_kpts_to_unit_ray(kpts_2d)
        root_distance = np.linalg.norm(kpts_3d[0])

        # Assuming kpts_3d is an array of shape (N, 3) where N is the number of joints
        root_joint = kpts_3d[0]
        relative_distance_vector = np.linalg.norm(kpts_3d - root_joint, axis=1)

        rho_i = np.ones(len(kpts_3d)) * root_distance + relative_distance_vector
        
        P_i = np.column_stack([rho_i, rho_i, rho_i]) * kpts_ray_unit
        # # 1. Convert pixel coordinates to spherical coordinates
        # K_inv = np.linalg.inv(K)
        # phi_theta = np.matmul(K_inv, np.column_stack((kpts_2d - np.array([[K[1, 2], K[0, 2]]]), np.ones(shape=(kpts_2d.shape[0],)))).T).T[:, :-1]
        # phi = phi_theta[:, 0]
        # theta = phi_theta[:, 1]

        # # # Compute radial distances (norms) of the 3D keypoints
        # # rho_i = np.linalg.norm(kpts_3d, axis=1)
        # # # Compute the 3D Cartesian coordinates from spherical coordinates
        # x_i = rho_i * np.sin(np.pi/2-theta) * np.cos(phi) 
        # y_i = rho_i * np.sin(np.pi/2-theta) * np.sin(phi)
        # z_i = rho_i * np.cos(np.pi/2-theta)
        
        # # Combine the Cartesian coordinates to get the 3D points
        # P_i = np.column_stack((x_i, y_i, z_i))
        
        pose_cloud = o3d.geometry.PointCloud()
        pose_cloud.points = o3d.utility.Vector3dVector(P_i)

        gt_cloud = o3d.geometry.PointCloud()
        gt_cloud.points = o3d.utility.Vector3dVector(kpts_3d)
        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1) # x: red, y: green, z: blue
        o3d.visualization.draw_geometries([pose_cloud, gt_cloud, coord_mesh])
        # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('test', width=1600, height=800)
        # cv2.imshow('test', test)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        continue
        joints_vectors_list = []
        for connection in KINTREE_TABLE.T:
            joints_vectors_list.append(kpts_2d[connection[1]] - kpts_2d[connection[0]]) # 2D vector of joint connections

        joints_vectors_arr = np.array(joints_vectors_list, dtype=np.float32)
        joints_vectors_arr_norm = joints_vectors_arr/np.linalg.norm(joints_vectors_arr, axis=1)[:, np.newaxis]

        '''
        next steps:
        idea: define a mean pose and learn the residual rotation of the joints to the mean pose

        concatenate UDV or 2D nroamlized vectors with ...
        ...vector between the important joint connections, needs 3 joints (i-->j, j-->k)
        [
         (0,1), (1,2),
         (1,2), (2,3),
         (0,4), (4,5),
         (4,5), (5,6),

         (0,1), (0,4),

         (0,7), (7,10)
         (7,10), (10,11),
         (10,11), (11,12),

         (0,7), (7,13)
         (7,13), (13,14),
         (13,14), (14,15),
         (7,10), (10,11),
         (7,13), (13,14)
         ]
        '''
        


        






    #     distance_hips.append(np.linalg.norm(kpts_3d[1] - kpts_3d[4]))
    #     distance_pelvis_R_Hip.append(np.linalg.norm(kpts_3d[0] - kpts_3d[4]))
    #     distance_pelvis_L_Hip.append(np.linalg.norm(kpts_3d[0] - kpts_3d[1]))
        
    # # Create histogram data
    # hist, bins = np.histogram(distance_hips, bins='auto')
    # # Plot histogram
    # fig = plt.figure('1')
    # plt.hist(distance_hips, bins='auto', alpha=0.7, color='blue')
    # plt.title('Histogram')
    # plt.xlabel('Data values')
    # plt.ylabel('Frequency')

    # fig = plt.figure('2')
    # plt.hist(distance_pelvis_R_Hip, bins='auto', alpha=0.7, color='blue')
    # plt.title('Histogram')
    # plt.xlabel('Data values')
    # plt.ylabel('Frequency')

    # fig = plt.figure('3')
    # plt.hist(distance_pelvis_L_Hip, bins='auto', alpha=0.7, color='blue')
    # plt.title('Histogram')
    # plt.xlabel('Data values')
    # plt.ylabel('Frequency')

    # # Show the plot
    # plt.show()




if __name__ == "__main__":
    # Load Dataset
    dataset_path = "/workspace/data/dataset/temp/train/labels/*"
    annotation_files = glob.glob(dataset_path)

    update_parameters = False   # flag to save mean and std of whole dataset, needs to be set true if new train data is acquired
    train_model = False
    pretrained = True
    training_start_time = datetime.datetime.now()
    training_start_time = training_start_time.strftime("%y%m%d_%H%M%S")

    if update_parameters:
        get_normalization_parameters = Update_normalization_parameters(get_gt_annoations)
        get_normalization_parameters(outfile="/workspace/repos/MLP/dataset_mean_std.npz")
    else:
        with np.load("/workspace/repos/MLP/dataset_mean_std.npz") as data:
            dset_mean = data['mean']
            dset_std = data['std']

    
    if train_model:
        # Define NUM_EPOCHS
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Defition of the MLP
        model = SkeletonNetwork(input_dim=(16, 3), hidden_dim=512, output_dim=(16,3))
        if pretrained:
            model.load_state_dict(torch.load('./MLP/weights/model_weights-231108_125549.pth', map_location=DEVICE))
        model.to(DEVICE)

        # Initialize a writer
        writer = SummaryWriter('./MLP/runs/experiment_6')

        # Register hook for the ReLU layer
        hook = model.selu2.register_forward_hook(get_activation)

        # Define the Huber Loss
        criterion = nn.HuberLoss()  # nn.HuberLoss()

        # Define the optimizer (using SGD) and scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',      # Minimize the monitored quantity
            factor=0.5,      # Reduce to 10% of current LR
            patience=10,     # Number of epochs with no improvement after which learning rate will be reduced
            threshold=0.01,  # Threshold for measuring the new optimum
            cooldown=5,      # Number of epochs to wait before resuming normal operation
            min_lr=1e-6,     # Minimum allowed learning rate
            eps=1e-8,        # Minimal decay to avoid updates that are too small
            verbose=True     # Print message to stdout on LR update
        )
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.3 * NUM_EPOCHS), int(0.8 * NUM_EPOCHS)], gamma=0.1)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        # Create the dataset and data loader
        for epoch in range(NUM_EPOCHS):
            model.train()
            for batch_idx, (keypoints_batch, keypoints_3d_batch) in \
                    enumerate(batch_generator(annotation_files, batch_size=BATCH_SIZE, dset_mean=dset_mean, \
                                              dset_std=dset_std, DEVICE=DEVICE)):
                # Reset gradients for this batch
                optimizer.zero_grad()

                # Forward pass: compute predicted outputs by passing inputs to the model
                outputs = model(keypoints_batch)

                # Calculate the loss
                loss = criterion(outputs, keypoints_3d_batch)   # Huberloss
                custom_loss = custom_loss_function(outputs.view(outputs.size(0), 16, 3), \
                                                   keypoints_3d_batch.view(outputs.size(0), 16, 3), \
                                                   distance_metric= 'mae', lambda_d=LAMBDA_D, lambda_c=LAMBDA_C)
                
                # Calculate MPJPE
                # .detach(): removes the tensor from the computation graph, so it no longer requires gradients. 
                # This is important because you cannot convert a tensor that requires gradients to a NumPy array directly.
                outputs_np = outputs.clone().detach().to('cpu').numpy().reshape(-1, 16, 3)
                gt_np = keypoints_3d_batch.clone().detach().to('cpu').numpy().reshape(-1, 16, 3)

                MPJPE = np.mean(np.mean(np.linalg.norm(outputs_np - gt_np, axis=-1), axis=0))
                
                # Log the loss value to TensorBoard
                writer.add_scalar('Loss/train/HuberLoss', loss.item(), epoch * len(annotation_files) / BATCH_SIZE + batch_idx)
                writer.add_scalar('Loss/train/Custom', float(custom_loss), epoch * len(annotation_files) / BATCH_SIZE + batch_idx)
                writer.add_scalar('MPJPE/train', MPJPE, epoch * len(annotation_files) / BATCH_SIZE + batch_idx)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch * len(annotation_files) / BATCH_SIZE + batch_idx)
                #writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch * len(annotation_files) / BATCH_SIZE + batch_idx)

                if batch_idx%10== 0:
                    #print(f"Epoch: {epoch+1}/{NUM_EPOCHS} | Batch: {batch_idx+1} | Loss: {loss.item():.4f} | MPJPE: {MPJPE}" , end='\r', flush=True)
                    print(f"Epoch: {epoch+1}/{NUM_EPOCHS} | LR: {optimizer.param_groups[0]['lr']} | Batch: {batch_idx+1} | Loss: {loss.item():.4f} | Custom Loss: {float(custom_loss)} | MPJPE: {MPJPE}" , end='\r', flush=True)

                # Backward pass: compute gradient of the loss with respect to model parameters
                custom_loss.backward()

                # Perform a single optimization step (parameter update)
                # updates the parameters of the model. It applies the gradients that were computed during loss.backward() to the parameters. 
                # This is the step where the optimizer's current learning rate is used to adjust the weights.
                optimizer.step()

                # Now `activation` contains the output of the layer you registered the hook on
                # Check for dead neurons
                dead_neurons = torch.nonzero((activation == 0).all(dim=0)).shape[0]  # Checks if any neuron was always zero for all examples in the batch

            # Update the learning rate of the optimizer
            scheduler.step(custom_loss)

        # Don't forget to remove the hook if it's no longer needed
        hook.remove()
        # save training parameters    
        torch.save(model.state_dict(), f'./MLP/weights/model_weights-{training_start_time}.pth')

        # Close the writer when you're done using it
        writer.close()


        line_set_pred, joints_pred = o3d_draw_skeleton(outputs_np[0], KINTREE_TABLE)
        line_set_gt, joints_gt = o3d_draw_skeleton(gt_np[0], KINTREE_TABLE, color_set="CMY")
        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1) # x: red, y: green, z: blue
        o3d.visualization.draw_geometries([coord_mesh, line_set_pred, joints_pred, line_set_gt, joints_gt])
    
    else:
        # statistical measurment
        boxplot_hips_distance(annotation_files)


        # visualization
        for keypoints_2d, keypoints_3d, segm_pcl in get_gt_annoations(annotation_files):
            kpts_3d = np.array(keypoints_3d, dtype=np.float32).reshape(-1, 4)[..., :-1] # exclude visibility flag

            segm_pcl_np = np.array(segm_pcl, dtype=np.float32).reshape(-1, 5)[..., 2:]
            median_pcl_loc = np.median(segm_pcl_np, axis=0)
            print(f"fistance to sensor: {np.linalg.norm(median_pcl_loc)}")
            centralized_kpts_3d = kpts_3d - kpts_3d[0]
            normalized_kpts = centralized_kpts_3d #(centralized_kpts_3d - dset_mean) / dset_std

            predicted_output_denorm = (normalized_kpts * dset_std) + dset_mean
            predicted_output_recentered = predicted_output_denorm + median_pcl_loc

            kpts_uv = np.array(keypoints_2d, dtype=np.float32).reshape(-1, 3)[..., :-1]

            kpts_ray_unit = transform_kpts_to_unit_ray(kpts_uv) # calc udv of the 2D points

            closest_point = closest_point_on_ray(segm_pcl_np, kpts_ray_unit[0])
            closest_point[:-1] = median_pcl_loc[:-1]

            # Visualization
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # # Plot each UDV as a line from the origin
            # for u in kpts_ray_unit:
            #     ax.quiver(0, 0, 0, u[0], u[1], u[2], length=1.0)

            # # Set equal aspect ratio for all axes
            # ax.set_box_aspect([1,1,1])

            # plt.show()

            
            skeleton_cloud = o3d.geometry.PointCloud()
            skeleton_cloud.points = o3d.utility.Vector3dVector(kpts_3d)

            pose_cloud = o3d.geometry.PointCloud()
            pose_cloud.points = o3d.utility.Vector3dVector(segm_pcl_np)
            color = [0, 0, 0] # Assigning black color to all points
            colors = np.tile(color, (segm_pcl_np.shape[0], 1))
            pose_cloud.colors = o3d.utility.Vector3dVector(colors)

            closest_point_cloud = o3d.geometry.PointCloud()
            closest_point_cloud.points = o3d.utility.Vector3dVector(closest_point[np.newaxis, ...])
            o3d.visualization.draw_geometries([pose_cloud, closest_point_cloud, skeleton_cloud])

            kpts_ray_unit = kpts_ray_unit * np.column_stack([np.linalg.norm(kpts_3d, axis=-1), np.linalg.norm(kpts_3d, axis=-1), np.linalg.norm(kpts_3d, axis=-1)])
            line_set_gt, joints_gt = o3d_draw_skeleton(normalized_kpts, KINTREE_TABLE, color_set="CMY") 
            line_set_2D, joints_2D = o3d_draw_skeleton(kpts_ray_unit, KINTREE_TABLE)
            coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1) # x: red, y: green, z: blue
            #o3d.visualization.draw_geometries([coord_mesh, line_set_2D, joints_2D])
            o3d.visualization.draw_geometries([coord_mesh, line_set_gt, joints_gt, line_set_2D, joints_2D])
            


        




        



    

