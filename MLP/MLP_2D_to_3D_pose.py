r'''
GOAL: estimate the body joint locations in the 3-dimensional space
To recover the 3D joint locations, we try to learn a direct 2D-to-3D mapping

This transformation can be implemented by a MLP in a supervised manner 
pose_3d = f(pose_2D, theta), where theta is a set of trainable parameters of function f

Optimization problem:
argmin_theta  [ 1/ #poses sum_{n=1}^{#poses} Huber Loss(f( x_i), y_i  )  ], 
    where x_i: input 2D poses, y_i: ground truth 3D poses
Latex: \underset{\theta}{\text{arg min}} \frac{1}{\mathcal{C}} \sum_{n=1}^{\mathcal{C}} \mathcal{L}\left(f_r(\mathbf{x}_i), \mathbf{y}_i\right)

    
Network design: 
image: https://www.ncbi.nlm.nih.gov/core/lw/2.0/html/tileshop_pmc/tileshop_pmc_inline.html?title=Click on image to zoom&p=PMC3&id=7180926_sensors-20-01825-g002.jpg

comprises linear layers, Batch Normalization (BN), Dropout, SELU and Identity connections.
              ____________________________________________________     ___________________________________________________
             |                                                    |   |                                                   |
[[2D input]] --> (Linear, 512) --> BN --> SELU --> (Dropout, 0.25) --> (Linear, 512) --> BN --> SELU --> (Dropout, 0.25) --> (Linear, 512) --> SELU --> (Dropout, 0.25) --> (Linear, 512) --> [[3D Output]]
             |____________________________________________________________________________________________________________|    

               
paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7180926/#sec3-sensors-20-01825
huber loss (pytorch): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import glob
import json
import tqdm
import open3d as o3d
import sys
sys.path.append('/workspace/repos/')

from utils.spherical import o3d_draw_skeleton
from save_dataset_detectron_format import calculate_projection_matrix, to_deflection_coordinates


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

# class CustomDataset(Dataset):
#     def __init__(self, annotation_files_path):
#         self.annotation_files = glob.glob(annotation_files_path)
        

#     def __len__(self):
#         return len(self.annotation_files)

#     def __getitem__(self, idx):
#         # Load the annotation file
#         with open(self.annotation_files[idx], 'r') as f:
#             annotations = json.load(f)['annotations'] 
        
#         # Extract keypoints and keypoints_3d
#         keypoints_list = []
#         keypoints_3d_list = []

#         for data in annotations:
#             keypoints_list.append(data["keypoints"])
#             keypoints_3d_list.append(data["keypoints_3d"])
        
#         return {
#             "keypoints": keypoints_list,
#             "keypoints_3d": keypoints_3d_list
#         }


def annotation_generator(annotation_files):
    """Generator that yields keypoints and keypoints_3d for each object in each annotation file."""
    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)['annotations']
            for annotation in annotations:
                yield annotation["keypoints"], annotation["keypoints_3d"]

def batch_generator(annotation_files, batch_size):
    """Generator that yields batches of keypoints and keypoints_3d."""
    keypoints_batch, keypoints_3d_batch = [], []
    for keypoints, keypoints_3d in annotation_generator(annotation_files):
        keypoints_batch.append(keypoints)
        keypoints_3d_batch.append(keypoints_3d)
        if len(keypoints_batch) == batch_size:
            yield torch.tensor(keypoints_batch, dtype=torch.float32).view(-1, 16, 3), torch.tensor(keypoints_3d_batch, dtype=torch.float32).view(-1, 16, 4)
            keypoints_batch, keypoints_3d_batch = [], []
    if keypoints_batch:  # handle the last batch which might be smaller than batch_size
        yield torch.tensor(keypoints_batch, dtype=torch.float32).view(-1, 16, 3), torch.tensor(keypoints_3d_batch, dtype=torch.float32).view(-1, 16, 4)


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
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.selu1 = nn.SELU()
        self.dropout1 = nn.Dropout(0.25)
        
        # Second Linear layer with BatchNorm, SELU and Dropout
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.selu2 = nn.SELU()
        self.dropout2 = nn.Dropout(0.25)
        
        # Third Linear layer with SELU and Dropout
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.selu3 = nn.SELU()
        self.dropout3 = nn.Dropout(0.25)
        
        # Fourth Linear layer for output
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # First block
        identity1 = self.project_identity(x)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.selu1(out)
        out = self.dropout1(out)
        out += identity1    # with identity connection
        
        # Second block
        identity2 = out
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.selu2(out)
        out = self.dropout2(out)
        out += identity2    # with identity connection
        out += identity1    # with identity connection from the 2D input
        
        # Third block
        out = self.fc3(out)
        out = self.selu3(out)
        out = self.dropout3(out)
        
        # Output layer
        out = self.fc4(out)
        return out

def main():
    # Define NUM_EPOCHS
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Defition of the MLP
    model = SkeletonNetwork(input_dim=(16, 2), hidden_dim=512, output_dim=(16,3)).to(DEVICE)

    # Define the Huber Loss
    criterion = nn.HuberLoss()

    # Define the optimizer (using SGD) and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.3 * NUM_EPOCHS), int(0.8 * NUM_EPOCHS)], gamma=0.1)

    # Initialize a writer
    writer = SummaryWriter('./MLP/runs/experiment_1')

    # Load Dataset
    dataset_path = "/workspace/data/dataset/temp/train/labels/*"
    annotation_files = glob.glob(dataset_path)

    # Create the dataset and data loader
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (keypoints_batch, keypoints_3d_batch) in enumerate(batch_generator(annotation_files, batch_size=BATCH_SIZE)):
            # Reset gradients for this batch
            optimizer.zero_grad()

            keypoints_batch_uv = keypoints_batch[..., :-1].clone().view(len(keypoints_batch), -1).to(DEVICE)
            keypoints_3d_batch_xyz = keypoints_3d_batch[..., :-1].clone().view(len(keypoints_batch), -1).to(DEVICE)

            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(keypoints_batch_uv)

            # Calculate the loss
            loss = criterion(outputs, keypoints_3d_batch_xyz)
            
            # Calculate MPJPE
            # .detach(): removes the tensor from the computation graph, so it no longer requires gradients. 
            # This is important because you cannot convert a tensor that requires gradients to a NumPy array directly.
            outputs_np = outputs.clone().detach().to('cpu').numpy() 
            gt_np = keypoints_3d_batch_xyz.clone().detach().to('cpu').numpy() 

            outputs_np = outputs_np.reshape(-1, 16, 3)
            gt_np  = gt_np.reshape(-1, 16, 3)

            MPJPE = np.mean(np.mean(np.linalg.norm(outputs_np - gt_np, axis=-1), axis=0))
              
            # Log the loss value to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(annotation_files) / BATCH_SIZE + batch_idx)
            writer.add_scalar('MPJPE/train', MPJPE, epoch * len(annotation_files) / BATCH_SIZE + batch_idx)

            if batch_idx%10== 0:
                print(f"Epoch: {epoch+1}/{NUM_EPOCHS} | Batch: {batch_idx+1} | Loss: {loss.item():.4f} | MPJPE: {MPJPE}" , end='\r', flush=True)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Perform a single optimization step (parameter update)
            # updates the parameters of the model. It applies the gradients that were computed during loss.backward() to the parameters. 
            # This is the step where the optimizer's current learning rate is used to adjust the weights.
            optimizer.step()

        # Update the learning rate of the optimizer
        scheduler.step()

    line_set_pred, joints_pred = o3d_draw_skeleton(outputs_np[0], KINTREE_TABLE)
    line_set_gt, joints_gt = o3d_draw_skeleton(gt_np[0], KINTREE_TABLE, color_set="CMY")
    o3d.visualization.draw_geometries([line_set_pred, joints_pred, line_set_gt, joints_gt])

    # save training parameters        
    torch.save(model.state_dict(), './MLP/model_weights.pth')

    # Close the writer when you're done using it
    writer.close()

if __name__ == "__main__":
    main()