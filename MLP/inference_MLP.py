from UDV_reconstruction import (SkeletonNetwork,
                                batch_generator, custom_loss_function, 
                                BATCH_SIZE, LAMBDA_D, LAMBDA_C, KINTREE_TABLE)
from utils.spherical import o3d_draw_skeleton
import torch
import glob
import numpy as np
import open3d as o3d
import time


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_path = "/workspace/data/dataset/temp/test/labels/*"
annotation_files = glob.glob(dataset_path)


# Defition of the MLP
model = SkeletonNetwork(input_dim=(16, 3), hidden_dim=512, output_dim=(16,3))
model.load_state_dict(torch.load('./MLP/weights/model_weights-231110_125915.pth', map_location=DEVICE))
model.to(DEVICE)

# After the training loop
model.eval()  # Set the model to evaluation mode

with torch.no_grad():  # Disable gradient computation
    for batch_idx, (keypoints_batch, keypoints_3d_batch) in \
            enumerate(batch_generator(annotation_files, batch_size=BATCH_SIZE, \
                                        DEVICE=DEVICE)):
        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(keypoints_batch)

        # Calculate the loss (optional, if you want to report loss on training data)
        custom_loss = custom_loss_function(outputs.view(outputs.size(0), 16, 3), \
                                        keypoints_3d_batch.view(outputs.size(0), 16, 3), \
                                        distance_metric= 'mae', lambda_d=LAMBDA_D, lambda_c=LAMBDA_C)

        # Calculate MPJPE
        outputs_np = outputs.clone().to('cpu').numpy().reshape(-1, 16, 3)
        gt_np = keypoints_3d_batch.clone().to('cpu').numpy().reshape(-1, 16, 3)
        MPJPE = np.mean(np.mean(np.linalg.norm(outputs_np - gt_np, axis=-1), axis=0))

        # Here, you can log the metrics or print them out as needed
        print(f"Eval Batch: {batch_idx+1} | Custom Loss: {float(custom_loss)} | MPJPE: {MPJPE}")

        line_set_pred, joints_pred = o3d_draw_skeleton(outputs_np[0], KINTREE_TABLE)
        line_set_gt, joints_gt = o3d_draw_skeleton(gt_np[0], KINTREE_TABLE, color_set="CMY")
        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1) # x: red, y: green, z: blue
        # o3d.visualization.draw_geometries([coord_mesh, line_set_pred, joints_pred, line_set_gt, joints_gt])

        # Create a visualizer object
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add geometries to the visualizer
        vis.add_geometry(coord_mesh)
        vis.add_geometry(line_set_pred)
        vis.add_geometry(joints_pred)
        vis.add_geometry(line_set_gt)
        vis.add_geometry(joints_gt)

        # Display the window for a certain amount of time (e.g., 5 seconds)
        end_time = time.time() + 5  # 5 seconds from now
        while time.time() < end_time:
            vis.poll_events()
            vis.update_renderer()

        # Destroy the visualizer window
        vis.destroy_window()