# kpt_det_detectron

## 1. Data Generation
        
with carla python and running carla simulator client, generate data with open3d_lidar_cuboids_skeleton.py
file ideally located locally in carla repo pythonapi/examples

Following files are being created:
PCL (.PLY), calib file with W2S matrix, labels (additional information), 
raw data (data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)])) )

## 2. Create Detectron usable Dataset
image representation of the lidar scans are created (img) & annototations (labels) used for training in COCO format
save_dataset_detectron_format.py

## 3. Visualization demo 
keypoint_trainer.py


## Extras:
### Explanations:
Losses:

'loss_rpn_loc' is responsible for making the initial guesses about where objects are (the region proposals), and 'loss_box_reg' is responsible for refining those guesses into the final predictions. The first loss function helps the model know where to look, and the second one helps it accurately identify the objects in those locations.