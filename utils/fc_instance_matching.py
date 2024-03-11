
from scipy.optimize import linear_sum_assignment
import numpy as np

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