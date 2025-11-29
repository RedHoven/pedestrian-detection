import numpy as np
import torch
from PIL import Image
import cv2
from typing import OrderedDict, Union, List, Tuple, Optional
from ultralytics import YOLO, RTDETR

class UltralyticsTargetLayers:
    """Helper class to get target layers from Ultralytics models"""
    @staticmethod
    def get_target_layers(model, model_type: str):
        if model_type.lower() == "yolov8":
            # For YOLOv8, we target the backbone feature extractor
            return [
                    model.model.model[9],  # SPFF
                    # model.model.model[15],  # High-res C2f
                    # model.model.model[18],  # Mid-res C2f
                    # model.model.model[21],  # Low-res C2f before Detect
                ] # Targeting a deep feature extraction layer
        elif model_type.lower() == "rtdetr":
            # For RT-DETR, we target the backbone feature extractor
            return [
                model.model.model[28].input_proj[0][0],
                model.model.model[28].input_proj[1][0],
                model.model.model[28].input_proj[2][0]
            ] 
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'yolov8' or 'rt-detr'")

class UltralyticsBoxScoreTarget:
    """Target for CAM that focuses on detected bounding boxes"""
    def __init__(self, boxes: np.ndarray, labels: np.ndarray, iou_threshold: float = 0.5):
        self.boxes = boxes
        self.labels = labels
        self.iou_threshold = iou_threshold
        
    def __call__(self, results):
        # For Ultralytics models, we need to work with the results format
        # This implementation focuses on the confidence scores of the detections
        # that match our original detections
        
        # Initialize output score
        output = torch.tensor([0.0])
        if torch.cuda.is_available():
            output = output.cuda()
            
        # If no detections, return zero score
        if len(results) == 0 or len(results[0].boxes) == 0:
            return output
        
        # Get the boxes from the results
        result_boxes = results[0].boxes.xyxy
        result_labels = results[0].boxes.cls
        result_scores = results[0].boxes.conf
        
        # Convert our boxes to tensor if they're not already
        boxes_tensor = torch.tensor(self.boxes, dtype=torch.float32)
        if torch.cuda.is_available():
            boxes_tensor = boxes_tensor.cuda()
            
        # For each of our original detected boxes
        for i, (box, label) in enumerate(zip(boxes_tensor, self.labels)):
            box = box.unsqueeze(0)  # Add batch dimension
            
            # Calculate IoU between this box and all result boxes
            ious = box_iou(box, result_boxes)
            
            if ious.numel() > 0:
                # Find the box with the highest IoU
                max_iou_idx = ious.argmax()
                max_iou = ious[0, max_iou_idx]
                
                # If IoU is above threshold and the label matches
                if max_iou > self.iou_threshold and result_labels[max_iou_idx] == label:
                    # Add the IoU plus the confidence score to our output
                    output = output + max_iou + result_scores[max_iou_idx]
            
        return output

def box_iou(box1, box2):
    """
    Compute IoU between bounding boxes
    
    Args:
        box1: tensor of shape (1, 4) - [x1, y1, x2, y2]
        box2: tensor of shape (n, 4) - [x1, y1, x2, y2]
        
    Returns:
        IoU tensor of shape (1, n)
    """
    # Calculate area of boxes
    area1 = (box1[0, 2] - box1[0, 0]) * (box1[0, 3] - box1[0, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Find the intersection coordinates
    inter_x1 = torch.max(box1[0, 0], box2[:, 0])
    inter_y1 = torch.max(box1[0, 1], box2[:, 1])
    inter_x2 = torch.min(box1[0, 2], box2[:, 2])
    inter_y2 = torch.min(box1[0, 3], box2[:, 3])
    
    # Calculate intersection area
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = inter_w * inter_h
    
    # Calculate IoU
    union = area1 + area2 - intersection
    iou = intersection / union
    
    return iou.unsqueeze(0)  # Shape (1, n)

def ultralytics_reshape_transform(features):
    """
    Reshape transform for Ultralytics models to prepare activations for CAM
    
    We aggregate features from different detection heads and reshape them for CAM
    """
    # For Ultralytics models, we need to extract and process the activations
    # This is a simplified version that works with most models
    activations = features[0]  # Take the first (main) activation tensor
    
    # Ensure proper shape for CAM processing
    if len(activations.shape) == 4:
        return activations
    else:
        # If activations have unexpected shape, reshape them
        # This might need to be adapted for specific model architectures
        return activations.unsqueeze(0)

def renormalize_cam_in_bounding_boxes(image, boxes, grayscale_cam):
    """
    Normalize the CAM to be in the range [0, 1] inside every bounding box,
    and zero outside of the bounding boxes.
    """
    # Create empty cam of same shape
    renormalized_cam = np.zeros_like(grayscale_cam, dtype=np.float32)
    images = []
    
    # For each bounding box
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Create a new image with zeros
        img = np.zeros_like(renormalized_cam, dtype=np.float32)
        
        # Scale the CAM inside the bounding box
        box_cam = grayscale_cam[y1:y2, x1:x2].copy()
        
        # Normalize to [0, 1]
        if box_cam.size > 0:
            box_cam = (box_cam - box_cam.min()) / (box_cam.max() - box_cam.min() + 1e-8)
            img[y1:y2, x1:x2] = box_cam
        
        images.append(img)
    
    # Take the maximum across all images
    if images:
        renormalized_cam = np.max(np.float32(images), axis=0)
    
    # Normalize again
    if renormalized_cam.max() > renormalized_cam.min():
        renormalized_cam = (renormalized_cam - renormalized_cam.min()) / (renormalized_cam.max() - renormalized_cam.min())
    
    return renormalized_cam