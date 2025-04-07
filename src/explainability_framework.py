import argparse
import json
import numpy as np
import glob
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO, RTDETR
from lime import lime_image
import os
from typing import List, Tuple, Dict, Optional
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from img_utils import find_label_for_image
from cam_utils import *
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from pytorch_grad_cam import AblationCAM, EigenCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm

class ExplainabilityFramework:
    def __init__(self, yolo_weights_path: str, rtdetr_weights_path: str):
        """
        Initialize the explainability framework with model paths
        
        :param yolo_weights_path: Path to YOLOv8 weights file

        :param rtdetr_weights_path: Path to RT-DETR weights file
        """

        self.debug = False
        self.yolo_weights_path = os.path.abspath(yolo_weights_path)
        self.rtdetr_weights_path = os.path.abspath(rtdetr_weights_path)
        self.yolo_model = YOLO(yolo_weights_path)
        self.rtdetr_model = RTDETR(rtdetr_weights_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move models to device
        self.yolo_model.to(self.device)
        self.rtdetr_model.to(self.device)
        
    def load_image(self, image_path: str, resize_scale: float = 0.5) -> np.ndarray:
        """
        Load an image from path with HWC dimensions
        
        :param image_path: Path to the image
        :param resize_scale: Scale factor for resizing the image (default: 0.5)

        :return: Image as numpy array (RGB)
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if resize_scale != 1.0:
            # Calculate new dimensions
            width = int(img.shape[1] * resize_scale)
            height = int(img.shape[0] * resize_scale)
            
            # Resize image
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        return img
    
    def load_labels(self, label_path: str) -> List[List[float]]:
        """
        Load ground truth labels from a text file
        
        :param label_path: Path to the label file  

        :return: List of labels [class_id, x_center, y_center, width, height]
        """
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    labels.append([float(x) for x in line.strip().split()])
        return labels
    
    def predict(self, img: np.ndarray, model_type: str = 'yolov8') -> Dict:
        """
        Run prediction on an image
    
        :param: img: Input image as numpy array
        :param: model_type: Type of model ('yolov8' or 'rtdetr')
        
        :return: Dictionary with detection results
        """
        model = self.yolo_model if model_type == 'yolov8' else self.rtdetr_model
        results = model.predict(img, verbose=False)[0]
        
        # Extract boxes, confidence scores and class IDs
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy()
        
        return {
            'boxes': boxes,
            'confidences': confs,
            'class_ids': cls_ids
        }
    
    def apply_lime(self, img: np.ndarray, model_type: str = 'yolov8', num_samples: int = 100, style: str = "boundary") -> np.ndarray:
        """
        Apply LIME explainability method
        
        :param img: Input image
        :param model_type: Type of model ('yolov8' or 'rtdetr')
        :param num_samples: Number of samples for LIME
        
        :return: Explanation mask
        """
        model = self.yolo_model if model_type == 'yolov8' else self.rtdetr_model
        
        # Define a prediction function for LIME
        def predict_fn(images):
            batch_preds = []
            for image in images:
                # Convert to uint8 as expected by the model
                image_uint8 = np.uint8(image)
                result = model.predict(image_uint8, verbose=False)[0]
                
                # Create a heatmap based on bounding boxes and confidence
                heatmap = np.zeros((image.shape[0], image.shape[1]))
                
                if len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = map(int, box)
                        heatmap[y1:y2, x1:x2] += conf
                
                # Normalize the heatmap
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()
                
                batch_preds.append(heatmap.flatten())
            
            return np.array(batch_preds)
        
        # Initialize LIME explainer
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img, 
            predict_fn,
            top_labels=5,
            hide_color=0, 
            num_samples=num_samples
        )

        label = explanation.top_labels[0]    
        segments = explanation.segments
        local_exp = dict(explanation.local_exp[label])

        mask = np.zeros_like(segments, dtype=np.float32)
        
        # Only take the top `num_features` by absolute weight
        num_features = 5
        top_features = sorted(local_exp.items(), key=lambda x: abs(x[1]), reverse=True)[:num_features]
        
        for seg_id, weight in top_features:
            mask[segments == seg_id] = weight

        # Normalize for visualization
        mask_normalized = mask / (np.abs(mask).max() + 1e-8)
        
        # Create RGB overlay: red for negative, green for positive
        overlay = np.zeros((*mask.shape, 3))
        overlay[:, :, 1] = np.clip(mask_normalized, 0, 1)    # Green for positive
        overlay[:, :, 0] = np.clip(-mask_normalized, 0, 1)   # Red for negative

        img_float = img.astype(np.float32) / 255.0
        blended = 0.5 * img_float + 0.5 * overlay

        return blended
    
    def apply_cam(self, img: np.ndarray, model_type: str, 
                  renormalize_boxes: bool = False, ratio_channels: float = 0.1):
        """
        Apply Class Activation Maps to detections from YOLOv8 or RT-DETR models.
        
        :param img: Input image as numpy array (BGR or RGB)
        :param model_type: Type of model ('yolov8' or 'rt-detr')
        :param renormalize_boxes: Whether to renormalize CAM within each bounding box
        :param ratio_channels: Ratio of channels to ablate (for AblationCAM only)
        
        :return: Image with CAM visualization overlaid
        """

        if img.shape[2] == 3 and img[0, 0, 0] > img[0, 0, 2]:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img.copy()
        
        img_float = np.float32(img_rgb) / 255.0

        model = self.yolo_model if model_type == 'yolov8' else self.rtdetr_model
        
        results = self.predict(img_rgb, model_type)        

        boxes = results['boxes']
        labels = results['class_ids']
        
        input_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            model.to('cuda')
        
        target_layers = UltralyticsTargetLayers.get_target_layers(model, model_type)
        targets = [UltralyticsBoxScoreTarget(boxes, labels)]
        cam = EigenCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=ultralytics_reshape_transform
        )

        grayscale_cam = cam(input_tensor, targets=targets)[0]

        if renormalize_boxes:
            grayscale_cam = renormalize_cam_in_bounding_boxes(img_rgb, boxes, grayscale_cam)

        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        return cam_image
    
    def _add_true_boxes(self, ax: Axes, labels: List[List[float]], img_shape: Tuple[int, int, int]) -> None:
        """
        Add ground truth bounding boxes to a plot
        
        :param ax: Matplotlib axis
        :param labels: List of labels [class_id, x_center, y_center, width, height]
        :param img_shape: Shape of the image (height, width, channels)
        
        :return: None
        """
        height, width = img_shape[:2]
        
        for label in labels:
            class_id, x_center, y_center, box_width, box_height = label
            
            # Convert from normalized coordinates to pixels
            x_center, y_center = x_center * width, y_center * height
            box_width, box_height = box_width * width, box_height * height
            
            # Calculate box corners
            x1, y1 = x_center - box_width / 2, y_center - box_height / 2
            
            rect = patches.Rectangle(
                (x1, y1), 
                box_width, 
                box_height, 
                alpha=0.5,
                linewidth=1, 
                edgecolor='green', 
                facecolor='none', 
                label='Ground Truth'
            )
            ax.add_patch(rect)
    
    def _add_pred_boxes(self, ax: Axes, preds: Dict, img_shape: Tuple[int, int, int], threshold: float = 0.2) -> None:
        """
        Add predicted bounding boxes to a plot
        
        :param ax: Matplotlib axis
        :param preds: Dictionary with prediction results
        :param img_shape: Shape of the image (height, width, channels)
        :param threshold: Confidence threshold for displaying boxes
        """
        boxes = preds['boxes']
        confs = preds['confidences']
        
        for box, conf in zip(boxes, confs):
            if conf < threshold:
                continue

            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Create and add rectangle patch
            rect = patches.Rectangle(
                (x1, y1), 
                width, 
                height,
                alpha=0.5,
                linewidth=1, 
                edgecolor='red', 
                facecolor='none', 
                label='Prediction'
            )
            ax.add_patch(rect)
            
            ax.text(
                x1, 
                y1 - 5, 
                f"{conf:.2f}", 
                color='red', 
                fontsize=4, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
    
    def _add_legend(self, ax: Axes) -> None:
        """
        Add legend to a plot

        :param ax: Matplotlib axis
        """
        # Create handles for legend
        green_patch = patches.Patch(color='green', label='Ground Truth')
        red_patch = patches.Patch(color='red', label='Prediction')
        
        ax.legend(handles=[green_patch, red_patch], loc='upper right', fontsize=8)
    
    def _calculate_model_score(self, true_labels: List[List[float]], preds: Dict, img_shape: Tuple[int, int, int]) -> float:
        """
        Calculate average IoU score between predictions and ground truth
        
        :param true_labels: List of ground truth labels
        :param preds: Dictionary containing prediction results
        :param img_shape: Shape of the image

        :return: Average IoU score as a float
        """
        if not true_labels or len(preds['boxes']) == 0:
            return 0.0
        
        height, width = img_shape[:2]
        true_boxes = []
        
        # Convert normalized coordinates to pixels
        for label in true_labels:
            _, x_center, y_center, box_width, box_height = label
            
            x_center, y_center = x_center * width, y_center * height
            box_width, box_height = box_width * width, box_height * height
            
            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2
            
            true_boxes.append([x1, y1, x2, y2])
        
        true_boxes = np.array(true_boxes)
        pred_boxes = preds['boxes']
        
        # Calculate IoU for each prediction with best matching ground truth
        ious = []
        for pred_box in pred_boxes:
            best_iou = 0
            for true_box in true_boxes:
                iou = self._calculate_iou(pred_box, true_box)
                best_iou = max(best_iou, iou)
            ious.append(best_iou)
        
        return np.mean(ious) if ious else 0.0
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate IoU between two bounding boxes
        
        :param box1: First bounding box in the format [x1, y1, x2, y2]
        :param box2: Second bounding box in the format [x1, y1, x2, y2]
        
        :return: IoU score as a float
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_detailed(
        self, 
        img_path: str, 
        label_path: str, 
        category: str = "low-complexity",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Detailed visualization for single image with 6 sections.
        
        :param img_path: Path to the image.
        :param label_path: Path to the label file.
        :param category: Image complexity category.
        :param save_path: Optional path to save the visualization.
        
        :return: Matplotlib figure with detailed visualizations.
        """

        img = self.load_image(img_path)
        true_labels = self.load_labels(label_path)
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))

        # Original image without boxes
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        self._add_true_boxes(axes[0], true_labels, img.shape)

        if not self.debug:
        
            # Get predictions
            yolo_preds = self.predict(img, 'yolov8')
            rtdetr_preds = self.predict(img, 'rtdetr')
            
            # Apply explainability methods
            lime_yolo = self.apply_lime(img, 'yolov8', num_samples=3)
            lime_rtdetr = self.apply_lime(img, 'rtdetr', num_samples=3)
            
            # YOLO LIME
            axes[1].imshow(lime_yolo)
            axes[1].set_title("YOLOv8 with LIME")
            self._add_pred_boxes(axes[1], yolo_preds, img.shape)

            # RT-DETR LIME
            axes[2].imshow(lime_rtdetr)
            axes[2].set_title("RT-DETR with LIME")
            self._add_pred_boxes(axes[2], rtdetr_preds, img.shape)
            
            # Info text
            yolo_score = self._calculate_model_score(true_labels, yolo_preds, img.shape)
            rtdetr_score = self._calculate_model_score(true_labels, rtdetr_preds, img.shape)
            info_text = (
                f"Image: {os.path.basename(img_path)}\n"
                f"Category: {category}\n"
                f"True Labels: {len(true_labels)}\n"
                f"YOLOv8 Detections: {len(yolo_preds['boxes'])}\n"
                f"YOLOv8 Avg Conf: {np.mean(yolo_preds['confidences']):.2f}\n"
                f"YOLOv8 IoU: {yolo_score:.2f}\n\n"
                f"RT-DETR Detections: {len(rtdetr_preds['boxes'])}\n"
                f"RT-DETR Avg Conf: {np.mean(rtdetr_preds['confidences']) if rtdetr_preds['confidences'].size else 0:.2f}\n"
                f"RT-DETR IoU: {rtdetr_score:.2f}"
            )
            filename = os.path.basename(img_path)
            with open(f"{os.path.dirname(save_path)}/{filename}_info.txt", "w") as f:
                f.write(info_text)
        else:
            axes[1].imshow(img)
            axes[2].imshow(img)
            axes[1].set_title("YOLOv8 with LIME")
            axes[2].set_title("RT-DETR with LIME")
            axes[0].set_title("Original Image")

        for ax in axes.flatten():
            ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        return fig

    def visualize_grid(
        self,
        image_paths: List[str],
        label_paths: List[str],
        categories: List[str],
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Grid visualization of images in 3 columns: Original, YOLOv8, RT-DETR.
        YOLOv8 and RT-DETR images will have their respective CAMs applied.
        The images will be aligned to the right with almost no vertical spacing between rows.

        :param image_paths: List of image paths.
        :param label_paths: List of corresponding label paths.
        :param categories: List of categories for each image.
        :param save_path: Optional path to save the visualization.

        :return: Matplotlib figure with brief report.
        """
        n_images = len(image_paths)
        img_dim = self.load_image(image_paths[0]).shape
        img_h = 1.5
        img_w = 3 * img_h * img_dim[1] / img_dim[0]

        fig, axes = plt.subplots(n_images, 3, figsize=(img_w, img_h * n_images))
        
        # Ensure axes is a 2D array even if n_images == 1
        if n_images == 1:
            axes = axes.reshape(1, -1)

        for i in tqdm(range(n_images)):
            img = self.load_image(image_paths[i])
            true_labels = self.load_labels(label_paths[i])

            if not self.debug:
                yolo_preds = self.predict(img, 'yolov8')
                rtdetr_preds = self.predict(img, 'rtdetr')
                cam_yolo = self.apply_cam(img, 'yolov8')
                cam_rtdetr = self.apply_cam(img, 'rtdetr')

                # Original with GT boxes
                axes[i, 0].imshow(img)
                self._add_true_boxes(axes[i, 0], true_labels, img.shape)

                # YOLOv8 CAM
                axes[i, 1].imshow(cam_yolo)
                # self._add_pred_boxes(axes[i, 1], yolo_preds, img.shape)

                # RT-DETR CAM
                axes[i, 2].imshow(cam_rtdetr)
                # self._add_pred_boxes(axes[i, 2], rtdetr_preds, img.shape)
            else:
                axes[i, 0].imshow(img)
                axes[i, 1].imshow(img)
                axes[i, 2].imshow(img)
            
            # Remove individual titles and axis markers
            for ax in axes[i]:
                ax.axis('off')

        # Overarching column titles at the top center of each column
        col_titles = ["Original Image", "YOLOv8", "RT-DETR"]
        for col, title in enumerate(col_titles):
            bbox = axes[0, col].get_position()
            fig.text(
                x=bbox.x0 + bbox.width/2,
                y=0.98,
                s=title,
                ha='center', va='top',
                fontsize=10 
            )

        plt.subplots_adjust(left=0.1, right=0.90, top=0.95, bottom=0.05, wspace=0.025, hspace=0.05)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig

if __name__ == "__main__":

    args_parser = argparse.ArgumentParser(description="Explainability Framework")
    args_parser.add_argument("--grid", type=str, default="report", help="Name of the grid report image")
    args_parser.add_argument("--detailed", type=str, default="report", help="Name of the detailed report image")

    args = args_parser.parse_args()

    yolo_weights_path = "../training/yolo_training/fine_tune_50/weights/best.pt"
    rtdetr_weights_path = "../training/detr_training/fine_tune_35/weights/last.pt"    
    
    framework = ExplainabilityFramework(yolo_weights_path, rtdetr_weights_path)
    # framework.debug = True
    
    experiments_dir = "../datasets/visualization_samples"
    label_dir = "../datasets/ecp_dataset/labels/test/"
    results_dir = f"results"
    categories = ["empty", "low_complexity", "medium_complexity", "high_complexity"]
    category_mapping = {cat: glob.glob(f"{experiments_dir}/{cat}/*") for cat in categories}

    img_path = category_mapping[categories[3]][2]
    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    label_path = find_label_for_image(img_name, img_folder, label_dir)

    print(f"Image: {img_name}")
    print(f"Label: {label_path}")

    img_paths = category_mapping["empty"][:10] + category_mapping["low_complexity"][:10] + category_mapping["medium_complexity"] + category_mapping["high_complexity"]

    complexity_report = "high_complexity"
    framework.visualize_grid(
        category_mapping[complexity_report][:3],
        [find_label_for_image(os.path.basename(img), f"{experiments_dir}/{complexity_report}", label_dir) for img in category_mapping[complexity_report][:3]],
        [complexity_report] * len(category_mapping[complexity_report][:3]),
        save_path=os.path.join(results_dir, f"grid_{args.grid}.png")
    )