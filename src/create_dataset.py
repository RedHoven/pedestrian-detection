import os
import json
from glob import glob
from tqdm import tqdm
import shutil
import random

# Paths
image_dir = "../datasets/ECP/day/img/val/"
labels_dir = "../datasets/ECP/day/labels/val/"
updated_labels_dir = "../datasets/ECP/day/updated_labels/val/"

# Ensure output directory exists
os.makedirs(updated_labels_dir, exist_ok=True)

# Define class mapping (modify if multiple classes exist)
class_map = {"pedestrian":0}

# Function to convert JSON to YOLO format
def convert_labels_to_yolo_compatible(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    image_width = data["imagewidth"]
    image_height = data["imageheight"]
    yolo_annotations = []
    
    for obj in data.get("children", []):
        class_name = obj["identity"]
        if class_name not in class_map:
            continue
        
        class_id = class_map[class_name]
        x0, y0, x1, y1 = obj["x0"], obj["y0"], obj["x1"], obj["y1"]
        
        # Convert to YOLO format [center_x, center_y, width, height]
        center_x = ((x0 + x1) / 2) / image_width
        center_y = ((y0 + y1) / 2) / image_height
        width = (x1 - x0) / image_width
        height = (y1 - y0) / image_height

        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    return yolo_annotations

# Process all JSON files
json_files = glob(os.path.join(labels_dir, "**/*.json"), recursive=True)

for json_file in tqdm(json_files):
    # Generate corresponding .txt filename
    txt_file = json_file.replace(labels_dir, updated_labels_dir).replace(".json", ".txt")
    os.makedirs(os.path.dirname(txt_file), exist_ok=True)
    yolo_data = convert_labels_to_yolo_compatible(json_file)
    with open(txt_file, "w") as f:
        f.write("\n".join(yolo_data))

print("Conversion completed. YOLO annotations saved in:", updated_labels_dir)

def prepare_yolo_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Organizes YOLO dataset into train/val/test splits.

    Args:
        image_dir (str): Path to the directory containing all images.
        label_dir (str): Path to the directory containing all labels.
        output_dir (str): Destination directory for the organized dataset.
        train_ratio (float): Fraction of dataset for training.
        val_ratio (float): Fraction of dataset for validation.
        test_ratio (float): Fraction of dataset for testing.
    """

    # Ensure ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Train, val, and test ratios must sum to 1."

    # Create directories for images and labels
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Collect all image-label pairs, shuffle them
    image_paths = glob(os.path.join(image_dir, "**/*.png"), recursive=True)
    random.shuffle(image_paths)

    paired_data = []
    for img_path in tqdm(image_paths):
        # Get corresponding label path
        label_path = img_path.replace(image_dir, label_dir).replace(".png", ".txt")
        if os.path.exists(label_path):  # Ensure label exists
            paired_data.append((img_path, label_path))

    # Compute dataset splits
    total_count = len(paired_data)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)

    train_set = paired_data[:train_count]
    val_set = paired_data[train_count:train_count + val_count]
    test_set = paired_data[train_count + val_count:]

    # Function to move files
    def move_files(data_split, split_name):
        for img_path, lbl_path in tqdm(data_split):
            shutil.copy(img_path, os.path.join(output_dir, "images", split_name, os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(output_dir, "labels", split_name, os.path.basename(lbl_path)))

    # Move images and labels to respective directories
    move_files(train_set, "train")
    move_files(val_set, "val")
    move_files(test_set, "test")

    # Create data.yaml
    yaml_content = f"""
    path: {output_dir}
    train: images/train
    val: images/val
    test: images/test
    
    nc: 1  # Number of classes
    names: [0]  # Modify class names if needed
    """
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"Dataset prepared at {output_dir} with {train_count} train, {val_count} val, {total_count - train_count - val_count} test samples.")

output_dir = "datasets/ecp_dataset"
prepare_yolo_dataset(image_dir, updated_labels_dir, output_dir)