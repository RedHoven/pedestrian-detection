import os
import random
import shutil

def select_and_copy_images(src_folder, dest_folder, num_images=100):
    if not os.path.exists(dest_folder):
        raise FileNotFoundError(f"Destination folder '{dest_folder}' does not exist.")
    
    image_extension = '.png'
    all_files = os.listdir(src_folder)
    image_files = [file for file in all_files if file.lower().endswith(image_extension)]
    
    if len(image_files) < num_images:
        print(f"Only found {len(image_files)} images. All will be copied.")
        selected_images = image_files
    else:
        selected_images = random.sample(image_files, num_images)

    for image in selected_images:
        src_path = os.path.join(src_folder, image)
        dest_path = os.path.join(dest_folder, image)
        shutil.copy(src_path, dest_path)
        print(f"Copied {src_path} to {dest_path}")

def find_label_for_image(image_filename, images_folder, labels_folder):
    """
    Given an image filename, this function will return the corresponding label's path.
    
    :param image_filename: The filename of the image
    :param images_folder: The path to the folder containing the images
    :param labels_folder: The path to the folder containing the annotations
    :return: The path to the corresponding annotation file
    """

    # Ensure the image file exists in the images folder
    image_path = os.path.join(images_folder, image_filename)
    if not os.path.exists(image_path):
        print(f"Image {image_filename} not found in {images_folder}.")
        return None
    
    # Replace the image file extension with '.txt' to find the corresponding label
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    label_path = os.path.join(labels_folder, label_filename)
    
    # Check if the label file exists
    if os.path.exists(label_path):
        return label_path
    else:
        print(f"Annotation for {image_filename} not found.")
        return None


if __name__ == '__main__':

    # Generate samples
    random.seed(42)
    # source_folder = input("Enter the source folder path: ").strip()
    # destination_folder = input("Enter the destination folder path: ").strip()
    # source_folder = "../datasets/ecp_dataset/images/test"
    # destination_folder = "../datasets/visualization_samples"
    # select_and_copy_images(source_folder, destination_folder)

    # Find label for image
    # img_folder = "../datasets/visualization_samples/unsorted/"
    # img_name = "amsterdam_01238.png"
    # labels_folder = "datasets/ecp_dataset/labels/test/"
    # label_path = find_label_for_image(img_name, img_folder, labels_folder)
    # if label_path is not None:
    #     print(open(label_path).read())
    #     print("EOF")
    # else:
    #     print("Label not found.")

