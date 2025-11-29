import os
import shutil
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

class ImageSorterApp:
    def __init__(self, root, src_folder, dest_folder_low, dest_folder_medium, dest_folder_high, dest_folder_empty):
        self.root = root
        self.src_folder = src_folder
        self.dest_folder_low = dest_folder_low
        self.dest_folder_medium = dest_folder_medium
        self.dest_folder_high = dest_folder_high
        self.dest_folder_empty = dest_folder_empty
        
        self.image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'))]
        self.current_image_index = 0
        
        # Create the GUI elements
        self.create_widgets()

        # Load the first image
        self.load_image()

        # Bind the key press events
        self.root.bind("1", lambda event: self.sort_image("empty"))
        self.root.bind("2", lambda event: self.sort_image("low"))
        self.root.bind("3", lambda event: self.sort_image("medium"))
        self.root.bind("4", lambda event: self.sort_image("high"))
        self.root.bind("<Right>", lambda event: self.load_next_image())  # Arrow key to go to the next image

    def create_widgets(self):
        self.img_label = Label(self.root)
        self.img_label.pack()

        # Buttons for sorting images
        self.empty_button = Button(self.root, text="Empty", command=lambda: self.sort_image("empty"))
        self.empty_button.pack(side=LEFT)

        self.low_button = Button(self.root, text="Low Complexity", command=lambda: self.sort_image("low"))
        self.low_button.pack(side=LEFT)

        self.medium_button = Button(self.root, text="Medium Complexity", command=lambda: self.sort_image("medium"))
        self.medium_button.pack(side=LEFT)

        self.high_button = Button(self.root, text="High Complexity", command=lambda: self.sort_image("high"))
        self.high_button.pack(side=LEFT)

        self.skip_button = Button(self.root, text="Skip", command=self.load_next_image)
        self.skip_button.pack(side=LEFT)

    def load_image(self):
        if self.current_image_index < len(self.image_files):
            image_name = self.image_files[self.current_image_index]
            image_path = os.path.join(self.src_folder, image_name)
            img = Image.open(image_path)
            img.thumbnail((800, 800))  # Resize the image to fit in the window
            img_tk = ImageTk.PhotoImage(img)

            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk  # Keep a reference to avoid garbage collection

    def sort_image(self, category):
        if self.current_image_index < len(self.image_files):
            image_name = self.image_files[self.current_image_index]
            image_path = os.path.join(self.src_folder, image_name)
            
            if category == "low":
                dest_folder = self.dest_folder_low
            elif category == "medium":
                dest_folder = self.dest_folder_medium
            elif category == "empty":
                dest_folder = self.dest_folder_empty
            else:
                dest_folder = self.dest_folder_high

            # Move the image to the corresponding folder
            shutil.move(image_path, os.path.join(dest_folder, image_name))
            print(f"Moved {image_name} to {category} folder.")
            
            # Load the next image
            self.load_next_image()

    def load_next_image(self):
        self.current_image_index += 1
        if self.current_image_index < len(self.image_files):
            self.load_image()
        else:
            messagebox.showinfo("Done", "You have sorted all the images!")
            self.root.quit()  # Close the app when all images are sorted

# Running the application
if __name__ == "__main__":
    # Set up the paths for source and destination folders
    src_folder = "../datasets/visualization_samples/high_complexity"
    general_dest_folder = "../datasets/visualization_samples/"
    dest_folder_low = general_dest_folder + "/low_complexity"
    dest_folder_medium = general_dest_folder + "/medium_complexity"
    dest_folder_high = general_dest_folder + "/high_complexity"
    dest_folder_empty = general_dest_folder + "/empty"

    root = Tk()
    root.title("Image Sorter")
    app = ImageSorterApp(root, src_folder, dest_folder_low, dest_folder_medium, dest_folder_high, dest_folder_empty)
    root.mainloop()
