import pandas as pd
import os 
import cv2
import numpy as np

class Preprocessor:
    def __init__(self):
        self.img_size = (160, 160) 
        BASE_DIR = os.path.abspath("data")
        OUTPUT_DIR = os.path.abspath("dataset/preprocessed_faces")

        self.attr_file_path = os.path.join(BASE_DIR, "list_attr_celeba.csv")
        self.bbox_file_path = os.path.join(BASE_DIR, "list_bbox_celeba.csv")
        self.image_folder_path = os.path.join(BASE_DIR, "img_align_celeba/img_align_celeba")
        self.output_path = OUTPUT_DIR 

        os.makedirs(self.output_path, exist_ok=True)

    def load(self):
        """Loads attribute data."""
        self.attributes = pd.read_csv(self.attr_file_path, header = 0)
        self.attributes.set_index("image_id", inplace=True)
        self.attributes = self.attributes.replace(-1, 0)

    def process_images(self, limit = 1000):
        """Processes images: resizes to model input size, optionally converts to grayscale."""
        processed_images = []
        processed_labels = []

        for img_name in self.attributes.index[:limit]:  
            
            img_path = os.path.join(self.image_folder_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, self.img_size)
            image = np.expand_dims(image, axis = -1)
            processed_images.append(image)
            processed_labels.append(self.attributes.loc[img_name].values)

        return np.array(processed_images), np.array(processed_labels)

    def preprocess_image(self, image_path):
        """Loads and preprocesses a single image for model inference."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        image = cv2.resize(image, self.img_size)  
        image = image.astype("float32") / 255.0  
        image = np.expand_dims(image, axis=0)  

        return image
    

