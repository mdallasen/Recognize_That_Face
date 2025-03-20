import pandas as pd
import os 
import cv2

class Preprocessor:
    def __init__(self):
        self.img_size = (160, 160)
        self.attr_file_path = "list_attr_celeba.csv"
        self.bbox_file_path = "list_bbox_celeba.csv"
        self.image_folder_path = "img_align_celeba"
        self.output_path = "preprocessed_faces/"

        os.makedirs(self.output_path, exist_ok=True)

    def load(self):
        """Loads dataset metadata."""
        self.attributes = pd.read_csv(self.attr_file_path)
        self.bbox = pd.read_csv(self.bbox_file_path)

        self.attributes.set_index("image_id", inplace=True)
        self.attributes = self.attributes.replace(-1, 0)

        self.bbox.set_index("image_id", inplace=True)

    def crop_and_resize(self, image_path, bbox):
        """Crops a face using bounding box and resizes to (160,160)."""
        image = cv2.imread(image_path)

        if image is None:
            return None 
        
        x, y, w, h = bbox
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, self.img_size)

        return face

    def process_images(self):
        """Processes all images by cropping and resizing."""
        for img_name, row in tqdm(self.bbox.iterrows(), total=len(self.bbox), desc="Processing Images"):
            img_path = os.path.join(self.image_folder_path, img_name)

            if not os.path.exists(img_path):
                print(f"Skipping missing image: {img_name}")
                continue

            bbox = row[['x_1', 'y_1', 'width', 'height']].astype(int).values
            face = self.crop_and_resize(img_path, bbox)

            if face is not None:
                cv2.imwrite(os.path.join(self.output_path, img_name), face)

    def preprocess(self):
        """Runs all preprocessing steps."""
        print("Loading dataset...")
        self.load()
        print("Processing images...")
        self.process_images()
        print("Preprocessing complete!")


