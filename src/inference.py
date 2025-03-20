import numpy as np
import tensorflow as tf
import cv2
import os
import pandas as pd
from preprocessor import Preprocessor

class Evaluate:
    def __init__(self, model_path="models/attribute_model.h5", attr_file="list_attr_celeba.csv", results_dir="results/"):
        """Initialize the model and attribute list."""
        self.model = tf.keras.models.load_model(model_path) 
        self.attributes_list = list(pd.read_csv(attr_file).columns[1:])  

        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.preprocessor = Preprocessor()

    def predict_attributes(self, image_path, top_k=5):
        """Predicts the top K attributes for a given image."""
        image = self.preprocessor.preprocess_image(image_path)  

        predictions = tf.nn.sigmoid(self.model.predict(image)[0]).numpy() 
        top_indices = np.argsort(predictions)[-top_k:][::-1] 

        top_attributes = [(self.attributes_list[i], predictions[i]) for i in top_indices]
        return top_attributes
    
    def draw_predictions(self, image_path, top_attributes, show_image=False):
        """Draws a bounding box and overlays predicted attributes onto an image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        start_x, start_y, box_width, box_height = 20, 20, 220, 150
        cv2.rectangle(image, (start_x, start_y), (start_x + box_width, start_y + box_height), (0, 255, 0), 2)

        y_offset = start_y + 30
        for attr, prob in top_attributes:
            text = f"{attr}: {prob:.2f}"
            cv2.putText(image, text, (start_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3) 
            cv2.putText(image, text, (start_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  
            y_offset += 25

        image_name = os.path.basename(image_path)
        output_path = os.path.join(self.results_dir, f"predicted_{image_name}")
        cv2.imwrite(output_path, image)
        print(f"Output image saved at {output_path}")

        if show_image:
            cv2.imshow("Predictions", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def evaluate_image(self, image_path, top_k=5, show_image=False):
        """Runs the full evaluation pipeline on a single image."""
        print(f"Evaluating {image_path}...")

        top_attributes = self.predict_attributes(image_path, top_k=top_k)
        self.draw_predictions(image_path, top_attributes, show_image=show_image)

        return top_attributes