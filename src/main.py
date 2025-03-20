import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocessor import Preprocessor  
from model import AttributeModel  
from train import Training 
from inference import Evaluate 

def main():
    """Main function to handle training, inference, and evaluation."""

    preprocessor = Preprocessor()

    print("Loading dataset...")
    preprocessor.load()  
    images, labels = preprocessor.process_images(limit = 5000)

    model = AttributeModel().model

    trainer = Training(images, labels, model)
    trainer.train(batch_size = 32, epochs = 5)

    evaluator = Evaluate(model_path = MODEL_PATH, attr_file = ATTRIBUTES_FILE)
    
    test_image = preprocessor.preprocess_image(TEST_IMAGE_PATH)
    
    top_attributes = evaluator.predict_attributes(TEST_IMAGE_PATH, top_k=5)
    evaluator.draw_predictions(TEST_IMAGE_PATH, top_attributes, show_image=True)

    print("Top Predicted Attributes:", top_attributes)

if __name__ == "__main__":
    main()
