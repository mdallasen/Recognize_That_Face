import numpy as np
import tensorflow as tf
import cv2
import os
import pandas as pd

# Load trained model
model = tf.keras.models.load_model("models/attribute_model.h5")

# Load attribute names
attributes_list = list(pd.read_csv("list_attr_celeba.csv").columns[1:])

# Ensure results folder exists
RESULTS_DIR = "results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

def preprocess_image(image_path):
    """Loads an image, resizes it to 160x160, and normalizes pixel values."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (160, 160))
    image = image.astype("float32") / 255.0  # Normalize pixels (0-1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_attributes(image_path, model, attributes_list, top_k=5):
    """Predicts the top K attributes for a given image."""
    image = preprocess_image(image_path)
    
    # Run model prediction
    predictions = model.predict(image)[0]  # Get first image's predictions

    # Get top K attributes
    top_indices = np.argsort(predictions)[-top_k:][::-1]  # Get indices of top 5
    top_attributes = [(attributes_list[i], predictions[i]) for i in top_indices]

    return top_attributes

def draw_predictions(image_path, top_attributes, results_dir=RESULTS_DIR):
    """Draws a bounding box and overlays predicted attributes onto an image."""
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Define box position
    start_x, start_y, box_width, box_height = 20, 20, 220, 150
    cv2.rectangle(image, (start_x, start_y), (start_x + box_width, start_y + box_height), (0, 255, 0), 2)

    # Overlay text
    y_offset = start_y + 30
    for attr, prob in top_attributes:
        text = f"{attr}: {prob:.2f}"
        cv2.putText(image, text, (start_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25

    # Generate output file path
    image_name = os.path.basename(image_path)
    output_path = os.path.join(results_dir, f"predicted_{image_name}")

    # Save and display image
    cv2.imwrite(output_path, image)
    print(f"✅ Output image saved at {output_path}")
    cv2.imshow("Predictions", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run inference and overlay results
if __name__ == "__main__":
    image_path = "test_face.jpg"  # Replace with an actual test image
    if not os.path.exists(image_path):
        print("❌ Error: Image does not exist.")
    else:
        top_attrs = predict_attributes(image_path, model, attributes_list)
        draw_predictions(image_path, top_attrs)
