import numpy as np
import tensorflow as tf
import os
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load trained model
model = tf.keras.models.load_model("models/attribute_model.h5")

# Load attribute names
attributes_list = list(pd.read_csv("list_attr_celeba.csv").columns[1:])

# Load test data
def load_test_data(image_dir="preprocessed_faces/", attr_file="list_attr_celeba.csv", test_size=0.2):
    """Loads test images and labels."""
    df = pd.read_csv(attr_file)
    df.set_index("image_id", inplace=True)
    df.replace(-1, 0, inplace=True)  # Convert -1 to 0 (binary classification)

    images = []
    labels = []

    for img_name in df.index:
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):  # Ensure image exists
            image = cv2.imread(img_path)
            image = cv2.resize(image, (160, 160))
            image = image.astype("float32") / 255.0  # Normalize pixels
            images.append(image)
            labels.append(df.loc[img_name].values)

    images = np.array(images)
    labels = np.array(labels)

    # Use last 20% of the data as test set
    split_index = int(len(images) * (1 - test_size))
    return images[split_index:], labels[split_index:]

# Load test set
X_test, y_test = load_test_data()

# Make predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Compute metrics
accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
precision = precision_score(y_test.flatten(), y_pred.flatten(), average="macro")
recall = recall_score(y_test.flatten(), y_pred.flatten(), average="macro")
f1 = f1_score(y_test.flatten(), y_pred.flatten(), average="macro")

# Print results
print(f"✅ Model Evaluation Results:")
print(f"🔹 Accuracy:  {accuracy:.4f}")
print(f"🔹 Precision: {precision:.4f}")
print(f"🔹 Recall:    {recall:.4f}")
print(f"🔹 F1 Score:  {f1:.4f}")

