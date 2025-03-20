import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import cv2
import pandas as pd
from src.model import AttributeModel  # Import the CNN model

class Training:
    def __init__(self, image_dir="preprocessed_faces/", attr_file="list_attr_celeba.csv"):
        """
        Initializes the Training class for facial attribute classification.

        :param image_dir: Directory containing preprocessed face images.
        :param attr_file: CSV file with facial attribute labels.
        """
        self.image_dir = image_dir
        self.attr_file = attr_file
        self.img_size = (160, 160)
        self.num_attributes = 40  # CelebA has 40 attributes

    def load_data(self):
        """Loads preprocessed images and corresponding attribute labels."""
        print("📥 Loading images and labels...")

        # Load attributes and convert -1 to 0
        df = pd.read_csv(self.attr_file)
        df.set_index("image_id", inplace=True)
        df.replace(-1, 0, inplace=True)  # Convert -1 to 0 (binary classification)

        # Store image data and labels
        images = []
        labels = []

        for img_name in df.index:
            img_path = os.path.join(self.image_dir, img_name)
            if os.path.exists(img_path):  # Ensure image exists
                image = cv2.imread(img_path)
                image = cv2.resize(image, self.img_size)
                image = image.astype("float32") / 255.0  # Normalize pixel values

                images.append(image)
                labels.append(df.loc[img_name].values)

        print(f"✅ Loaded {len(images)} images and labels.")
        return np.array(images), np.array(labels)

    def train(self, batch_size=32, epochs=10, test_size=0.2, model_path="models/attribute_model.h5"):
        """
        Trains the AttributeModel using Binary Cross-Entropy Loss.

        :param batch_size: Number of images per batch.
        :param epochs: Number of training epochs.
        :param test_size: Percentage of data used for validation.
        :param model_path: Path to save the trained model.
        """
        print("🚀 Starting training...")

        # Load preprocessed images & labels
        images, labels = self.load_data()

        # Split into train and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=test_size, stratify=labels, random_state=42
        )

        # Initialize & Compile Model
        model = AttributeModel().model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Train the Model
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs
        )

        # Save Model
        model.save(model_path)
        print(f"✅ Model training complete. Saved at: {model_path}")

# Run Training
if __name__ == "__main__":
    trainer = Training()
    trainer.train(batch_size=32, epochs=10)
