import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split, KFold

class Training:
    def __init__(self, images, labels, model, model_path = "models/attribute_model.h5"):

        self.images = images
        self.labels = labels
        self.model = model
        self.model_path = model_path

    def train(self, batch_size = 32, epochs = 10):

        print("Starting training...")

        kfold = KFold(n_splits = 5, random_state = 42, shuffle = True)
        fold = 1 
        val_scores = []
        best_val_acc = 0.0

        for train_idx, val_idx in kfold.split(self.images): 

            X_train, X_val = np.array(self.images)[train_idx], np.array(self.images)[val_idx]
            y_train, y_val = np.array(self.labels)[train_idx], np.array(self.labels)[val_idx]

            self.model = tf.keras.models.clone_model(self.model)
            self.model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"])

            results = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size = batch_size,
                epochs = epochs, 
                verbose = 1
            )

            val_acc = max(results.history["val_accuracy"])
            print(f"Fold {fold} Validation Accuracy: {val_acc:.4f}")
            val_scores.append(val_acc)

            if val_acc > best_val_acc: 
                best_val_acc = val_acc
                self.model.save_weights(self.model_path)
                print(f"New Best Model Saved")
            
            fold += 1

        print(f"Average Validation Accuracy: {np.mean(val_scores):.4f}")
