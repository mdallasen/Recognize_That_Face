import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

class AttributeModel:
    def __init__(self, input_shape=(160, 160, 3), num_attributes=40):
        """
        Initializes the CNN model for facial attribute classification.

        :param input_shape: Shape of input images (height, width, channels).
        :param num_attributes: Number of facial attributes (default: 40).
        """
        self.input_shape = input_shape
        self.num_attributes = num_attributes
        self.model = self.build_model()

    def build_model(self):

        model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(256, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(), 

            Dense(512, activation="relu"),
            Dropout(0.5),

            Dense(self.num_attributes, activation="sigmoid")  
        ])

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model



