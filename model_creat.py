# Import necessary libraries
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    ZeroPadding2D,
    UpSampling2D,
    Concatenate
)
from tensorflow.keras.models import Model

# Set the input image size and number of classes
input_size = 416
num_classes = 2

# Prepare the labeled images and annotations
image_files = glob.glob("labeled_img/*.jpg")
annotation_files = [img_file.replace(".jpg", ".txt") for img_file in image_files]

# Configure the YOLO model
def create_yolo_model():
    input_layer = Input(shape=(input_size, input_size, 3))

    # Downsample path
    x = Conv2D(32, (3, 3), strides=(1, 1), padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Bridge
    x = Conv2D(128, (3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Upsample path
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate()([x, Conv2D(64, (3, 3), strides=(1, 1), padding="same")(input_layer)])
    x = Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate()([x, Conv2D(32, (3, 3), strides=(1, 1), padding="same")(input_layer)])
    x = Conv2D(32, (3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    output_layer = Conv2D(num_classes + 5, (1, 1), strides=(1, 1), padding="same")(x)

    model = Model(input_layer, output_layer)
    return model

# Train the YOLO model
model = create_yolo_model()
model.compile(optimizer="adam", loss="mse")
image_data = []
label_data = []

for i in range(len(image_files)):
    image = tf.keras.preprocessing.image.load_img(image_files[i], target_size=(input_size, input_size))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image_data.append(image)

    label = np.loadtxt(annotation_files[i])
    label = label.reshape((-1, 5))
    label_data.append(label)

image_data = np.array(image_data)
label_data = np.array(label_data)

model.fit(image_data, label_data, batch_size=4, epochs=10)

# Save the model for posterior use
model.save_weights("custom_yolo_weights.h5")
model.save("custom_yolo_model.h5")
