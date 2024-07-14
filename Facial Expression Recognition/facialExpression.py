import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2
import os
from glob import glob
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.callbacks import ModelCheckpoint

train_dir = 'Facial Expression Recognition/fer2013/train'
model_path = 'Facial Expression Recognition/facialExpression.h5'

classes = os.listdir(train_dir)

# Create training and validation datasets
train_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    label_mode='categorical'  # Ensure labels are one-hot encoded
)

val_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    label_mode='categorical'  # Ensure labels are one-hot encoded
)

# Normalize the images to [0,1] range
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(normalize_img)
val_dataset = val_dataset.map(normalize_img)

# Cache and prefetch for performance optimization
train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

if os.path.exists(model_path):
    print("Loading existing model...")
    model = keras.models.load_model(model_path)
else:
    print("Creating a new model...")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), padding="same", activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"),
        Conv2D(96, (3, 3), padding="same", activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(8, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy", optimizer= tf.keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])



# Define the callback
checkpoint_callback = ModelCheckpoint(
    filepath=model_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)

# Train the model with the callback
history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_dataset),
    epochs=50,
    validation_data=val_dataset,
    validation_steps=len(val_dataset),
    callbacks=[checkpoint_callback]
)

model.save('Facial Expression Recognition/facialExpression.h5')
