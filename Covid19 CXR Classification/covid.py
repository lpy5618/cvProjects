import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Set data path and check classes
data_path = 'Covid19 CXR Classification/COVID-19_Radiography_Dataset'
model_path = 'Covid19 CXR Classification/covid.h5'
classes = os.listdir(data_path)
print(classes)

# Load Xception model
base = Xception(weights="imagenet", input_shape=(299, 299, 3), include_top=False)
# Set base model layers to non-trainable
for layer in base.layers:
    layer.trainable = False

base.summary()

# ImageDataGenerator setup
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

if os.path.exists(model_path):
    print("Loading existing model...")
    model = keras.models.load_model(model_path)
else:
    print("Creating a new model...")
    # Create the model
    model = Sequential([
        Input(shape=(299, 299, 3)),
        base,
        Dropout(0.2),
        Flatten(),
        Dropout(0.2),
        Dense(16),
        Dense(len(classes), activation='softmax')
    ])
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
checkpoint_callback = ModelCheckpoint(
    filepath=model_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)

# Train the model
model.fit(train_generator, epochs=10, 
          validation_data=validation_generator, 
          callbacks=[checkpoint_callback])

# Save the model
model.save(model_path)
