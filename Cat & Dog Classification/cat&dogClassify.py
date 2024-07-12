import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import Callback
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

train_dir = 'Cat & Dog Classification/train'
test_dir = 'Cat & Dog Classification/test1'
model_path = 'Cat & Dog Classification/cat&dog.h5'

# Ensure the directories exist
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("The specified directory does not exist.")

# Create training and validation datasets
train_datagen = image_dataset_from_directory(train_dir, image_size=(200, 200), batch_size=32, subset='training', validation_split=0.2, seed=123)
val_datagen = image_dataset_from_directory(train_dir, image_size=(200, 200), batch_size=32, subset='validation', validation_split=0.2, seed=123)

# Custom callback to stop training when validation accuracy exceeds 92%
class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='accuracy', value=0.92, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            print(f"Warning: Early stopping requires {self.monitor} available!")
            return
        
        if current > self.value:
            if self.verbose > 0:
                print(f"Epoch {epoch}: early stopping as {self.monitor} reached {self.value}")
            self.model.stop_training = True

# Check if the model file exists
if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
else:
    print("Creating a new model...")
    # Define the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.1),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with the custom callback
early_stopping = EarlyStoppingByAccuracy(monitor='accuracy', value=0.92, verbose=1)
history = model.fit(train_datagen, validation_data=val_datagen, epochs=10, callbacks=[early_stopping])

# Plot the training history
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()

# Save the model
model.save(model_path)
