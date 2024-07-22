import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

# Set data path and check classes
model_path = 'Covid19 CXR Classification/covid.h5'
data_path = 'Covid19 CXR Classification/COVID-19_Radiography_Dataset'
classes = os.listdir(data_path)
print("Classes:", classes)

# Load Xception model
model = tf.keras.models.load_model(model_path)
print("Model loaded.")

# Get image paths
image_paths = []
for class_name in classes:
    class_path = os.path.join(data_path, class_name)
    for img_file in os.listdir(class_path):
        image_paths.append(os.path.join(class_path, img_file))

# Randomly select 20 images
random_images = random.sample(image_paths, 20)

# Function to predict images
def predict_images(image_paths):
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)
        predicted_class = classes[predicted_class_index[0]]


        plt.figure()
        plt.imshow(img)
        plt.title(f"Filename: {os.path.basename(img_path)}\nPredicted: {predicted_class}")
        plt.axis('off')
        plt.show()

# Predict random images
predict_images(random_images)
