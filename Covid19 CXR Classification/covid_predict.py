import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

# Set model path, data path, and output path
model_path = 'Covid19 CXR Classification/covid.h5'
data_path = 'Covid19 CXR Classification/COVID-19_Radiography_Dataset'
output_path = 'Covid19 CXR Classification/Output'
classes = os.listdir(data_path)

# Create output directory if it does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)


model = tf.keras.models.load_model(model_path)
print("Model loaded.")

# Get random images for prediction
image_paths = []
for class_name in classes:
    class_path = os.path.join(data_path, class_name)
    for img_file in os.listdir(class_path):
        image_paths.append(os.path.join(class_path, img_file))

# Get 20 random images for prediction
random_images = random.sample(image_paths, 20)

# Predict and save images
def predict_and_save_images(image_paths):
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


        save_path = os.path.join(output_path, f"{os.path.basename(img_path)}_predicted.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved prediction result to {save_path}")


predict_and_save_images(random_images)
