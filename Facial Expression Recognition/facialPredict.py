from tensorflow.keras.preprocessing import image
from tensorflow import keras
import os
import numpy as np
import random
from glob import glob
import matplotlib.pyplot as plt

train_dir = 'Facial Expression Recognition/fer2013/train'
test_dir = 'Facial Expression Recognition/fer2013/test'
model_path = 'Facial Expression Recognition/facialExpression.h5'
classes = os.listdir(train_dir)


model = keras.models.load_model(model_path)

def preprocess_image(img_path):
    img = image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_emotion(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = classes[predicted_class]
    return predicted_label


def display_random_predictions(test_dir, num_images=5):
    plt.figure(figsize=(20, 10))
    all_images = []
    for label in os.listdir(test_dir):
        all_images.extend(glob(os.path.join(test_dir, label, '*')))
    
    random_images = random.sample(all_images, num_images)
    
    for i, img_path in enumerate(random_images):
        predicted_label = predict_emotion(img_path)
        img = image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
        img_array = image.img_to_array(img)
        
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img_array.squeeze(), cmap='gray')
        plt.title(f'Predicted: {predicted_label}')
        plt.axis('off')
    
    plt.show()

display_random_predictions(test_dir, num_images=10)