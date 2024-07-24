import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set the path to the test data
test_dir = 'Cat & Dog Classification/test1'
output_dir = 'Cat & Dog Classification/Output'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of test image paths and randomly select 10 images
test_images = [os.path.join(test_dir, img) for img in os.listdir(test_dir)]
test_images = random.sample(test_images, 10)

# Load the pre-trained model
model = tf.keras.models.load_model('Cat & Dog Classification/cat&dog.h5')

# Predict and save the results
for img_path in test_images:
    # Load and preprocess the image
    img = load_img(img_path, target_size=(200, 200))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = 'Dog' if prediction >= 0.5 else 'Cat'

    # Print the prediction result
    print(f'Filename: {os.path.basename(img_path)}, Predicted: {predicted_class}')

    # Display the image and prediction result
    plt.figure()
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')

    # Save the result image
    save_path = os.path.join(output_dir, f"{os.path.basename(img_path)}_predicted.png")
    plt.savefig(save_path)
    plt.close()
    print(f'Saved prediction result to {save_path}')
