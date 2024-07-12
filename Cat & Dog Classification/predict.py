import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import random

from tensorflow.keras.preprocessing.image import load_img, img_to_array

test_dir='Cat & Dog Classification/test1'

test_images=[os.path.join(test_dir,img) for img in os.listdir(test_dir)]
test_images=random.sample(test_images,10)

model=tf.keras.models.load_model('Cat & Dog Classification/cat&dog.h5')

for img in test_images:
    img=load_img(img,target_size=(200,200))
    img_array=img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    prediction=model.predict(img_array)
    if prediction>=0.5:
        print('Dog')
    else:
        print('Cat')
    plt.imshow(img)
    plt.show()