# endpoint for model prediction

'''
TODO:
- set trained model as endpoint
- return response on HTML rendered page after HTTP request hits trained model and outputs prediction
    - output HTML page with response

'''


import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


from tensorflow.keras.models import load_model
model = load_model('/root/glioAI/glioai/models/mri_tumor.h5') 
# image path should be set to the http response from views.py, and stored in django sql database
img_path = ('/root/glioAI/data/tumortest/Y20.jpg') 
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
img_data = preprocess_input(x)

rs = model.predict(img_data)
print(rs)

rs[0][0]

rs[0][1]

if rs[0][0] == 1:
    prediction = 'This image is NOT tumorous.'
else:
    prediction = 'Warning! This image IS tumorous.'

print(prediction)
