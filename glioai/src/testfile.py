import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from tensorflow.python.keras.models import load_model
model = load_model('/root/glioAI/glioai/models/tumor_prediction.h5')

# route to any of the labaled malignant images that model hasn't seen before 
img_path = ('/root/glioAI/data/tumortest/8 no.jpg')
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
img_data = preprocess_input(x)

# make prediction
rs = model.predict(img_data)
print(rs)

rs[0][0]
rs[0][1]

if rs[0][0] >= 0.9:
    prediction = 'This image is NOT tumorous.'
elif rs[0][0] < 0.9:
    prediction = 'Warning! This image IS tumorous.'

print(prediction)