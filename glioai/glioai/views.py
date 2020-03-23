import numpy as np 
import pandas as pd 
import os
import base64
import io
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
from tensorflow import keras
from PIL import Image
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from django.shortcuts import render
global graph
global model
global session
import h5py
import requests
from django.conf import settings

import warnings
warnings.filterwarnings("ignore")

# launch graph in session
session = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
with session.graph.as_default():
    tf.keras.backend.set_session(session)
    model = tf.keras.models.load_model('/root/glioAI/glioai/models/tumor_prediction.h5', compile=False)

# initialize global variables
# init = tf.global_variables_initializer()
# session.run(init)

def home(request):
    return render(request, 'index.html')

def analysis(request, *args, **kwargs):
    # load image
    img_path = request.FILES['myfile'] 
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    # convert image to an array
    x = image.img_to_array(img)
    # expand image dimensions
    x = preprocess_input(x)
    x = np.expand_dims(x,axis=0)
    with session.graph.as_default():
        tf.keras.backend.set_session(session)
        rs = model.predict(x, **kwargs)
    result = ""
    if rs[0][0] == 1:
        result = "This image is NOT tumorous."
    else:
        result = "Warning! This image IS tumorous."
    
    return render(request, 'analysis.html', {'result': result })


    # if request.method == 'POST' and request.FILES['myfile']:
    #     #user makes a post req via image upload to receive output (diagnosis)
    #     post = request.method == 'POST'
    #     myfile = request.FILES['myfile']
    #     img = tf.keras.preprocessing.image.image.load_img(myfile, target_size=(224,224))
    #     img = image.img_to_array(img)
    #     img = np.expand_dims(img, axis=0)
    #     with graph.as_default():
    #         pred = model.predict(img)
    #     img_data = preprocess_input(img)
    #     # make prediction
    #     rs = model.predict(img_data)
    #     # print(rs)

    #     return render(request,"/root/glio.ai/glioai/templates/prediction.html", {
    #     'result': rs})
    # else:
    #     return render(request, "/root/glio_ai/glio.ai/glioai/templates/prediction.html")