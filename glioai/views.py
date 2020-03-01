
from django.shortcuts import render
import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
global graph,model
import requests


graph = tf.get_default_graph()

print("glio.ai loading.......")
model = tf.keras.models.load_model('/glio.ai/glioai/models/mri_tumor.h5')
print("...Model loaded!")


def prediction(request):
    if request.method == 'POST' and request.FILES['myfile']:
        post = request.method == 'POST'
        myfile = request.FILES['myfile']
        img = tf.keras.preprocessing.image.image.load_img(myfile, target_size=(224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        with graph.as_default():
            pred = model.predict(img)
        img_data = preprocess_input(img)
        # make prediction
        rs = model.predict(img_data)
        print(rs)
        return render(request,"/glio.ai/glioai/templates/prediction.html", {
        'result': rs})
    else:
        return render(request, "/glio.ai/glioai/templates/prediction.html")