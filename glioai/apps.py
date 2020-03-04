from django.apps import AppConfig
import html
import pathlib
import tensorflow as tf
import keras
import sys
from django.apps import AppConfig
from django.conf import settings
import os
import h5py
from tensorflow.keras.models import load_model

class TumorPredictorConfig(AppConfig):
    name = 'glioai'
    path = os.path.join(settings.MODELS, 'mri_tumor.h5')
    model = load_model("/root/brain_tumor/tumorapp/GlioAI/models/mri_tumor.h5")
    MODEL_PATH = ("/root/brain_tumor/tumorapp/GlioAI/models/mri_tumor.h5")
    VGG_PATH = ("https://github.com/fchollet/deep-learning-models/''releases/download/v0.1/''vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    predictor = ImageDataGenerator(model_path = MODEL_PATH/"/root/brain_tumor/tumorapp/GlioAI/models/mri_tumor.h5", 
                                            pretrained_path = VGG_PATH, 
                                            multi_label=True)  

    
    

