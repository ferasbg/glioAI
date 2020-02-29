from django.apps import AppConfig
import html
import pathlib
import os
import tensorflow as tf
import keras
from tensorflow.keras.applications.vgg16 import VGG16
import sys
sys.path.append('../brain_tumor/tumorapp/GlioAI/src/tumor_prediction.py')
from tensorflow.keras.preprocessing.image import ImageDataGenerator


MODELS = os.path.join(BASE_DIR, 'glioai/models')


class WebappConfig(AppConfig):
    name = 'glioai'

    MODEL_PATH = ("/root/brain_tumor/tumorapp/GlioAI/models/mri_tumor.h5")
    VGG_PATH = ("https://github.com/fchollet/deep-learning-models/''releases/download/v0.1/''vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    predictor = ImageDataGenerator(model_path = MODEL_PATH/"/root/brain_tumor/tumorapp/GlioAI/models/mri_tumor.h5", 
                                            pretrained_path = VGG_PATH, 
                                            multi_label=True)  



