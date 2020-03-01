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
    model = load_model('mri_tumor.h5')
    

