import numpy as np
import os
import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

graph = tf.get_default_graph()
model = tf.keras.models.load_model('/root/glioAI/glioai/models/tumor_prediction.h5')
print("done")