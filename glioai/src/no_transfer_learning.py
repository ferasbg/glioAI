# MIT License
# Copyright (c) 2019 Feras Baig
# model without transfer learning

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import preprocessing
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.models import Sequential
import numpy as np

# add all layers of neural net
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=2, activation='sigmoid'))



from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# train model
train_datagen=ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_generator=train_datagen.flow_from_directory('/root/glio.ai/data/braintumordata', 
color_mode='rgb', 
batch_size=192, 
class_mode='categorical', 
shuffle=True)



print(train_generator.n)
print(train_generator.batch_size)
print(239//32)



classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
step_size_train=train_generator.n//train_generator.batch_size
r = classifier.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=25)

import matplotlib.pyplot as plt
print(r.history.keys())

# loss
plt.plot(r.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# accuracy

plt.plot(r.history['accuracy'])
plt.title('Model Accuracy')
plt.legend(['Training Accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()



