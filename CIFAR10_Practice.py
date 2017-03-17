
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from keras.constraints import maxnorm
from keras.datasets import cifar10

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import cv2
import glob
import os
import re



# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0



# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# Reinstallise models 
img_size = 32

def cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, img_size, img_size), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


model2 = cnn_model()
modeladam = cnn_model()
modelada = cnn_model()

lr = 0.01

sgd = SGD(lr=lr, decay=1e-5, momentum=0.8, nesterov=True)

model2.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])

modeladam.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

modelada.compile(loss='categorical_crossentropy',
          optimizer='adadelta',
          metrics=['accuracy'])


#def lr_schedule(epoch):
    #return lr*(0.1**int(epoch/10))


# In[18]:

# fitting the model
batch_size = 128
epochs = 30

modelada.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs)


# In[20]:

# Final evaluation of the model
scores = model2.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


##
## attempt with leaky Relu
from keras.layers.advanced_activations import LeakyReLU

# Reinstallise models 
img_size = 32

def cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, img_size, img_size), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, W_constraint=maxnorm(3)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.2))
    model.add(Dense(512, W_constraint=maxnorm(3)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


model5 = cnn_model()

##
model5.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

nb_epoch =50
batch_size = 128
model5.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, nb_epoch=nb_epoch)

# In[]:

# Final evaluation of the model
scores = model5.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))





