#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 11:04:15 2017

@author: raghu
"""

# Fix error with TF and Keras

import tensorflow as tf
tf.python.control_flow_ops = tf


from keras.models import Sequential
#from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Activation, Flatten, Lambda, SpatialDropout2D
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import LSTM
from keras.layers import Merge
from keras.layers.convolutional import Cropping2D


#from keras.layers.normalization import BatchNormalization
#from keras.regularizers import l2, activity_l2

# dimensions of our images.
img_width, img_height = 40, 120
batch_size = 64
# img_width, img_height = 66, 200

# rescaling inputs from 0-255 to 0-1
def normalize(image):
    cast_image = tf.cast(image, tf.float32)
    reshaped_image = image = tf.transpose(cast_image, (0, 3, 1, 2))
    reshaped_image.set_shape([None,3,66,200])
    return reshaped_image / 255.0

def resize(image):
    #import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive.py
    image = tf.transpose(image, (0, 2, 3, 1))
    return tf.image.resize_images(image, (66, 200))


#Total params: 252,219
#Trainable params: 252,219
#Non-trainable params: 0
def nvidia_model():
    
    # Create the Sequential model
    model = Sequential()

    # No. of units to reduce at the top / bottom of the image - (width, height)
    model.add(Cropping2D(cropping=((13, 0), (9, 0)), input_shape=(3,160,320)))
    model.add(Lambda(resize))

    #model.add(Lambda(normalize), input_shape=(3, img_width, img_height))
    model.add(Lambda(normalize))

    model.add(Convolution2D(24, 5, 5, activation='elu', border_mode='valid', subsample=(2,2)))
    model.add(SpatialDropout2D(0.2))
    
    model.add(Convolution2D(36, 5, 5, activation='elu', border_mode='valid', subsample=(2,2)))
    model.add(SpatialDropout2D(0.2))
    
    model.add(Convolution2D(48, 5, 5, activation='elu', border_mode='valid', subsample=(2,2)))
    model.add(SpatialDropout2D(0.2))
    
    model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='valid', subsample=(1,1)))
    model.add(SpatialDropout2D(0.2))
    
    model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='valid', subsample=(1,1)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Activation('elu'))
    
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, name='y_pred'))

    model.compile('adam', 'mean_squared_error')

    return model


# https://github.com/commaai/research/blob/master/train_steering_model.py
#Total params: 854,641
#Trainable params: 854,641
#Non-trainable params: 0

def commaai_model():
    
    model = Sequential()

    # rescaling inputs from 0-255 to 0-1

#    model.add(Lambda(lambda x: tf.cast((x/255.0), tf.float32), input_shape=(3, img_width, img_height)))
    model.add(Lambda(lambda x: x / 255.0, input_shape=(3, img_width, img_height)))
    
    model.add(Convolution2D(16, 8, 8, activation='relu', border_mode='same', subsample=(4,4)))
    model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='same', subsample=(2,2)))
    model.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same', subsample=(2,2)))
    
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile('adam', 'mean_squared_error')

    return model
    

# Keras ex  https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
# Comma.ai  https://github.com/commaai/research/blob/master/train_steering_model.py

#Total params: 207,891
#Trainable params: 207,891
#Non-trainable params: 0

def define_cnn():

    # Create the Sequential model
    model = Sequential()

    # Add batch_normalization
    #model.add(BatchNormalization(input_shape=(3, img_width, img_height)))

    # rescaling inputs from 0-255 to 0-1
    model.add(Lambda(lambda x: x / 255.0, input_shape=(3, img_width, img_height)))

#    model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='valid', subsample=(1,1)))
    model.add(Convolution2D(32, 3, 3, activation='elu', border_mode='same', subsample=(2,2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
#    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='valid', subsample=(1,1)))
    model.add(Convolution2D(32, 3, 3, activation='elu', border_mode='same', subsample=(2,2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
#    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid', subsample=(1,1)))
    model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='same', subsample=(1,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
#    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    
#    model.add(Dense(800, W_regularizer=l2(0.0001), activity_regularizer=activity_l2(0.0001)))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))

#    model.add(Dense(400, W_regularizer=l2(0.0001), activity_regularizer=activity_l2(0.0001)))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.5))

#    model.add(Dense(80, W_regularizer=l2(0.0001), activity_regularizer=activity_l2(0.0001)))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    
    return model

    
def custom_model():

    model = define_cnn()
    model.load_weights('model150_47.h5')

#    model.compile('adam', 'mean_squared_error', ['accuracy'])
    model.compile('adam', 'mean_squared_error')

    return model
    

def define_rnn():

    model = Sequential()
    model.add(LSTM(16, input_dim=(1)))
    model.add((Dense(1)))
    return model

    

def cnn_rnn():

    cnn_model = define_cnn()
    cnn_model.load_weights('model_pass2.h5')
    
    rnn_model = define_rnn()
    
    # let's concatenate these 2 vector sequences.
    model = Sequential()
    model.add(Merge([cnn_model, rnn_model], mode='concat', concat_axis=-1))

    model.add(Dense(1))
    model.compile('adam', 'mean_squared_error')
    
    return model
    
