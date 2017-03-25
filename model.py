#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 11:04:15 2017

@author: raghu
"""

import numpy as np
import pandas as pd
import json
import cv2
from PIL import Image

import model_definition

from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf


img_width, img_height = 120, 40     # dimensions of image
#img_width, img_height = 320, 160     # dimensions of image
nb_epoch = 10


BATCH_SIZE = 64


#------------- Pre-Process Image ----------------------------------------------

# Change image size to 120x60
# Chop off 10 pixels at the top and at the bottom - 120x40
def crop_resize(img):

    # re-size image
    basewidth = 120

    wpercent = (basewidth / (img.size[0]))
    hsize_float = float(img.size[1]) * float(wpercent)

    hsize = int(hsize_float)
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)

    # crop image
    #img = img.crop((0, 19, basewidth, hsize-15))
    img = img.crop((0, 10, basewidth, hsize-10))
    return img

'''
def enhance_contrast(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm
'''
    
'''
def enhance_contrast(rgb):
    b, g, r = cv2.split(rgb)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((red, green, blue))
''' 
# ------------ Augment Data ---------------------------------------------------

# Adjust Steering
def adjust_steering(camera, steering):

    # adjust the steering angle for left / right cameras
    # Discussion in slacl community
    if camera == 'left':
        steering += 0.25
    elif camera == 'right':
        steering -= 0.25

    return steering

    
def perturb_angle(angle):
    new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
    return new_angle

    
# Flip image horizontally
def flip_image(image, steering):

    # decide whether to horizontally flip the image:
    # This is done to reduce the bias for turning left that is present in the training data
    flip_prob = np.random.random()

    if flip_prob > 0.5:
        # flip the image and reverse the steering angle
        steering = -1*steering
        image = cv2.flip(image, 1)

    return image, steering


# brightness adjustments
def adjust_brightness(image):

    image1 = np.transpose(image, (1,2,0))

    # convert to HSV so that its easy to adjust brightness
    image2 = cv2.cvtColor(image1,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image2[:,:,2] = image2[:,:,2]*random_bright

    # convert to RBG again
    image2 = cv2.cvtColor(image2,cv2.COLOR_HSV2RGB)

    image3 = np.transpose(image2, (2,0,1))

    return image3


# ------------ Data Generater -------------------------------------------------

def preprocess_augment(row, train):

    steering = row['steering']

    # randomly choose the camera to take the image from
    camera = np.random.choice(['center', 'left', 'right'])          # commented while training cnn_rnn

    image = load_img(row[camera].strip())                           # change from row[camera] to row['center']
    
    # pre-process : applicable for training and validation
    image = crop_resize(image)
    image = img_to_array(image)
    #image = enhance_contrast(image)
    
    # Augment data applicable only during training
    if (train):
        steering = adjust_steering(camera, steering)                # commented while training cnn_rnn
        image, steering = flip_image(image, steering)               # commented while training cnn_rnn
#        steering = perturb_angle(steering)
        image = adjust_brightness(image)

    return image, steering


#   Ref: https://www.youtube.com/watch?v=bD05uGo_sVI&feature=youtu.be
def get_data_generator(rows, batch_size, train=False):
    N = rows.shape[0]
    batches_per_epoch = N // batch_size

    # Shuffle rows after each epoch
    rows = rows.sample(frac=1).reset_index(drop=True)

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 3, img_height, img_width), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and generate augmented data for each row in the chunk on the fly
        for index, row in rows.loc[start:end].iterrows():
            X_batch[j], y_batch[j] = preprocess_augment(row, train)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch


def get_rnn_input(dataset, time_step=1):
    X, Y = [],[]
    X.append(0.0)
    Y.append(dataset[0])
    for i in range(len(dataset)-time_step):
        X.append(dataset[i])
        Y.append(dataset[i+time_step])
    return np.array(X), np.array(Y)


def get_cnn_rnn_generator(rows, batch_size, train=False):
    N = rows.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_cnn = np.zeros((batch_size, 3, img_height, img_width), dtype=np.float32)
        X_rnn = np.zeros((batch_size,), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and generate augmented data for each row in the chunk on the fly
        for index, row in rows.loc[start:end].iterrows():

            X_cnn[j], y_batch[j] = preprocess_augment(row, train)
            j += 1

        X_rnn, y_batch = get_rnn_input(y_batch, time_step=1)        
        # LSTM expects input format as: [samples, time steps, features].
        X_rnn = np.reshape(X_rnn, (X_rnn.shape[0], 1, 1))
        
        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield [X_cnn, X_rnn], y_batch

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Reading the driving log to create X_train and y_train (steering)
    csv_file = 'driving_log.csv'
    df = pd.read_csv(csv_file)

    # Shuffle rows before train/validation split
    df = df.sample(frac=1).reset_index(drop=True)
    
    row_num = len(df['center'])
    ntrain = int(row_num * 0.8)
    nvalid = row_num - ntrain
    
    train_rows = df.loc[0:ntrain-1]
    valid_rows = df.loc[ntrain:]

    # clear data-frame from memory
    df = None
    
    train_generator = get_data_generator(train_rows, batch_size=BATCH_SIZE, train=True)
#    train_generator = get_cnn_rnn_generator(train_rows, batch_size=BATCH_SIZE, train=True)

    validation_generator = get_data_generator(valid_rows, batch_size=BATCH_SIZE)
#    validation_generator = get_cnn_rnn_generator(valid_rows, batch_size=BATCH_SIZE)    

    sdc_model = model_definition.custom_model()
    #sdc_model = model_definition.nvidia_model()
    #sdc_model = model_definition.commaai_model()
    #sdc_model = model_definition.cnn_rnn()

    nb_train_samples = (ntrain//BATCH_SIZE)*BATCH_SIZE
    nb_validation_samples = (nvalid//BATCH_SIZE)*BATCH_SIZE

    filepath = 'model-{epoch:02d}-{val_loss:0.4f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False)
#    sdc_model.fit_generator(..,..,...,callbacks=[checkpoint, early_stopping])
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=1)
    
    sdc_model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, 
                            nb_epoch=nb_epoch, callbacks=[checkpoint],
                            validation_data=validation_generator, nb_val_samples=nb_validation_samples)
    
    # Ref:  https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    # Ref:  http://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # SAVE MODEL - save the architecture of a model, and not its weights or its training configuration
    with open('model.json', 'w') as outfile:
        json.dump(sdc_model.to_json(), outfile)

    # SAVE WEIGHTS - save the weights of a model
    sdc_model.save_weights('model.h5')
    # model.load_weights('model.h5')


