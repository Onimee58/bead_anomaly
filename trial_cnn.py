# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 23:53:17 2022

@author: Saif
"""

#%% import necessary modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import glob, os, shutil
import cv2
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model, image_dataset_from_directory
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split

tf.config.run_functions_eagerly(True)

#%% processing Data
csv_file = 'S0D0206_1.csv'
df = pd.read_csv(csv_file)
time = np.array(df['Time ms'])
faults = np.array(df['Faults'], dtype=str)

#%% moving all to make dataset
'''
for i in tqdm(range(len(faults))):
    if faults[i] == 'nan':
        shutil.move('frames/' + str(time[i])+'.png', 'frames/normal/' + str(time[i])+'.png')
    else:
        shutil.move('frames/' + str(time[i])+'.png', 'frames/abnormal/' + str(time[i])+'.png')
'''
#%% creating teest and train data

batch_size = 1
location = 'frames/'
any_data = 'frames/normal/' + str(time[0])+'.png'
label_mode = 'binary'
im_shape = cv2.imread(any_data).shape
im_size = [512, 512, 3]
seed = 0
class_names = ['normal', 'abnormal']

tr_dataset = image_dataset_from_directory(directory=location, label_mode= label_mode, class_names=class_names,
                                          seed=seed, labels='inferred', image_size=im_size[:-1], 
                                          subset = 'training', batch_size=batch_size, validation_split=.2)

val_dataset = image_dataset_from_directory(directory=location, label_mode= label_mode, class_names=class_names,
                                          seed=seed, labels='inferred', image_size=im_size[:-1],
                                          subset = 'validation', batch_size=batch_size, validation_split=.2)

#%% build model

i = Input(shape=im_size, name= 'I/P Layer')
r = Conv2D(512, 3, activation='relu')(i)
r = Conv2D(512, 3, activation='relu')(r)
r = Conv2D(512, 3, activation='relu')(r)
r = MaxPool2D(3, padding='same')(r)
r = Conv2D(512, 3, activation='relu')(r)
r = Conv2D(512, 3, activation='relu')(r)
r = Conv2D(512, 3, activation='relu')(r)
r = MaxPool2D(3, padding='same')(r)
# r = LSTM(256, return_sequences=True)(r)
r = Dense(64, 'relu')(r)
r = Dense(32, 'relu')(r)
r = Dense(16, 'relu')(r)
o = Dense(2, activation='sigmoid')(r)

model = Model(inputs=i, outputs=o)
model.compile(optimizer=Adam(.001), loss= BinaryCrossentropy(), metrics=['acc', 'mse'])
model.summary()

#%% training

model.fit(tr_dataset, validation_data=val_dataset, epochs=10)






















