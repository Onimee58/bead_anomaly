# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 23:21:53 2022

@author: Saif
"""
#%% importing needed modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split

tf.config.run_functions_eagerly(True)

#%% processing Data
csv_file = 'S0D0206_1.csv'
df = pd.read_csv(csv_file)
time = np.array(df['Time ms'])
faults = np.array(df['Faults'], dtype=str)
avg_voltage = np.array(df['Avg Voltage'])
avg_current = np.array(df['Avg Current'])
X = np.array(list(zip(avg_current, avg_voltage)))
X = np.expand_dims(X, axis=1)
Y = np.zeros(len(faults))
Y[faults!='nan'] = 1
Y = np.expand_dims(Y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y , test_size=0.15, random_state=0)

#%% extracting frames in timestamps
vid = 'LCS Test1 WFR 75 TS 10  - Window map - 2021-06-02 14-46-15.mp4'
for tt in tqdm(time):
    cap = cv2.VideoCapture(vid)
    cap.set(cv2.CAP_PROP_POS_MSEC,tt)
    success,image = cap.read()
    if success:
        cv2.imwrite('frames/' + str(tt) + '.png', image)
        #cv2.imshow('video',image)
        #cv2.waitKey()
cap.release()
cv2.destroyAllWindows()

#%% creating training and testing data

i = Input(shape=(1,2), name= 'I/P Layer')
r = LSTM(8, return_sequences=True)(i)
r = LSTM(16, return_sequences=True)(r)
r = LSTM(32, return_sequences=True)(r)
r = LSTM(64, return_sequences=True)(r)
r = LSTM(128, return_sequences=True)(r)
# r = LSTM(256, return_sequences=True)(r)
# r = Dense(64, 'relu')(r)
# r = Dense(32, 'relu')(r)
r = Dense(16, 'relu')(r)
o = Dense(1, activation='sigmoid')(r)

model = Model(inputs=i, outputs=o)
model.compile(optimizer=Adam(.001), loss= BinaryCrossentropy(), metrics=['acc', 'mse'])
model.summary()

#%%
model.fit(X_train, y_train, epochs = 100, batch_size = 512,
          validation_data=[X_test, y_test])
hs = model.history

plt.plot(hs.epoch, hs.history['loss'])

plt.plot(hs.epoch, hs.history['acc'])

plt.plot(hs.epoch, hs.history['mse'])


plt.plot(avg_voltage)




