# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:59:53 2019

@author: Dr.C
"""

#use tensorflow to predict rotor temp under varying conditions
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks



data = pd.read_csv("pmsm_temperature_data.csv")
print (data.columns)

p4 = data[data["profile_id"] ==4]
print (p4.head())
#train and build model
#choose profile
#choose feature you want to predict
#choose number of epochs to train for 
def trainForRotorTemp(train_frame, num_epochs):
    train_X = train_frame.drop(columns = ['profile_id', 'pm','stator_yoke', 'stator_tooth', 'stator_winding']) 
    train_Y = train_frame[['pm','stator_yoke', 'stator_tooth', 'stator_winding']]
    
    test_X = train_X
    n_cols = train_X.shape[1]
    model = keras.Sequential()
    
    model.add(keras.layers.Dense((len(train_X.columns)+1), activation='relu',input_shape=(n_cols,)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(64, activation ='relu'))
    model.add(keras.layers.Dense(len(train_Y.columns)))
    
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, 
                                            patience=5, min_lr=0.001)
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])
    model.fit(train_X, train_Y, validation_split = 0.2,
              epochs = num_epochs, shuffle = True,
              callbacks =[reduce_lr])
    X_predictions = model.predict(test_X)
    XP = pd.DataFrame(X_predictions, columns=['pm','stator_yoke', 'stator_tooth', 'stator_winding'])
    print (XP.head())

    