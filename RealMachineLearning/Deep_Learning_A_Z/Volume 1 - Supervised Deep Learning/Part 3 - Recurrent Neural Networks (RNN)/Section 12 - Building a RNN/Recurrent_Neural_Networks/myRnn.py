#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:43:30 2019

@author: parthsareen
LSTM to capture upward and downward google stock trends
many layers- dropout reg. 
5 years worth of google data

"""
# libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Data preprocessing
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[: ,1:2].values
#using normalization for rnn with sigmoid

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#data structure with 6- timesteps and 1 output
"""
60 timesteps of past information to understand trends from
basically 3 months of data
"""
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i - 60 : i, 0])
    y_train.append(training_set_scaled[i , 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Building the RNN


# Prediction
