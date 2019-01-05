#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:58:01 2018

@author: parthsareen
Business problem is that why are customers leaving?
binary outcome
"""

# Part 1 Data Preproccessing, classification problem and a binary outcome.
# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values #upper bound exlcluded
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])# gets the country column
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Let's make the ANN!
#Keras and stuff:
import keras
from keras.models import Sequential
from keras.layers import Dense #for layers

#Initializing the ANN
'''
We define this by defining the layers
this nn is a classifer
'''
classifier = Sequential()
# adding input and hidden layers
#rectifier for hidden sigmoid for output
classifier.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu', input_shape = (11,)))
#second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu'))

#output layer
classifier.add(Dense(units = 1, kernel_initializer = 'random_uniform', activation = 'sigmoid')) #if 3 categories, use 3 for indp  var, soft max needs to be applied u need to use softmax 
 
#compiling the ANN, apply stochastic gradient descent, adam algo, loss function from intuition tuts, binary for one cat, cross entrop for more than 1
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#threshold
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#vars with impact: c-score, geography, gender, age, tenure(years been in bank), balance, number of credits
#credit card?, active member?, estimated salary?
# indexes should be from 3-12
