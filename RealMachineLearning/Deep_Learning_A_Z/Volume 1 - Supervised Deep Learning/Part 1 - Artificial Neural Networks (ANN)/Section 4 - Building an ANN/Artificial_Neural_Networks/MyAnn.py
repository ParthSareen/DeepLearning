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
import matplotlib
matplotlib.use('TkAgg')
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
from keras.layers import Dropout
#Initializing the ANN
"""
We define this by defining the layers
this nn is a classifer
"""

classifier = Sequential()
# adding input and hidden layers
#rectifier for hidden sigmoid for output 
#first layer with dropout
classifier.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu', input_shape = (11,)))
classifier.add(Dropout(rate = 0.1))#start with a rate of 0.1 then raise 10% don't go over 0.5
#second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))#start with a rate of 0.1 then raise 10% don't go over 0.5

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
# predicting a single new observation
"""
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""

new_pred = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))#needs to be a horizontal vector
new_pred = (new_pred >0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#vars with impact: c-score, geography, gender, age, tenure(years been in bank), balance, number of credits
#credit card?, active member?, estimated salary?
# indexes should be from 3-12

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense #for layers

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu', input_shape = (11,)))
    classifier.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'random_uniform', activation = 'sigmoid')) #if 3 categories, use 3 for indp  var, soft max needs to be applied u need to use softmax 
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()

#if data has overfitting - high variance

#dropout regulaisation

'''
hides nodes and lets ann find correlation with one var without being dependant 
on other variables
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#Silences the cpu warnings

#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense #for layers

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu', input_shape = (11,)))
    classifier.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'random_uniform', activation = 'sigmoid')) #if 3 categories, use 3 for indp  var, soft max needs to be applied u need to use softmax 
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
#grid search
parameters = {'batch_size': [25, 32], 
              'nb_epoch' : [100, 500],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
