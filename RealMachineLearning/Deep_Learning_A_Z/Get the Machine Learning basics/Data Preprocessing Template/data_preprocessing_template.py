#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 13:18:09 2018

@author: parthsareen
"""
#importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Data.csv')

#matrix of features
X = dataset.iloc[:, :-1].values #column, column-1, last not included
Y = dataset.iloc[:, -1].values # column, index of last
#FOR MISSING DATA, take mean of all values
"""
#taking care of missing data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer= imputer.fit(X[:,1:3]) #not 2, 3, upper bound is excluded
X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#to not make a name variable being interpretted as larger, we will use dummy vards
ohc = OneHotEncoder(categorical_features = [0])
X = ohc.fit_transform(X).toarray() #column transformer this is gonna be deprecated in .22
"""
#splitting data set into training and tesitng

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) #percent splitting the data set, 10 observations, therefore 2 observations in the test set

##Feature Scaling
#many models follow euclidean distance distance fromula, general
#salary in tutorial case will dominate the e-dist since the age will not have much of an impact on the distance
#f-scaling allows us to take data on the same scale
"""
Standardization: xstand = (x-mean(x))/(std(x))
Normalization: xnorm = (x-min(x))/(max(x) - min(x))

"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)#fit then transform for train just transform for test
X_test = sc_X.transform(X_test) 
"""
Scaling Dummy Vars:
    depends of context , not scaling won't break the model
even if algo not based on e-dist, feature scaling will help the program converge faster

"""
