#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:34:44 2019

@author: parthsareen
"""

# Part 1 - bulding the cnn
# dataset also available on kaggle

# 10k images total the splitting is the data preprocessing
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Intitializing the ANN
classifier = Sequential()

# Step 1 - Convolution layer built with feature maps
classifier.add(Convolution2D(filters = 32, kernel_size = 3, input_shape = (64, 64, 3), activation = 'relu'))# filters = fmaps, also the number of rows

# Step 2 - Pooling reduces size of feature maps
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second conv layer
classifier.add(Convolution2D(filters = 32, kernel_size = 3, activation = 'relu'))# filters = fmaps, also the number of rows
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# add another with 64
# Step 3 - Flattening
classifier.add(Flatten()) #spatial structure is conserved

# Step 4 - Full Connection, fully connected layers with a classic ann
classifier.add(Dense(units = 128, activation = 'relu'))#hidden layer
# Output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the cnn
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - fitting the cnn to the images
# data augmentation needs to be done to not have overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch = (8000/32),
        epochs = 25,
        validation_data = test_set,
        validation_steps = (2000/32),
        use_multiprocessing = True)

# Part 3 - Making new prediction
