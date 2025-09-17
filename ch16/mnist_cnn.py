# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 15:27:12 2025

@author: user
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)/255.0
y_train = to_categorical(y_train,10)

# 모델
model = Sequential([
    Conv2D(32,3,padding='same',activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,padding='same'),
    Conv2D(64,3,padding='same',activation='relu'),
    MaxPooling2D(2,padding='same'),
    Flatten(),
    Dense(256,activation='relu'),
    Dense(10,activation='softmax', name='prob')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 학습
model.fit(x_train, y_train, epochs=5, batch_size=100)

# HDF5 저장
model.save("mnist_cnn.h5")
print("Model saved as HDF5 format")
