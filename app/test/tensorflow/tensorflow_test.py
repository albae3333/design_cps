from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv
import random
import PIL.Image

DATADIR = "app/test/tensorflow/dataset"
CATEGORIES = ["Rohr", "keinRohr"]
BATCH_SIZE = 100
EPOCHS = 5
IMG_HEIGHT = 100
IMG_WIDTH = 100

num_rohr_tr = len(os.listdir(DATADIR + "/training/rohr"))
num_other_tr = len(os.listdir(DATADIR+"/training/other"))

num_rohr_val = len(os.listdir(DATADIR + "/validation/rohr"))
num_other_val = len(os.listdir(DATADIR+"/validation/other"))

total_train = num_rohr_tr + num_other_tr
total_val = num_rohr_val + num_other_val

training_data = []


""" def create_training_data():

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        category_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv.imread(os.path.join(path, img))
            reshaped_array = cv.resize(img_array, (IMG_HEIGHT, IMG_WIDTH))
            training_data.append([reshaped_array, category_num])

    random.shuffle(training_data)


create_training_data() """

train_image_generator = ImageDataGenerator(
    rescale=1./255, horizontal_flip=True, rotation_range=45)
validation_image_generator = ImageDataGenerator(
    rescale=1./255, horizontal_flip=True, rotation_range=45)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=DATADIR+"/validation",
                                                           shuffle=True,
                                                           target_size=(
                                                               IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=1,
                                                              directory=DATADIR+"/training",
                                                              target_size=(
                                                                  IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=total_val
)

print(len(training_data))
print(training_data[0])
