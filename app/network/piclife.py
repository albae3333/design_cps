#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm_notebook, tnrange
from itertools import chain
#from skimage.io import imread, imshow, concatenate_images
#from skimage.transform import resize
#from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model, model_from_json
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[2]:


import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd 
import imutils
import cv2
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


json_file = open('./app/network/modeladv3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./app/network/modeladv3.h5")
print("Loaded model from disk")
loaded_model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])


# In[4]:


crosssize=20
crossthick=2
rectanglesize=80
imgsize = 80
imgsavesize = 120

x = 320
y = 240


# In[5]:


def cross(imgpar, x, y):
    cv2.rectangle(imgpar, (x-rectanglesize//2,y-rectanglesize//2), (x+rectanglesize//2,y+rectanglesize//2),(255,0,0),crossthick)
    return imgpar


# In[6]:


def predictedmask(piclist, thresh=0.6):
    y_list = []
    for pic in piclist:
        assert pic.shape == (imgsize, imgsize, 3)
            
        imgret = np.zeros((imgsize, imgsize,3), np.uint8)
        for i in range(imgsize):
            for j in range(imgsize):
                if pic[i,j,2] >= thresh:
                    imgret[i,j,2] = 255
                else:
                    imgret[i,j,2] = 0
        y_list.append(imgret)
    return y_list


# In[7]:


pathname = './app/network/images'
i = 605
cap = cv2.VideoCapture(2) # video capture source camera (Here webcam of laptop) 

ret,frame = cap.read() # return a single frame in variable `frame`
#imgrect = frame[y-rectanglesize//2:y+rectanglesize//2, x-rectanglesize//2:x+rectanglesize//2]
img = frame.copy()
img = cross(img,x,y)
runpred=False

while(True):


    cv2.imshow('img1',img) #display the captured image
    key = cv2.waitKey(100)
    ret,frame = cap.read() # return a single frame in variable `frame`
    img = frame.copy()
    img = cross(img,x,y)
    if key&0xFF == ord("+"): 
        rectanglesize = rectanglesize + 2
    if key&0xFF == ord("-"): 
        rectanglesize = rectanglesize - 2
    if key&0xFF == ord('r'):
        if runpred==False:
            runpred=True
        else:
            runpred=False
    if key&0xFF == ord('s') and runpred == False:
        imgsave = frame[y-rectanglesize//2:y+rectanglesize//2, x-rectanglesize//2:x+rectanglesize//2]
        imgsave = cv2.resize(imgsave, (imgsavesize,imgsavesize), interpolation = cv2.INTER_AREA)
        cv2.imwrite(join(pathname, '{}.png'.format(i)),imgsave)
        i += 1
    if runpred==True:
        X_test = []
        imgsave = frame[y-rectanglesize//2:y+rectanglesize//2, x-rectanglesize//2:x+rectanglesize//2]
        imgsave = cv2.resize(imgsave, (imgsize,imgsize), interpolation = cv2.INTER_AREA)
        X_test.append(imgsave)
        X_test = np.array(X_test, dtype=np.float32)       
        X_test -= X_test.mean()
        X_test /= X_test.std()
        predictions_test = loaded_model.predict(X_test, batch_size=1, verbose=0)
        predictions_pics = predictedmask(predictions_test,0.5)
        imgsave = cv2.addWeighted(imgsave, 0.7, predictions_pics[0], 0.3, 0)
        imgrect = cv2.resize(imgsave, (rectanglesize, rectanglesize), interpolation = cv2.INTER_AREA)
        img[y-rectanglesize//2:y+rectanglesize//2,x-rectanglesize//2:x+rectanglesize//2] = imgrect    
        
    if key&0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


# In[12]:


cv2.destroyAllWindows()
cap.release()


# In[ ]:




