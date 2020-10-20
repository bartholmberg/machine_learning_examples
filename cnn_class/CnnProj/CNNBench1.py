import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import *
import glob
import shutil
#test git access
from tensorflow.python.keras.applications import ResNet50
import dataAssemble as ad

my_seed = 42 # 480 could work too
np.random.seed(my_seed)
tf.set_random_seed(my_seed)
model = keras.Sequential()
model.add(layers.Conv2D(filters=32, activation='relu', kernel_size=3, strides=(3, 3), input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=32, activation='relu',kernel_size=3, strides=(3, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, activation='relu', kernel_size=3, strides=(3, 3)))
model.add(layers.MaxPooling2D(pool_size=(1, 1)))
model.add(layers.Flatten())  
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
a=model.summary()
print(a)

picasso_dir = 'u:\\picasso\\'
data_dir=[picasso_dir,'d:\\train_9\\', 'd:\\train_8\\','d:\\train_7\\','d:\\train_6\\','u:\\train_1\\','u:\\train_2\\','u:\\train_3\\','u:\\train_4\\',  'u:\\train_5\\']
allTrainInfo = pd.read_csv('u:\\train_info.csv')
b=ad.getData(allTrainInfo,data_dir[0])
ad.plotPic(b, data_dir[0])
print(b)
