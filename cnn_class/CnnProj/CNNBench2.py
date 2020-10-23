from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

import os 
import cv2
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras
from pathlib import *
import glob
import shutil
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dataAssemble as da
import matplotlib.pyplot as plt

num_classes = 2 # picasso or not picasso
# this file is the resnet50 model trained on ImageNet data...
# "notop" means the file does not include weights for the last layer (prediction layer)
# in order to allow for transfer learning
weights_notop_path = 'u:\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# declare new Sequential model
 

model = Sequential()

# now let's set up the first layers
model.add(ResNet50(    # add a whole ResNet50 model
  include_top=False,          # without the last layer
  weights=weights_notop_path, # and with the "notop" weights file
  pooling='avg' # means collapse extra "channels" into 1D tensor by taking an avg across channels
))


# Now lets add a "Dense" layer to make predictions
model.add(Dense(
  2, # this last layer just has 2 nodes
  activation='softmax' # apply softmax function to turn values of this layer into probabilities
))

# do not train the first layer
# because it is already smart
# it learned cool patterns from ImageNet
model.layers[0].trainable = False
model.compile(
  optimizer='sgd', # stochastic gradient descent (how to update Dense connections during training)
  loss='categorical_crossentropy', # aka "log loss" -- the cost function to minimize 
  #loss='mean_squared_error',
  # so 'optimizer' algorithm will minimize 'loss' function
  metrics=['accuracy'] # ask it to report % of correct predictions
)
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import image
image_size = 224
data_generator_aug = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1.0,  horizontal_flip=True,vertical_flip=True,rotation_range=180,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

working_train_dir='D:\\workTrain'
working_test_dir='D:\\workTest'
isRefreshWeights=True
if (isRefreshWeights):
    train_generator_aug = data_generator_aug.flow_from_directory(working_train_dir,
        target_size=(image_size, image_size),
        batch_size=50,
        class_mode='categorical')
    validation_generator_aug = data_generator_aug.flow_from_directory(working_test_dir,
        target_size=(image_size, image_size),
        batch_size=50,
        class_mode='categorical')
    validation_generator_no_aug = data_generator_no_aug.flow_from_directory(working_test_dir,
        target_size=(image_size, image_size),
        batch_size=50,
        class_mode='categorical')
    
model.load_weights("wpicasso.h5")  
misses=0
falarm=0
correct=0
if not isRefreshWeights:
    model.load_weights("wpicasso.h5")    
if not isRefreshWeights:
    idg =data_generator_no_aug.flow_from_directory(directory=working_test_dir,
       target_size=(image_size, image_size),batch_size=50,class_mode='categorical')
    for imgs in idg:
      print('false alarm: ',falarm,'misses: ',misses,'correct: ',correct)
      yhat=np.squeeze(model.predict(imgs))
      yhat = np.rint(yhat).astype(int)
      labels =np.array(imgs[1][:].astype(int))
      #img = image.array_to_img(a)
      a=np.squeeze(imgs[0])
      for i in range(0,50):
          if ( labels[i][0] != yhat[i][0] ) :
            b=np.squeeze(a[i,:,:])
            isPicasso=(labels[i][0]==0)
            if isPicasso:
                b = cv2.putText(b, 'Picasso',  (30, 30) , cv2.FONT_ITALIC,  1, (10, 0, 0) , 2, cv2.LINE_AA) 
                misses=misses+1
            else:
                b = cv2.putText(b, 'Not Picasso',  (30, 30) , cv2.FONT_ITALIC,  1, (0, 0, 10) , 2, cv2.LINE_AA) 
                falarm=falarm+1
            plt.imshow(b)
            plt.show(block=True)
            plt.draw()
          else:
            correct=correct+1
print("\n\nmodel - train_generator")
if isRefreshWeights:
    history = model.fit_generator(
      train_generator_aug,
      steps_per_epoch=10,
      epochs=10,
      #class_weight={0:85,1:15},
      validation_data=validation_generator_aug,
      validation_steps=3)
    model.save_weights("wpicasso.h5")
model.summary()

da.plotPic2(model,'d:\\workTrain\\picasso')
#yhat = model.predict(x_test[ind,:].reshape(1,784))