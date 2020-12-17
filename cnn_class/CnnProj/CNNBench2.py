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
import sys, getopt
import argparse
from tensorflow.python.keras.utils.np_utils import to_categorical
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS=int(4024 * 4024 * 4024 / 4 / 3)
argparser = argparse.ArgumentParser(
    description='test picasso judge')

argparser.add_argument(
    '-rw',
    '--refreshWeights',
    help='load weights (h5) and begin learning')


def main(args):
    a = args.rw
num_classes = 2 # picasso or not picasso
# this file is the resnet50 model trained on ImageNet data...
               # "notop" means the file does not include weights for the last layer
               # (prediction layer)
               # in order to allow for transfer learning
weights_notop_path = 'u:\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# declare new Sequential model
 
model = Sequential()

# now let's set up the first layers
model.add(ResNet50(# add a whole ResNet50 model
  include_top=False,          # without the last layer
  weights=weights_notop_path, # and with the "notop" weights file
  pooling='avg' # means collapse extra "channels" into 1D tensor by taking an avg across
                # channels
))

model.add(Dense(64, # 
  activation='softmax'
))
# Now lets add a "Dense" layer to make predictions
model.add(Dense(2, # this last layer just has 2 nodes
  activation='softmax' # apply softmax function to turn values of this layer into probabilities
))


# it learned cool patterns from ImageNet
model.layers[0].trainable = False
#model.compile(optimizer='sgd', # stochastic gradient descent (how to update Dense connections during
                               # training)
model.compile(  optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False),
  loss='categorical_crossentropy', # aka "log loss" -- the cost function to minimize
  #loss='mean_squared_error',
  # so 'optimizer' algorithm will minimize 'loss' function
  metrics=['accuracy'] # ask it to report % of correct predictions
)
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import image
image_size = 224
#data_generator_aug = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1.0,  horizontal_flip=True,brightness_range=[0.2,1.0],rotation_range=5,
#                                   width_shift_range = 0.2,
#                                   height_shift_range = 0.2)
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input,rescale=1.0)





isRefreshWeights = True
isStartFreshWeights = False # start from random, otherwise start from prev
isTrainAndTestSwapped=False

working_train_dir = 'u:\\workTrain'
working_test_dir = 'd:\\workTest'
if isTrainAndTestSwapped:  
    save=working_test_dir 
    working_test_dir=working_train_dir
    working_train_dir=save
    
if (isRefreshWeights):
#   train_generator_aug = data_generator_aug.flow_from_directory(working_train_dir,
#       shuffle=True,
#       target_size=(image_size, image_size),
#       batch_size=20,
#        class_mode='categorical')

#    validation_generator_aug = data_generator_aug.flow_from_directory(working_test_dir,
#       target_size=(image_size, image_size),
#       batch_size=20,
#        class_mode='categorical')

    validation_generator_no_aug = data_generator_no_aug.flow_from_directory(working_test_dir,
        target_size=(image_size, image_size),
        batch_size=20,
        class_mode='categorical')    

misses = 0
falarm = 0
correct = 0
train_generator_no_aug = data_generator_no_aug.flow_from_directory(working_train_dir,
    shuffle=True,
    target_size=(image_size, image_size),
    batch_size=20,
    class_mode='categorical')

if  not isStartFreshWeights:
    model.load_weights("wpicasso.h5")
if not isRefreshWeights:
    #idg = data_generator_no_aug.flow_from_directory(directory=working_test_dir,shuffle=True,
    #   target_size=(image_size, image_size),batch_size=20,class_mode='categorical')
    #for imgs in idg:
    #idg = validation_generator_no_aug
    idg = train_generator_no_aug
    for imgs in idg:
      #idx = (idg.batch_index - 1) * idg.batch_size
      #fn=idg.filenames[idx : idx + idg.batch_size]
      chunkOfPic,labels=idg.next(); #provides next image and label
      chunkOfPic.shape
      labels.shape

      yhat = np.squeeze(model.predict(chunkOfPic))
      #yhat = model.predict(imgs)
      #yclass=yhat.argmax(axis=-1)
      #  yclass = to_categorical(yhat,2)
      yhatf = yhat
      if 0:
        upperThresh = 0.96
        thresh = yhatf[:,1]
        yhat[thresh < upperThresh] = [1,0]
        #thresh = yhatf[:,0]
        #yhat[thresh > upperThresh] = [0,1]
      yhat = np.rint(yhat).astype(int)
      #yclass = np.rint(yclass).astype(int)
      ##labels = np.array(imgs[1][:].astype(int)) # these are all the names but not in order
      #plabel = sorted(labels)[yclass]
      #img = image.array_to_img(a)
      #a = np.squeeze(imgs[0])
      for i in range(0, len(yhat)):
          #print(labels[i][:])
          #if 'not' in fn[i]:  
          #  labels[i][:] = [1,0]
          #else: 
          #  labels[i][:] = [0,1]
          #  print( 'a picasso')
          # print('yclass: ' ,yclass[i][:], yclass.shape, yhat.shape )  
          #         print('Next line of labels')
          # if (labels[i][0] != yclass[i][0][0]) :
          if (labels[i][0] != yhat[i][0]) :
            print('yhat:',yhatf[i][:],'labels:',labels[i][:],'error:' ,yhatf[i][:] - labels[i][:])
            b = np.squeeze(chunkOfPic[i][:])
            isPicasso = (labels[i][0] == 0)
            if isPicasso:
                b = cv2.putText(b, 'Picasso',  (30, 30) , cv2.FONT_ITALIC,  1, (10, 0, 0) , 2, cv2.LINE_AA) 
                misses = misses + 1
            else:
                b = cv2.putText(b, 'Not Picasso',  (30, 30) , cv2.FONT_ITALIC,  1, (0, 0, 10) , 2, cv2.LINE_AA) 
                falarm = falarm + 1
            plt.imshow( b.astype('uint8')-255 )
            plt.show(block=False)
            plt.pause(0.5)
            plt.draw()
          else:
            correct = correct + 1

          print('false alarm: ',falarm,'misses: ',misses,'correct: ',correct)
print("\n\nmodel - train_generator")
if isRefreshWeights:
    history = model.fit_generator(train_generator_no_aug,
      steps_per_epoch=20,
      epochs=23,
      shuffle=True,
      class_weight='auto',
      validation_data=validation_generator_no_aug,
      validation_steps=1)
    model.save_weights("wpicasso.h5")
model.summary()

#da.plotPic2(model,'d:\\workTrain\\picasso')
#yhat = model.predict(x_test[ind,:].reshape(1,784))
if __name__ == "main":
   main(sys.argv[1:])