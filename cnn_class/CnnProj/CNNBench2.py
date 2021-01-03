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
  activation='relu'
))
# Now lets add a "Dense" layer to make predictions
model.add(Dense(2, # this last layer just has 2 nodes
  activation='softmax' # apply softmax function to turn values of this layer into probabilities
))


# it learned cool patterns from ImageNet
model.layers[0].trainable = False
#model.compile(optimizer='sgd', # stochastic gradient descent (how to update Dense connections during
tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)                            # training)
#model.compile(  optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False),
model.compile(  optimizer=keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07,name='Adadelta'),
  #oss='sparse_categorical_crossentropy',
  loss='mean_squared_error',
  # so 'optimizer' algorithm will minimize 'loss' function
  #metics=['accuracy',BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)],
  metrics=['accuracy','TruePositives','FalsePositives'] # ask it to report % of correct predictions
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
        batch_size=40,
#        labels='inferred',
#        class_names=['non-picasso', 'picasso'],
#        class_mode='binary')   
        class_mode='categorical')    

misses = 0
falarm = 0
correct = 0
train_generator_no_aug = data_generator_no_aug.flow_from_directory(working_train_dir, 
    shuffle=True,
    target_size=(image_size, image_size),
    batch_size=40,
#    class_mode='binary')   
    class_mode='categorical')

if  not isStartFreshWeights:
    model.load_weights("wpicasso.h5")
if not isRefreshWeights:
    #idg = data_generator_no_aug.flow_from_directory(directory=working_test_dir,shuffle=True,
    #   target_size=(image_size, image_size),batch_size=20,class_mode='categorical')
    #for imgs in idg:
    #idg = validation_generator_no_aug
    idg = train_generator_no_aug
    #for imgs in idg:
    #for j in range(1,len(idg)
    chunkOfPic,labels=idg.next();
    while len(chunkOfPic) !=0 :
      #idx = (idg.batch_index - 1) * idg.batch_size
      #fn=idg.filenames[idx : idx + idg.batch_size]
    
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
        upperThresh = 0.96
      yhat = np.rint(yhat).astype(int)

      for i in range(0, len(yhat)-1):
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
      chunkOfPic,labels=idg.next(); #provides next image and label

print("\n\nmodel - train_generator")

if isRefreshWeights:
    #history = model.fit_generator(generator=train_generator_no_aug,
    #  steps_per_epoch=20,
    #  epochs=5,
    #  use_multiprocessing=True,
    #  workers=2,
    #  shuffle=True,
    #  class_weight='auto',
    #  validation_data=validation_ge
    #chunkOfPic,labels=train_generator_no_aug.next()
    chunkOfPic,labels=validation_generator_no_aug.next()
    i=0;
    detectionThreshold=0.02
    while(len(chunkOfPic)>0) :
        history = model.fit(
          x=chunkOfPic,
          y=labels,
          batch_size=20,
          steps_per_epoch=1,
          epochs=2,
          use_multiprocessing=True,
          workers=2,
          shuffle=True)
        yhat = np.squeeze(model.predict(chunkOfPic))
      
        labx=labels[:,0] 
        yhatx=yhat[:,0]
        yhatx=(yhatx > detectionThreshold).astype(int)
        errxf=labx-yhat[:,0]
        errx=(labx-yhatx).astype(int)
        m_acc = history.history.get('acc')[-1] 
        print( "Accuracy: ",m_acc)
        # only get new data if the accuracy is good enough
        # otherwise train on the same data
        if ( m_acc > 0.78 ) :
            chunkOfPic,labels=train_generator_no_aug.next()
            print( "Good Accuracy, get new chunk")
        else:
            print( "Bad Accuracy, train on same chunk")
        if (i%10 ==0) :
           model.save_weights("wpicasso.h5")
           print( ["{:0.1f}".format(x) for x in labx ]  )
           print( ["{:0.1f}".format(x) for x in yhatx ]  )
           print( ["{:0.1f}".format(x) for x in errx ]  )
           print( ["{:0.3f}".format(x) for x in errxf ]  )
        error_index = np.squeeze( np.where((errx ==1) | (errx == -1)) )
        for eind in error_index:
            b = np.squeeze(chunkOfPic[eind][:])
            if( labx[eind] > 0 ):
                b = cv2.putText(b, 'Picasso',  (30, 30) , cv2.FONT_ITALIC,  1, (10, 0, 0) , 2, cv2.LINE_AA) 
            else:
                b = cv2.putText(b, 'Not Picasso',  (30, 30) , cv2.FONT_ITALIC,  1, (0, 0, 10) , 2, cv2.LINE_AA) 

            plt.imshow( b.astype('uint8')+120)
            plt.show(block=False)
            plt.pause(0.5)
            plt.draw()
        i += 1
    model.save_weights("wpicasso.h5")

    model.summary()

#da.plotPic2(model,'d:\\workTrain\\picasso')
#yhat = model.predict(x_test[ind,:].reshape(1,784))
if __name__ == "main":
   main(sys.argv[1:])