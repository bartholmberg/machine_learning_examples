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

def plotPic2(model,image_dir = 'D:\\train_9\\'):
    #files =[os.path.basename(x) for x in glob.glob(image_dir+ '**/*', recursive=True)]
    fns=glob.glob(image_dir+ '**/*', recursive=True)
    for i in range( len(fns)):
        filename=fns[i]
        img = cv2.imread(filename)
        b =cv2.resize(img,(224,224))
         
        yhat = model.predict(b.reshape (1, 224, 224, 3))
        img = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        plt.imshow(b)
        plt.show(block=True)
        plt.draw()
    return
num_classes = 2 # picasso or not picasso

# this file is the resnet50 model trained on ImageNet data...
# "notop" means the file does not include weights for the last layer (prediction layer)
# in order to allow for transfer learning
weights_notop_path = 'u:\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# declare new Sequential model
# meaning each layer is in sequence, one after the other. 
# apparently there can be non-sequential neural networks... wow!
model = Sequential()

# now let's set up the first layers
model.add(ResNet50(    # add a whole ResNet50 model
  include_top=False,          # without the last layer
  weights=weights_notop_path, # and with the "notop" weights file
  pooling='avg' # means collapse extra "channels" into 1D tensor by taking an avg across channels
))


# Now lets add a "Dense" layer to make predictions
model.add(Dense(
  num_classes, # this last layer just has 2 nodes
  activation='softmax' # apply softmax function to turn values of this layer into probabilities
))

# do not train the first layer
# because it is already smart
# it learned cool patterns from ImageNet
model.layers[0].trainable = False
model.compile(
  optimizer='sgd', # stochastic gradient descent (how to update Dense connections during training)
  loss='categorical_crossentropy', # aka "log loss" -- the cost function to minimize 
  # so 'optimizer' algorithm will minimize 'loss' function
  metrics=['accuracy'] # ask it to report % of correct predictions
)
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
image_size = 224
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)
working_train_dir='D:\workTrain'
working_test_dir='D:\workTest'
if (False is True):
    train_generator_no_aug = data_generator_no_aug.flow_from_directory(
        working_train_dir,
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

    validation_generator = data_generator_no_aug.flow_from_directory(
        working_test_dir,
        target_size=(image_size, image_size),
        class_mode='categorical')
else:
    data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

    train_generator_with_aug = data_generator_with_aug.flow_from_directory(
        working_train_dir,
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')
    validation_generator = data_generator_with_aug.flow_from_directory(
        working_test_dir,
        target_size=(image_size, image_size),
        class_mode='categorical')
print("\n\nmodel - train_generator_no_aug")
model.load_weights("wpicasso.h5") 
if True is False:
    history = model.fit_generator(
      train_generator_with_aug,
      steps_per_epoch=8,
      validation_data=validation_generator,
      validation_steps=2)
model.summary()
plotPic2(model,working_test_dir+'\\not-picasso')
#yhat = model.predict(x_test[ind,:].reshape(1,784))