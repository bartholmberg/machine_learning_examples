import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.python.keras.applications import ResNet50
def GetData():
    filename = "2640.jpg"
    filename="24792.jpg"
    data_dir = "C:\\repo\\machine_learning_examples\\cnn_class\\"
    train_dir = data_dir + "train_2\\"
    img = cv2.imread(train_dir + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show(block=True)
    plt.draw()
    df = pd.read_csv(data_dir + 'train_info.csv')
    print("there are " + str(df.shape[0]) + " paintings inside train_2") 
    # get a dataframe that has rows referring to files starting with 2
    # because we only have those files downloaded currently
    # eg. '2.jpg' or '2640.jpg'
    mask = (df['filename'].str.startswith('2'))
    train_2_df = df[mask]

    # string of just the artist's hash code
    aaa = train_2_df[(train_2_df['filename'] == filename)].artist.to_numpy();

    img_artist = train_2_df[(train_2_df['filename'] == filename)].artist.array[0]
    
    artist_data = train_2_df[(train_2_df['artist'] == img_artist)]
    not_artist_data = train_2_df[(train_2_df['artist'] != img_artist)]

    num_artist = len(artist_data)
    print("Picasso has " + str(num_artist) + " paintings inside train_2")
    return artist_data

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
b=GetData()
print(b)
