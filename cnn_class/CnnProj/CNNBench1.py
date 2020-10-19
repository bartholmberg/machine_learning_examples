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

def plotPic(fns,   train_dir = 'D:\\'+ 'train_9\\'):
    for i in range( len(fns)):
        filename=fns[i]
        img = cv2.imread(train_dir + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show(block=True)
        plt.draw()
    return
def moveData(fromList, fromDir,toDir):
    for i in range(len(fromList)):
        fromFile= fromList[i]
        #fn=os.path.basename(fromFile)
        #os.rename(fromDir+fromFile, toDir+fromFile)
        shutil.move(fromDir+fromFile,toDir+fromFile)
    return  
def extract(lst): 
    return [item[0] for item in lst] 
def getData( df, train_dir = 'D:\\'+ 'train_9\\'):
    print("there are " + str(df.shape[0]) + " paintings inside train") 
    # get a dataframe that has rows referring to files starting with 2
    # because we only have those files downloaded currently
    # eg. '2.jpg' or '2640.jpg'
    #mask = (df['filename'].str.startswith('9'))
    #df = df[mask]

    files =[os.path.basename(x) for x in glob.glob(train_dir+ '**/*', recursive=True)]
    
    #files = (x for x in mpath if x.is_file())
    # string of just the artist's hash code
    #aaa = train_2_df[(train_2_df['filename'] == filename)].artist.to_numpy();
    img_artist='1950e9aa6ad878bc2a330880f77ae5a1'
    #img_artist = train_df[(train_df['filename'] == filename)].artist.array[0]
    
    artist_data = df[(df['artist'] == img_artist)]
    not_artist_data = df[(df['artist'] != img_artist)]

    num_artist = len(artist_data)
    print("Picasso has " + str(num_artist) + " paintings inside train")
    allArtists=extract(artist_data.values.tolist())
    c=list(set(allArtists) & set(files))
    return c

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
data_dir=['d:\\train_9\\', 'd:\\train_8\\','d:\\train_7\\','d:\\train_6\\','u:\\train_1\\','u:\\train_2\\','u:\\train_3\\','u:\\train_4\\',  'u:\\train_5\\']
picasso_dir = 'd:\\picasso\\'
allTrainInfo = pd.read_csv('u:\\train_info.csv')
for i in range( len(data_dir)):
    b=getData(allTrainInfo,data_dir[i])
    moveData(b, data_dir[i], picasso_dir);
plotPic(b, data_dir[4])
print(b)
