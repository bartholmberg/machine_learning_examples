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


