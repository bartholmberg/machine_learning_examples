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

fixedSeed = 42 # 480 could work too
np.random.seed(fixedSeed)
tf.set_random_seed(fixedSeed)


picasso_dir = 'd:\\picasso\\'
#data_dir=[picasso_dir,'d:\\train_9\\', 'd:\\train_8\\','d:\\train_7\\','d:\\train_6\\','u:\\train_1\\','u:\\train_2\\','u:\\train_3\\','u:\\train_4\\',  'u:\\train_5\\']
#data_dir = 'u:\\train\\'
data_dir = 'd:\\test\\'
allTrainInfo = pd.read_csv('u:\\all_data_info.csv')
b=ad.getData(allTrainInfo,data_dir)
ad.moveData(b, data_dir,picasso_dir)
print(b )
#ad.plotPic(b, data_dir)
#print(b)

