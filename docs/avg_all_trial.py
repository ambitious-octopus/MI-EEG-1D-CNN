#Importing stuff
import os
import sys

import matplotlib.pyplot as plt

print(os.getcwd())
print(sys.path)
sys.path.append("/home/kubasinska/data/repos/eeGNN")
from model_set.models import HopefullNet
import numpy as np
import tensorflow as tf
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)
from sklearn.preprocessing import minmax_scale
tf.autograph.set_verbosity(0)
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


PATH = "/home/kubasinska/datasets/eegbci/paper"

channels = Utils.combinations["e"]


exclude =  [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1,109) if n not in exclude]
#Load data
x, y = Utils.load(channels, subjects, base_path=PATH)

datas = {key: list() for key in np.unique(y)}

for xi, yi in zip(x,y):
    datas[yi].append(xi)

fig, axs = plt.subplots(2)






