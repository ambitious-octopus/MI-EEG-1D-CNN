import numpy as np
from data_processing import utils as u
import os
import tensorflow as tf
from tensorflow import keras

#%%
"""
Load only
"""
base_path = "D:"
dir_path = os.path.join(base_path, "eeg_data")

#Load data
x_data = np.load(os.path.join(dir_path, "x_C3_C4.npy"))
y = np.load(os.path.join(dir_path, "y_C3_C4.npy"))
#%%
# REMEMBER = (Height, Width, Channels)
#https://www.frontiersin.org/articles/10.3389/fnhum.2020.00338/full
#todo: Model here https://www.frontiersin.org/files/Articles/559321/fnhum-14-00338-HTML/image_m/fnhum-14-00338-t001.jpg
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(630, (2,25), activation="relu", padding="same", input_shape=(2,641,1)))


model.add(keras.layers.Conv2D())
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D())
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D())
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D())
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Flatten())



