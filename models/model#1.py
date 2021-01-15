import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Check dll library
tf.test.gpu_device_name()


"""
Load only and split
"""
base_path = "D:"
dir_path = os.path.join(base_path, "eeg_data")
#Load data
x_data = np.load(os.path.join(dir_path, "x_C3_C4.npy"))
y = np.load(os.path.join(dir_path, "y_C3_C4.npy"))
x_data = Utils.cut_width(x_data)
#Scale
scaler = StandardScaler()
x_data_scale = scaler.fit_transform(x_data.reshape(-1, x_data.shape[-1])).reshape(x_data.shape)

y_resh = y.reshape(y.shape[0], 1)
y_categorical = keras.utils.to_categorical(y_resh, 5)
x_train, x_test, y_train, y_test = train_test_split(x_data_scale, y, test_size=0.20, random_state=42)
#%%
#Test simple model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[x_train.shape[1], x_train.shape[2]]))
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dense(5, activation="relu"))
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test,y_test))
model.predict(x_test[:4])







