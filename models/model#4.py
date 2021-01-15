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

# TEST


#########
model = keras.models.Sequential()
model.add(keras.layers.Conv1D(64,2, activation="relu", padding="same", input_shape=[640, 2]))
model.add(keras.layers.MaxPooling1D(1))
model.add(keras.layers.Conv1D(128, 2, activation="relu", padding="same"))
model.add(keras.layers.Conv1D(128, 2, activation="relu", padding="same"))
model.add(keras.layers.MaxPooling1D(1))
model.add(keras.layers.Conv2D(256, (2, 2), activation="relu", padding="same"))
model.add(keras.layers.Conv2D(256, (2, 2), activation="relu", padding="same"))
model.add(keras.layers.MaxPooling1D(1))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(5, activation="relu"))
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test,y_test))
model.predict(x_test[:4])