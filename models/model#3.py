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
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
x_train, x_test, y_train, y_test = train_test_split(x_data_scale, y_categorical, test_size=0.20, random_state=42)
#%%
#Convolution Neural Network
# [samples, time steps, features].
real_x_train = x_train.reshape(14808, 640, 2)
real_x_test = x_test.reshape(3703, 640, 2)

model = keras.models.Sequential()
model.add(keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(640, 2)))
model.add(keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(real_x_train, y_train, epochs=30, shuffle=True, validation_data=(real_x_test,y_test))

#todo: shuffle data?
#todo: Perchè sul validation va malissimo, overfitta?
#todo: Che tipo di regolarizzazione devo fare?

# prediction = model.predict(real_x_test[:4])