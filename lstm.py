#Importing stuff
import os
import sys
print(os.getcwd())
print(sys.path)
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
from sklearn.preprocessing import minmax_scale
tf.autograph.set_verbosity(0)
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
import pickle

TRAIN_PATH = "/home/kubasinska/datasets/eegbci/seq"

def gen_train():
    for p in os.scandir(TRAIN_PATH):
        with open(p.path, "rb") as file:
            data = pickle.load(file)
        yield np.moveaxis(data[0], 1, 2), data[1]

x_train = tf.data.Dataset.from_generator(gen_train,
                                       output_types=(np.float64, np.float64),
                                       output_shapes=((8, 80, 2),(5,)))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=6,
                                                                 activation='relu'),
                                          input_shape=(None,80,2)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.SpatialDropout1D(0.5)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                                                                 activation='relu')))
# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.AvgPool1D(pool_size=2)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.SpatialDropout1D(0.5)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))

model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
model.add(tf.keras.layers.LSTM(16))
# model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.summary()
model.fit(x = x_train.batch(32), epochs=30)