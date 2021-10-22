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

try:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPU Available")

import pickle

TRAIN_PATH = "/home/kubasinska/datasets/eegbci/seq"

# Find the maximum value in order to interate correctly
max_index = np.max([int(p.name[:-4]) for p in os.scandir(TRAIN_PATH)])

def gen_train():
    global max_index
    for i in range(1, max_index):
        with open(os.path.join(TRAIN_PATH, str(i)+".pkl"), "rb") as file:
            data = pickle.load(file)
        yield np.moveaxis(data[0], 1, 2), data[1]


x_train = tf.data.Dataset.from_generator(gen_train,
                                       output_types=(np.float64, np.float64),
                                       output_shapes=((8, 80, 2),(5,)))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=32, kernel_size=2,
                                                                 activation='relu', padding="same"),
                                          input_shape=(None,80,2)))
# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.SpatialDropout1D(0.5)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3,
#                                                                  activation='relu')))
# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.AvgPool1D(pool_size=2)))
# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.SpatialDropout1D(0.5)))
# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))

# todo: Non flattenare ma passarlo a 2 dimensioni!
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
model.add(tf.keras.layers.LSTM(32))
# model.add(tf.keras.layers.LSTM(8))
# model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(100, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.summary()
model.fit(x = x_train.batch(15), epochs=70)