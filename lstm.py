#Importing stuff
import os
import sys
print(os.getcwd())
print(sys.path)
import numpy as np
import tensorflow as tf
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
from sklearn.preprocessing import minmax_scale
tf.autograph.set_verbosity(0)
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
import pickle

TRAIN_PATH = "C:\\datasets\\generated"

rawx = list()
rawy = list()

for p in os.scandir(TRAIN_PATH):
    if p.path.endswith("pkl"):
        with open(p.path, "rb") as file:
            data = pickle.load(file)
            rawx.append(data[0])
            rawy.append(data[1])

x = np.concatenate(rawx)
y = np.concatenate(rawy)

x_train = x.reshape(int(x.shape[0]/8), 8, 80, 2)

# y_train = y.reshape((int(y.shape[0]/4), 4, 5))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                                                                 activation='relu'),
                                   input_shape=(None,80,2)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.AvgPool1D(pool_size=2)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.summary()
model.fit(x = x_train, y = y, batch_size=10, epochs=18)


#Test cobined B and F
model.evaluate(x_train[:1,:4, :,:], y[:1])
