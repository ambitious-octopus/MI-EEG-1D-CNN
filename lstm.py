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

TRAIN_PATH = "E:\\datasets\\eegnn\\seq"

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

# Modify y
y_mod = list()
for yi in y:
    for i in range(8):
        y_mod.append(yi)
y = np.stack(y_mod)

"""
Define label assignement
"""
map = {0: np.array([1,0,0,0,0]),
       1: np.array([0,1,0,0,0]),
       2: np.array([0,0,1,0,0]),
       3: np.array([0,0,0,1,0]),
       4: np.array([0,0,0,0,1])}

y_final = list()

for i in range(len(y)-8): #8632
    sample = y[i:i+8,:]
    labels = [np.argmax(s) for s in sample]
    count = np.unique(labels, return_counts=True)

    #se sono tutte uguali aggiungi quella unica
    if len(count[0]) == 1:
        y_final.append(map[count[0][0]])

    # se sono due dello stesso numero aggiungi l'ultima
    elif count[1][0] == count[1][1]:
        y_final.append(map[count[0][1]])

    #se una è più di un'altra aggiungi su quella in più
    elif count[1][0] < count[1][1]:
        y_final.append(map[count[0][1]])

    elif count[1][0] > count[1][1]:
        y_final.append(map[count[0][0]])

    else:
        raise Exception("Unable to assign label")


y_train = np.stack(y_final)

x_final = list()
for i in range(len(x)-8): #8632
    sample = x[i:i+8,:,:]
    x_final.append(sample)

x_train = np.moveaxis(np.stack(x_final), 2, 3)

x_test = x_train[-1000:, :, : , :]
y_test = y_train[-1000:, :]
x_train = x_train[:-1000, : , :, :]
y_train = y_train[:-1000, :]

# x_train = x.reshape(int(x.shape[0]/8), 8, 80, 2)
# y_train = y.reshape((int(y.shape[0]/4), 4, 5))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3,
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
model.add(tf.keras.layers.LSTM(64))
# model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.summary()
model.fit(x = x_train, y = y_train, batch_size=5, epochs=30, validation_data=(x_test, y_test))

# Test cobined B and F
for i in range(1,10):
    pred = model.predict(x_train[0+i:1+i])
    print("Pred: ", np.argmax(pred))
    print("Ground: ", np.argmax(y_train[0+i:1+i]))
    print()

# x_mod = x_train.reshape(1080*8, 80, 2)
# temp_y_mod = list()
# for sample in y:
#     for i in range(8):
#         temp_y_mod.append(sample)
# y_mod = np.stack(temp_y_mod)
#
# # running = True
# # mod = 0
# # while running:
# #     sample = x_mod[0+mod:8+mod].reshape(1, x_mod[0+mod:8+mod].shape[0], x_mod[0+mod:8+mod].shape[1],
# #                                     x_mod[0+mod:8+mod].shape[2])
# #     pred = model.predict(sample)
# #     print("prediction: ", np.argmax(pred))
# #     print("ground: ", [np.argmax(g) for g in y_mod[0+mod:8+mod]])
# #     print()
# #     mod += 1