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

TRAIN_PATH = "E:\\datasets\\eegnn\\2_seconds\\nosmote\\train"
TEST_PATH = "E:\\datasets\\eegnn\\2_seconds\\nosmote\\test"


def gen_train():
    for p in os.scandir(TRAIN_PATH):
        with open(p.path, "rb") as file:
            data = pickle.load(file)
        yield data[0], data[1]

def gen_test():
    for p in os.scandir(TEST_PATH):
        with open(p.path, "rb") as file:
            data = pickle.load(file)
        yield data[0], data[1]


x_train = tf.data.Dataset.from_generator(gen_train,
                                       output_types=(np.float64, np.float64),
                                       output_shapes=((320,2),(5,)))

x_test = tf.data.Dataset.from_generator(gen_test,
                                       output_types=(np.float64, np.float64),
                                       output_shapes=((320,2),(5,)))


#%%
learning_rate = 1e-4 # default 1e-3

loss = tf.keras.losses.categorical_crossentropy  #tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

kernel_size_0 = 4
kernel_size_1 = 2
drop_rate = 0.5

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32,
                                kernel_size=kernel_size_0,
                                activation='relu',
                                padding= "same",
                                input_shape=(320, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=kernel_size_0,
                                            activation='relu',
                                            padding= "same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
model.add(tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=kernel_size_1,
                                            activation='relu',
                                            padding= "same"))
model.add(tf.keras.layers.AvgPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=kernel_size_1,
                                            activation='relu',
                                            padding= "same"))
model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(296, activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate))
model.add(tf.keras.layers.Dense(148, activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate))
model.add(tf.keras.layers.Dense(74, activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate))
model.add(tf.keras.layers.Dense(5, activation='softmax'))


model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

model.summary()

modelPath = os.path.join(os.getcwd(),'bestModel.h5')

checkpoint = ModelCheckpoint( # set model saving checkpoints
    modelPath, # set path to save model weights
    monitor='val_loss', # set monitor metrics
    verbose=1, # set training verbosity
    save_best_only=True, # set if want to save only best weights
    save_weights_only=False, # set if you want to save only model weights
    mode='auto', # set if save min or max in metrics
    period=1 # interval between checkpoints
    )

earlystopping = EarlyStopping(
    monitor='val_loss', # set monitor metrics
    min_delta = 0.001, # set minimum metrics delta
    patience = 15, # number of epochs to stop training
    restore_best_weights = True, # set if use best weights or last weights
    )
callbacksList = [checkpoint, earlystopping] # build callbacks list
#%

hist = model.fit(x = x_train.batch(32), epochs = 200, validation_data = x_test.batch(32), callbacks = callbacksList) #32
