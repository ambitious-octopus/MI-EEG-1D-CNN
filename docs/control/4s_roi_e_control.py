import sys
import os
print(os.getcwd())
from model_set.models import HopefullNet
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
import pyautogui as pg
import time

PATH = "E:\\datasets\\eegnn\\n_ch_base"
MODEL_PATH = "E:\\rois\\e"


exclude =  [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1,109) if n not in exclude]
#Load data
x, y = Utils.load(Utils.combinations["e"], subjects, base_path=PATH)
#Transform y to one-hot-encoding
y_one_hot  = Utils.to_one_hot(y, by_sub=False)
#Reshape for scaling
reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
#Grab a test set before SMOTE
x_train_raw, x_valid_test_raw, y_train, y_valid_test_raw = train_test_split(reshaped_x,
                                                                            y_one_hot,
                                                                            stratify=y_one_hot,
                                                                            test_size=0.20,
                                                                            random_state=42)

#Scale indipendently train/test
#Axis used to scale along. If 0, independently scale each feature, otherwise (if 1) scale each sample.
x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
x_test_valid_scaled_raw = minmax_scale(x_valid_test_raw, axis=1)

#Create Validation/test
x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_valid_scaled_raw,
                                                    y_valid_test_raw,
                                                    stratify=y_valid_test_raw,
                                                    test_size=0.50,
                                                    random_state=42)

x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], int(x_valid_raw.shape[1]/2),2).astype(np.float64)
x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1]/2),2).astype(np.float64)

x_train = x_train_scaled_raw.reshape(x_train_scaled_raw.shape[0], int(x_train_scaled_raw.shape[1]/2), 2).astype(np.float64)

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"CustomModel": HopefullNet})

target_names={0:"B",1: "R",2: "RL",3: "L",4: "F"}
map_key = {"B": "w",
           "R": "d",
           "L": "a",
           "RL": "w",
           "F": "space"}
time.sleep(10)
for xi, yi in zip(x_train, y_train):
    x = xi.reshape(1, xi.shape[0], xi.shape[1])
    out = target_names[np.argmax(model.predict(x))]
    print(out)
    key = map_key[out]
    print(key)
    pg.keyDown(key)
    time.sleep(0.2)
    pg.keyUp(key)

