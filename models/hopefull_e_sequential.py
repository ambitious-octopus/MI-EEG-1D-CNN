#Importing stuff
import os
import sys
print(os.getcwd())
print(sys.path)
sys.path.append("/home/kubasinska/data/repos/eeGNN")
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

PATH = "/home/kubasinska/datasets/eegbci/paper"
SAVE_TO = "/home/kubasinska/datasets/eegbci/e"
plot = False

channels = Utils.combinations["e"]


exclude =  [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1,109) if n not in exclude]
#Load data
x, y = Utils.load(channels, subjects, base_path=PATH)
#Transform y to one-hot-encoding
y_one_hot  = Utils.to_one_hot(y, by_sub=False)
#Reshape for scaling
reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
#Grab a test set before SMOTE
x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(reshaped_x,
                                                                            y_one_hot,
                                                                            stratify=y_one_hot,
                                                                            test_size=0.20,
                                                                            random_state=4532)

#Scale indipendently train/test
#Axis used to scale along. If 0, independently scale each feature, otherwise (if 1) scale each sample.
x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
x_test_valid_scaled_raw = minmax_scale(x_valid_test_raw, axis=1)

#Create Validation/test
x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_valid_scaled_raw,
                                                    y_valid_test_raw,
                                                    stratify=y_valid_test_raw,
                                                    test_size=0.50,
                                                    random_state=4342)

x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], int(x_valid_raw.shape[1]/2),2).astype(np.float64)
x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1]/2),2).astype(np.float64)

#apply smote to train data
print('classes count')
print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
# smote
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=4542)
x_train_smote_raw, y_train = sm.fit_resample(x_train_scaled_raw, y_train_raw)
print('classes count')
print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
print ('after oversampling = {}'.format(y_train.sum(axis=0)))

x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1]/2), 2).astype(np.float64)


#%%
learning_rate = 1e-4 # default 1e-3

loss = tf.keras.losses.categorical_crossentropy  #tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

inp_shape = (640,2)
kernel_size_0 = 20
kernel_size_1 = 6
drop_rate = 0.5

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=kernel_size_0,
                                            activation='relu',
                                            padding= "same",
                                            input_shape=inp_shape))

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=(6, 2),
                                            activation='relu',
                                            padding= "valid"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
model.add(tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=kernel_size_1,
                                            activation='relu',
                                            padding= "valid"))
model.add(tf.keras.layers.AvgPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=kernel_size_1,
                                            activation='relu',
                                            padding= "valid"))
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

model.fit(x_train, y_train, epochs=1, batch_size=10,
                 validation_data=(x_valid, y_valid)) #32

# Make an auxiliary model that exposes the output from the intermediate layer
# of interest, which is the first Dense layer in this case.
aux_model = tf.keras.Model(inputs=model.inputs,
                           outputs=model.outputs + [model.layers[0].output])

# Access both the final and intermediate output of the original model
# by calling `aux_model.predict()`.
final_output, intermediate_layer_output = aux_model.predict(x_train[:20])

