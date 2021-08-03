import os
import sys
print(os.getcwd())
print(sys.path)
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

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

PATH = "E:\\datasets\\eegnn\\n_ch_base"
SOURCE_MODEL = "E:\\rois\\roi_e_t5"


channels = Utils.combinations["e"]

subjects = [101]
#Load data
x, y = Utils.load(channels, subjects, base_path=PATH)
#Transform y to one-hot-encoding
y_one_hot  = Utils.to_one_hot(y, by_sub=False)
#Reshape for scaling
reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

#Grab a test set before SMOTE
x_train_raw, x_test_raw, y_train_raw, y_test = train_test_split(reshaped_x,
                                                                            y_one_hot,
                                                                            stratify=y_one_hot,
                                                                            test_size=0.20,
                                                                            random_state=42)

#Scale indipendently train/test
#Axis used to scale along. If 0, independently scale each feature, otherwise (if 1) scale each sample.
x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
x_test_scaled_raw = minmax_scale(x_test_raw, axis=1)

x_test = x_test_scaled_raw.reshape(x_test_scaled_raw.shape[0], int(x_test_scaled_raw.shape[1]/2),2).astype(np.float64)

#apply smote to train data
print('classes count')
print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
# smote
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x_train_smote_raw, y_train = sm.fit_resample(x_train_scaled_raw, y_train_raw)
print('classes count')
print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
print ('after oversampling = {}'.format(y_train.sum(axis=0)))

x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1]/2), 2).astype(np.float64)


#%%
model = tf.keras.models.load_model(SOURCE_MODEL, custom_objects={"CustomModel": HopefullNet})

before_testLoss, before_testAcc = model.evaluate(x_test, y_test)

# 4 -> 94%

#Freze conv layers
for l in model.layers[:6]:
    l.trainable = False

for l in model.layers:
    print(l._name, l.trainable)


learning_rate = 1e-4 # default 1e-3
loss = tf.keras.losses.categorical_crossentropy  #tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
modelPath = os.path.join(os.getcwd(),'bestModel.h5')

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

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
    min_delta=0.001, # set minimum metrics delta
    patience=4, # number of epochs to stop training
    restore_best_weights=True, # set if use best weights or last weights
    )
callbacksList = [checkpoint, earlystopping] # build callbacks list
#%%

hist = model.fit(x_train, y_train, epochs=30, batch_size=10,
                 validation_data=(x_test, y_test), callbacks=callbacksList) #32


testLoss, testAcc = model.evaluate(x_test, y_test)

print('Before Accuracy:', before_testAcc)
print('Before Loss: ', before_testLoss)

print('After Accuracy:', testAcc)
print('After Loss: ', testLoss)

from sklearn.metrics import classification_report, confusion_matrix
# get list of MLP's prediction on test set
yPred = model.predict(x_test)

# convert from one hot encode in class
yTestClass = np.argmax(y_test, axis=1)
yPredClass = np.argmax(yPred,axis=1)

print('\n Classification report \n\n',
  classification_report(
      yTestClass,
      yPredClass,
       target_names=["B", "R", "RL", "L", "F"],
      digits=4
      )
  )
print('\n Confusion matrix \n\n',
  confusion_matrix(
      yTestClass,
      yPredClass,
      )
  )





