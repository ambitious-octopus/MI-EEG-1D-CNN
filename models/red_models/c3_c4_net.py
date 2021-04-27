from model_set.models import GreeAnxietyNet
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
from sklearn.preprocessing import minmax_scale
tf.autograph.set_verbosity(0)
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

channels = [["C3", "C4"]]

exclude =  [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1,109) if n not in exclude]
#Load data
x, y = Utils.load(channels, subjects)
#Transform y to one-hot-encoding
y_one_hot  = Utils.to_one_hot(y, by_sub=False)
#Reshape for scaling
reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
#Grab a test set before SMOTE
x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(reshaped_x,
                                                                            y_one_hot,
                                                                            stratify=y_one_hot,
                                                                            test_size=0.20,
                                                                            random_state=426)

#Scale indipendently train/test
#Axis used to scale along. If 0, independently scale each feature, otherwise (if 1) scale each sample.
x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
x_test_valid_scaled_raw = minmax_scale(x_valid_test_raw, axis=1)

#Create Validation/test
x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_valid_scaled_raw,
                                                    y_valid_test_raw,
                                                    stratify=y_valid_test_raw,
                                                    test_size=0.50,
                                                    random_state=42764)

x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], int(x_valid_raw.shape[1]/2),2).astype(np.float64)
x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1]/2),2).astype(np.float64)

#apply smote to train data
print('classes count')
print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
# smote
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=4674)
x_train_smote_raw, y_train = sm.fit_resample(x_train_scaled_raw, y_train_raw)
print('classes count')
print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
print ('after oversampling = {}'.format(y_train.sum(axis=0)))

x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1]/2), 2).astype(np.float64)


#%%
learning_rate = 1e-5 # default 1e-3

loss = tf.keras.losses.categorical_crossentropy  #tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

kernel_size_0 = 6
kernel_size_1 = 4
drop_rate = 0.5

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=kernel_size_0,
                                 activation='relu', padding= "same", input_shape=(640, 2)))
# model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=kernel_size_1, activation='relu',
                                 padding= "valid"))
model.add(tf.keras.layers.BatchNormalization())

# model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=kernel_size_1, activation='relu',
                                 padding= "valid"))
model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
# model.add(tf.keras.layers.MaxPool1D(pool_size=2))

#
# model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0, activation='relu',
#                                  padding= "valid"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(drop_rate))
# model.add(tf.keras.layers.Conv1D(filters=15, kernel_size=kernel_size, activation='relu',
#                                  strides=1, padding="valid"))
# model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.AvgPool1D(pool_size=2))
# model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
# model.add(tf.keras.layers.Dense(300))
# model.add(tf.keras.layers.Conv1D(filters=100, kernel_size=kernel_size, activation='relu', strides=1, padding= "valid"))
# model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate))
model.add(tf.keras.layers.Dense(5, activation='softmax'))


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

hist = model.fit(x_train, y_train, epochs=40, batch_size=1,
                 validation_data=(x_valid, y_valid), callbacks=callbacksList) #32

#%%
plt.style.use('seaborn')
plt.subplot(1,2,1, title="train accuracy")
plt.plot(hist.history["accuracy"], label="Train")
plt.plot(hist.history["val_accuracy"], label="Test")
plt.legend(loc='lower right')
plt.subplot(1,2,2, title="train loss")
plt.plot(hist.history["val_loss"], label="Test")
plt.plot(hist.history["loss"], label="Train")
plt.legend(loc='upper right')
plt.show()

#%%
"""
Test model
"""
# path = "C:\\Users\\franc_pyl533c\\OneDrive\\Repository\\eeGNN\\models\\bestModel.h5"
# model.load_weights(path)

# model.load_weights(os.path.join(os.getcwd(),
#                                 "bestModel.h5"))

# model = tf.keras.models.load_model("D:\\hopefull_c3_c4", custom_objects={"CustomModel": HopefullNet})
testLoss, testAcc = model.evaluate(x_test, y_test)
print('\nAccuracy:', testAcc)
print('\nLoss: ', testLoss)

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
       target_names=["B", "R", "RL", "L", "F"]
      )
  )
print('\n Confusion matrix \n\n',
  confusion_matrix(
      yTestClass,
      yPredClass,
      )
  )

# model.save("D:\\hopefull_c3_c4")