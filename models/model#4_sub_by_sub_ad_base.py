import numpy as np
import os
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler # Usare MIn MAx scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.autograph.set_verbosity(0)
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
#############################################################################
exclude =  [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1,109) if n not in exclude]
# data_path = "D:\\datasets\\eeg_dataset\\C3_C4_sub"
data_path = "D:\datasets\eeg_dataset\C3_C4_sub"
sub_name = "_C3_C4_sub_"

xs, ys = Utils.load_sub_by_sub(subjects, data_path,sub_name)
xs, ys = Utils.scale_sub_by_sub(xs, ys)


# Questo fa una modifica alle label aggiungendo il nome del sogetto in modo da stratificare
new_y = list()
for x, y, index_sub in zip(xs, ys, range(len(xs))):
    subj_array = list()
    for index_label, label in enumerate(y):
        subj_array.append(label + str(index_sub))
    new_y.append(np.array(subj_array))
y = np.concatenate(new_y)
x = np.concatenate(xs)

x_resh = x.reshape(x.shape[0], x.shape[2]*x.shape[1])
y_cat = Utils.to_numerical(y, by_sub=True)
y_real = tf.keras.utils.to_categorical(y_cat)

print('classes count')
print ('before oversampling = {}'.format(y_real.sum(axis=0)))

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
XRes, desYRes = sm.fit_sample(x_resh, y_real)
print('classes count')
print ('before oversampling = {}'.format(y_real.sum(axis=0)))
print ('after oversampling = {}'.format(desYRes.sum(axis=0)))
x_train, x_test, y_train, y_test = train_test_split(XRes, desYRes, stratify=desYRes,
                                                    test_size=0.20,
                                                    random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, stratify=y_test,
                                                    test_size=0.50,
                                                    random_state=42)

#Processing y
# y_train = Utils.to_numerical(y_train, by_sub=True)
# y_test = Utils.to_numerical(y_test, by_sub=True)

#Reshape x -> (sample, width, 1)
x_train_resh = x_train.reshape(x_train.shape[0], int(x_train.shape[1]/2), 2).astype(np.float64)
x_test_resh = x_test.reshape(x_test.shape[0], int(x_test.shape[1]/2),2).astype(np.float64)
x_valid_resh = x_valid.reshape(x_valid.shape[0], int(x_valid.shape[1]/2),2).astype(np.float64)

# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)

#%%
#Convolution Neural Network
# [samples, time steps, features].
# real_x_train = x_train.reshape(14808, 640, 2)
# real_x_test = x_test.reshape(3703, 640, 2)
learning_rate = 1e-4 # default 1e-3
kernel_size_0 = 6 #5 e 6 good learning_rate = 1e-4 good
kernel_size_1 = 4
drop_rate = 0.4 #0.2 good #0.4 local minimum


loss = tf.keras.losses.categorical_crossentropy  #tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
# optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0,
                                 activation='relu', padding= "same", input_shape=(640, 2)))
model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=kernel_size_1, activation='relu',
                                 padding= "valid"))
# model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=kernel_size_1, activation='relu',
                                 padding= "valid"))
model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
model.add(tf.keras.layers.MaxPool1D(pool_size=2))


# model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=kernel_size_1, activation='relu',
                                 # padding= "valid"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(drop_rate))
# model.add(tf.keras.layers.Conv1D(filters=15, kernel_size=kernel_size, activation='relu',
#                                  strides=1, padding="valid"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.AvgPool1D(pool_size=2))
# model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
# model.add(tf.keras.layers.Dense(300))
# model.add(tf.keras.layers.Conv1D(filters=100, kernel_size=kernel_size, activation='relu', strides=1, padding= "valid"))
# model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate))
model.add(tf.keras.layers.Dense(64, activation='relu'))
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
    min_delta=0.001, # set minimum metrics delta
    patience=10, # number of epochs to stop training
    restore_best_weights=True, # set if use best weights or last weights
    )
callbacksList = [checkpoint, earlystopping] # build callbacks list

hist = model.fit(x_train_resh, y_train, epochs=60, batch_size=10,
                 validation_data=(x_valid_resh, y_valid), callbacks=callbacksList) #32


#%%
#Più il batch size è basso e più aumento la variabilità individuale
plt.subplot(1,2,1, title="accuracy")
plt.plot(hist.history["accuracy"], color="red", label="Train")
plt.plot(hist.history["val_accuracy"], color="blue", label="Test")
plt.legend(loc='lower right')
plt.subplot(1,2,2, title="loss")
plt.plot(hist.history["val_loss"], color="blue", label="Test")
plt.plot(hist.history["loss"], color="red", label="Train")
plt.legend(loc='lower right')
plt.show()

print(np.round(model.predict(x_test_resh[:4], 2)))
print(y_test[:4])

#%%
"""
Test model
"""
model.load_weights(os.path.join(os.getcwd(),
                                "bestModel.h5"))
testLoss, testAcc = model.evaluate(x_test_resh,y_test)
print('\nAccuracy:', testAcc)
print('\nLoss: ', testLoss)

from sklearn.metrics import classification_report, confusion_matrix
# get list of MLP's prediction on test set
yPred = model.predict(x_test_resh)

# convert from one hot encode in class
yTestClass = np.argmax(y_test, axis=1)
yPredClass = np.argmax(yPred,axis=1)

# print('\n Classification report \n\n',
#   classification_report(
#       yTestClass,
#       yPredClass,
#        target_names=["B", "A1", "A2", "A3", "A4"]
#       )
#   )
# print('\n Confusion matrix \n\n',
#   confusion_matrix(
#       yTestClass,
#       yPredClass,
#       )
#   )
