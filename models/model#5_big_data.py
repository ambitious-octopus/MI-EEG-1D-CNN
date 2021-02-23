import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
import os
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.autograph.set_verbosity(0)
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Load big dataset
channels = [["C3", "C4"],
            ["FC3", "FC4"],
            ["C1", "C2"],
            ["C5", "C6"],
            ["FC1", "FC2"],
            ["FC5", "FC6"]]
exclude =  [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1,109) if n not in exclude]
# subjects = [109]
data_x = list()
data_y = list()
base_path = "D:\\datasets\\eeg_dataset\\n_ch_base"

for couple in channels:
    data_path = os.path.join(base_path, couple[0]+ couple[1])
    sub_name = "_sub_"
    xs, ys = Utils.load_sub_by_sub(subjects, data_path, sub_name)
    data_x.append(np.concatenate(xs))
    data_y.append(np.concatenate(ys))

not_scaled_x = np.concatenate(data_x)
y = np.concatenate(data_y)

reshaped_x = not_scaled_x.reshape(not_scaled_x.shape[0], not_scaled_x.shape[1] * not_scaled_x.shape[2])
#Scaling
from sklearn.preprocessing import minmax_scale
#Axis used to scale along. If 0, independently scale each feature, otherwise (if 1) scale each sample.
scaled_x = minmax_scale(reshaped_x, axis=1)
#one hot encoding
y_numerical = Utils.to_numerical(y, by_sub=False)
y_one_hot = tf.keras.utils.to_categorical(y_numerical)
print('classes count')
print ('before oversampling = {}'.format(y_one_hot.sum(axis=0)))
# smote
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
XRes, desYRes = sm.fit_sample(scaled_x, y_one_hot)
print('classes count')
print ('before oversampling = {}'.format(y_one_hot.sum(axis=0)))
print ('after oversampling = {}'.format(desYRes.sum(axis=0)))

x_train, x_test, y_train, y_test = train_test_split(XRes, desYRes, stratify=desYRes,
                                                    test_size=0.20,
                                                    random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, stratify=y_test,
                                                    test_size=0.50,
                                                    random_state=42)

x_train_resh = x_train.reshape(x_train.shape[0], int(x_train.shape[1]/2), 2).astype(np.float64)
x_test_resh = x_test.reshape(x_test.shape[0], int(x_test.shape[1]/2),2).astype(np.float64)
x_valid_resh = x_valid.reshape(x_valid.shape[0], int(x_valid.shape[1]/2),2).astype(np.float64)


learning_rate = 1e-4 # default 1e-3
kernel_size_0 = 20 #5 e 6 good learning_rate = 1e-4 good
kernel_size_1 = 6
drop_rate = 0.5


loss = tf.keras.losses.categorical_crossentropy  #tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
# optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0,
                                 activation='relu', padding= "same", input_shape=(640, 2)))
# model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_1, activation='relu',
                                 padding= "valid"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
# model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_1, activation='relu',
                                 padding= "valid"))
model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
# model.add(tf.keras.layers.MaxPool1D(pool_size=2))


model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0, activation='relu',
                                 padding= "valid"))
# model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.SpatialDropout1D(drop_rate))
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

#%%

hist = model.fit(x_train_resh, y_train, epochs=400, batch_size=15,
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
path = "C:\\Users\\franc_pyl533c\\OneDrive\\Repository\\eeGNN\\models\\bestModel.h5"
model.load_weights(path)

# model.load_weights(os.path.join(os.getcwd(),
#                                 "bestModel.h5"))

testLoss, testAcc = model.evaluate(x_test_resh,y_test)
print('\nAccuracy:', testAcc)
print('\nLoss: ', testLoss)

from sklearn.metrics import classification_report, confusion_matrix
# get list of MLP's prediction on test set
yPred = model.predict(x_test_resh)

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

#%%
#Generate graph

y = np.load("C:\\Users\\franc_pyl533c\\OneDrive\\Repository\\eeGNN\\labels.npy")
x = np.load("C:\\Users\\franc_pyl533c\\OneDrive\\Repository\\eeGNN\\window.npy")
x_scaled = minmax_scale(x.reshape(x.shape[0], x.shape[1]*x.shape[2]), axis=1)
x_final = x_scaled.reshape(x_scaled.shape[0], int(x_scaled.shape[1]/2), 2).astype(np.float64)

y_numerical = Utils.to_numerical(y, by_sub=False)
y_one_hot = tf.keras.utils.to_categorical(y_numerical)

acc = model.predict(x_final)



import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots
time = pd.read_pickle("time.pickle")
time.drop(time.tail(1).index, inplace=True)
fig = make_subplots(rows=5, cols=1)
classes = ["B", "F", "L", "LR", "R"]
for index, cla in enumerate(classes):
    fig.add_trace(go.Scatter(y=np.round(acc[:, index]),  name=cla), row=index+1, col=1)


for index, r in enumerate(time.iterrows()):
    if r[1].task == "nan" or r[1].task == "B":
        fig.add_vrect(
            x0=index, x1=index+1,
            fillcolor="LightSalmon", opacity=0.2,
            layer="below", line_width=1, row=1, col=1
        )
    elif r[1].task == "F":
        fig.add_vrect(
            x0=index, x1=index+1,
            fillcolor="LightSalmon", opacity=0.2,
            layer="below", line_width=1, row=2, col=1
        )
    elif r[1].task == "L":
        fig.add_vrect(
            x0=index, x1=index+1,
            fillcolor="LightSalmon", opacity=0.2,
            layer="below", line_width=1, row=3, col=1
        )
    elif r[1].task == "LR":
        fig.add_vrect(
            x0=index, x1=index+1,
            fillcolor="LightSalmon", opacity=0.2,
            layer="below", line_width=1, row=4, col=1
        )
    elif r[1].task == "R":
        fig.add_vrect(
            x0=index, x1=index+1,
            fillcolor="LightSalmon", opacity=0.2,
            layer="below", line_width=1, row=5, col=1
        )
fig.write_html("sa.html")
