#Importing stuff
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
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print(physical_devices)
from sklearn.preprocessing import minmax_scale
tf.autograph.set_verbosity(0)
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
import pickle

#TRAIN_PATH = "E:\\split_eegnn\\train"
#TEST_PATH = "E:\\split_eegnn\\test"

TRAIN_PATH = "/home/sbargione/datasets/test15nosmote/train"
TEST_PATH = "/home/sbargione/datasets/test15nosmote/test"


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
model = HopefullNet(inp_shape=(320, 2))
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
    min_delta = 0.001, # set minimum metrics delta
    patience = 15, # number of epochs to stop training
    restore_best_weights = True, # set if use best weights or last weights
    )
callbacksList = [checkpoint, earlystopping] # build callbacks list
#%

hist = model.fit(x = x_train.batch(32), epochs = 200, validation_data = x_test.batch(32), callbacks = callbacksList) #32

import pickle

with open(os.path.join(os.cwd(), "hist.pkl"), 'wb') as file:
    pickle.dump(hist, file)



#%%

#import matplotlib
#matplotlib.use("TkAgg")
#import matplotlib.pyplot as plt
#plt.style.use('seaborn')
#plt.subplot(1,2,1, title="train accuracy")
#plt.plot(hist.history["accuracy"], label="Train")
#plt.plot(hist.history["val_accuracy"], label="Test")
#plt.legend(loc='lower right')
#plt.subplot(1,2,2, title="train loss")
#plt.plot(hist.history["val_loss"], label="Test")
#plt.plot(hist.history["loss"], label="Train")
#plt.legend(loc='upper right')
#plt.show()


#%%
"""
Test model
"""

#model.load_weights(modelPath)

# model = tf.keras.models.load_model("D:\\hopefull", custom_objects={"CustomModel": HopefullNet})



testLoss, testAcc = model.evaluate(x_test)
print('\nAccuracy:', testAcc)
print('\nLoss: ', testLoss)

#from sklearn.metrics import classification_report, confusion_matrix
# get list of MLP's prediction on test set
#yPred = model.predict(x_test)

# convert from one hot encode in class
#yTestClass = np.argmax(y_test, axis=1)
#yPredClass = np.argmax(yPred,axis=1)

#print('\n Classification report \n\n',
 # classification_report(
  #    yTestClass,
   #   yPredClass,
    #   target_names=["B", "R", "RL", "L", "F"]
    #  )
 # )
#print('\n Confusion matrix \n\n',
 # confusion_matrix(
  #    yTestClass,
  #    yPredClass,
  #    )
 # )

#%%
#if plot:
 #   conf = confusion_matrix(yTestClass,yPredClass)
 #   import seaborn as sns
 #   sns.heatmap(conf, annot=True, fmt="", xticklabels=["B", "R", "RL", "L", "F"], yticklabels=["B",
  #                                                                                             "R",
 #                                                                                      "RL", "L", "F"])
#%%

#model.save(SAVE_TO)
