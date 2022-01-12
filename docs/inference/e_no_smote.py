"""
A 1D CNN for high accuracy classiﬁcation in motor imagery EEG-based brain-computer interface
Journal of Neural Engineering (https://doi.org/10.1088/1741-2552/ac4430)
Copyright (C) 2022  Francesco Mattioli, Gianluca Baldassarre, Camillo Porcaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import sys
sys.path.append("/workspace")
from model_set.models import HopefullNet
import numpy as np
import tensorflow as tf
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from sklearn.preprocessing import minmax_scale
tf.autograph.set_verbosity(0)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Params
plot = False
source_path = "/dataset/paper/"
save_path = os.path.join("/dataset/saved_models", "roi_e_no_smote")


# Load data
channels = Utils.combinations["e"] #[["FC1", "FC2"],["FC3", "FC4"],["C3", "C4"],["C1", "C2"],["CP1", "CP2"],["CP3", "CP4"]]

exclude =  [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1,110) if n not in exclude]
#Load data
x, y = Utils.load(channels, subjects, base_path=source_path)
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

x_train = x_train_raw.reshape(x_train_raw.shape[0], int(x_train_raw.shape[1]/2), 2).astype(np.float64)

#%%

model = tf.keras.models.load_model(save_path, custom_objects={"CustomModel": HopefullNet})
import pickle
with open(os.path.join(save_path, "hist.pkl"), "rb") as file:
    hist = pickle.load(file)

#%%
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
line_w = 3
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, axs = plt.subplots(1,2, figsize=(15,8))
axs[0].plot(hist["accuracy"], label="train", linewidth=line_w)
axs[0].plot(hist["val_accuracy"], label="validation", linewidth=line_w)
axs[0].legend(loc='lower right')
axs[0].set_title("Accuracy")
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("accuracy")

axs[1].plot(hist["loss"], label="train", linewidth=line_w)
axs[1].plot(hist["val_loss"], label="validation", linewidth=line_w)
axs[1].legend(loc='upper right')
axs[1].set_title("Loss")
axs[1].set_xlabel("epoch")
axs[1].set_ylabel("accuracy")
plt.show()


#%%

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