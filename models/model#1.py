import mne
import numpy as np
import utils as u
from mne.datasets import eegbci
from mne.epochs import Epochs
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#%%
"""
If Generate and save
"""
base_path = "D:"
dir_path = os.path.join(base_path, "eeg_data")

#Create data
os.mkdir(dir_path)
s = np.arange(1,86)
x, y = u.get_data(list(s), [4,6,8,10,12,14], 0, 4, ["C3", "C4"])
np.save(os.path.join(dir_path, "x_C3_C4"), x, allow_pickle=True)
np.save(os.path.join(dir_path, "y_C3_C4"), y, allow_pickle=True)
#%%
"""
Load only
"""
base_path = "D:"
dir_path = os.path.join(base_path, "eeg_data")

#Load data
x_data = np.load(os.path.join(dir_path, "x_C3_C4.npy"))
y = np.load(os.path.join(dir_path, "y_C3_C4.npy"))

"""
OneHot encoding
"""
total_labels = np.unique(y)
mapping = {}
for x in range(len(total_labels)):
  mapping[total_labels[x]] = x
for x in range(len(y)):
  y[x] = mapping[y[x]]
y_data = keras.utils.to_categorical(y)

