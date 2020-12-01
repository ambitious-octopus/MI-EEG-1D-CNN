import numpy as np
from data_processing import utils as u
import os
import tensorflow as tf
from tensorflow import keras

#%%
"""
Load only
"""
base_path = "D:"
dir_path = os.path.join(base_path, "eeg_data")

#Load data
x_data = np.load(os.path.join(dir_path, "x_C3_C4.npy"))
y = np.load(os.path.join(dir_path, "y_C3_C4.npy"))
#%%



