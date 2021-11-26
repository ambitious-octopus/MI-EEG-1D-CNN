import os
import mne
import matplotlib
import numpy as np
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

"""
Don't know what this script does?
    Go Here: -> readme.txt
"""
from mne.externals.pymatreader import read_mat

def load_finger_dataset(paths):
    xs = list()
    ys = list()
    for p in paths:
        mat_data = read_mat(p)
        slice = list()
        # slice the ROI
        channels = [9, 10, 12, 11, 17, 18, 45, 44, 48, 49, 55, 54]
        for ch in channels:
            slice.append(mat_data["eeg"]["imagery_left"][ch])
        data_slice = np.stack(slice)  # SHAPE = CHANNELS * TIME
        # Divide de 100 trials


        data = data_slice.T.reshape(int(data_slice.shape[1] / (512 * 7)), 512 * 7, 12)
        x = data.reshape(mat_data["eeg"]["n_imagery_trials"] * 7, 512, 12)
        trial_sequence = ["B", "B", "L", "L", "L", "B", "B"]
        cat_y = np.array(trial_sequence * mat_data["eeg"]["n_imagery_trials"])
        map = {"B": np.array([0.0, 1.0]),
               "L": np.array([1.0, 0.0])}
        y = np.array([map[i] for i in cat_y])
        xs.append(x)
        ys.append(y)
    return xs, ys


base_path = "/home/kubasinska/dataset/finger_dataset"
paths = list()
for i in range(1,49):
    if len(str(i)) == 1:
        paths.append(os.path.join(base_path, "s0" + str(i) + ".mat"))
    else:
        paths.append(os.path.join(base_path, "s" + str(i) + ".mat"))

xs, ys = load_finger_dataset(paths)

x = np.concatenate(xs)
y = np.concatenate(ys)




import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5, input_shape=(512, 12),
                                 data_format="channels_last"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv1D(16, 2, data_format="channels_last"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv1D(16, 2, data_format="channels_last"))
model.add(tf.keras.layers.AvgPool1D())
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, data_format="channels_last"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(150))
model.add(tf.keras.layers.Dense(2))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.categorical_crossentropy, metrics="accuracy")

model.summary()
model.fit(x, y, epochs=20, batch_size=1)
