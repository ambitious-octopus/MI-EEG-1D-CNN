import os
import pickle
import numpy.random
import numpy as np

"""
Before running this generate the data with generator_fix.py  
"""

PATH = "/dataset/sub_by_sub_motor_imagery"

exclude =  [38, 88, 89, 92, 100, 104]
subjects = [str(n) for n in np.arange(1,110) if n not in exclude]

all_subs = list()

for subject_number in subjects:
    with open(os.path.join(PATH, subject_number + ".pkl"), "rb") as file:
        all_subs.append(pickle.load(file))
        # each file is composed as follows:
        # [numpy array [n_task, channels, time], list of labels, ch_names]


channels_mapping = {name:index for name, index in
                    zip(all_subs[0][2], range(len(all_subs[0][ 2])))}

couples = [["FC1", "FC2"],
           ["FC3", "FC4"],
           ["FC5", "FC6"],
           ["C5",  "C6"],
           ["C3",  "C4"],
           ["C1",  "C2"],
           ["CP1", "CP2"],
           ["CP3", "CP4"],
           ["CP5", "CP6"]]

couples_mapping =  [[channels_mapping[couple[0]],
                     channels_mapping[couple[1]]] for couple in couples]



one_hot_encoding = {"R": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                    "L": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
                    "LR":np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
                    "F": np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
                    "B": np.array([0.0, 0.0, 0.0, 0.0, 1.0])}

RANDOM = numpy.random.default_rng(42)

x_train_raw, y_train_raw, x_test_raw, y_test_raw = list(), list(), list(), list()
for sub in all_subs:
    for trial in range(len(sub[0])):
        if RANDOM.random() > 0.3:
            for couple in couples_mapping:
                x_train_raw.append(np.array([sub[0][trial, couple[0], :],
                                            sub[0][trial, couple[1], :]]))

                y_train_raw.append(one_hot_encoding[sub[1][trial]])
        else:
            for couple in couples_mapping:
                x_test_raw.append(np.array([sub[0][trial, couple[0], :],
                                            sub[0][trial, couple[1], :]]))

                y_test_raw.append(one_hot_encoding[sub[1][trial]])


x_train, y_train, x_test, y_test = np.stack(x_train_raw)[:, :, :640], np.stack(y_train_raw), np.stack(
    x_test_raw)[:, :, :640], np.stack(y_test_raw)
