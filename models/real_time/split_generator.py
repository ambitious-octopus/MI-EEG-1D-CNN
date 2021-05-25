import os
import sys
print(os.getcwd())
print(sys.path)
import numpy as np
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

import pickle

channels = [["FC1", "FC2"],
            ["FC3", "FC4"],
            ["FC5", "FC6"],
            ["C5", "C6"],
            ["C3", "C4"],
            ["C1", "C2"],
            ["CP1", "CP2"],
            ["CP3", "CP4"],
            ["CP5", "CP6"]]

exclude = [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1, 110) if n not in exclude]
# subjects = [1]
runs = [4, 6, 8, 10, 12, 14]

data_path = "E:\\datasets\\eegbci"

final_x = list()
final_y = list()


for couple in channels:
    for sub in subjects:
        x, y = Utils.epoch(
            Utils.select_channels(
                Utils.eeg_settings(
                    Utils.del_annotations(
                        Utils.concatenate_runs(
                            Utils.load_data(subjects=[sub],
                                            runs=runs,
                                            data_path=data_path)))),
                couple),
            exclude_base=False)
        new_x = np.delete(x, -1, axis=2)
        del x
        new_y = np.array(y)
        del y
        for xi, yi in zip(new_x, new_y):
            final_x.append(xi[:, :320])
            final_x.append(xi[:, 320:])
            for a in range(2):
                final_y.append(yi)

x = np.stack(final_x)
y = Utils.to_one_hot(np.stack(final_y))

reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(reshaped_x,
                                                                            y,
                                                                            test_size=0.20,
                                                                            random_state=42)

x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
x_test_valid_scaled_raw = minmax_scale(x_test_raw, axis=1)

x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1]/2),2).astype(np.float64)


#apply smote to train data
print('classes count')
print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x_train_smote_raw, y_train = sm.fit_resample(x_train_scaled_raw, y_train_raw)
print('classes count')
print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
print ('after oversampling = {}'.format(y_train.sum(axis=0)))

x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1]/2), 2).astype(np.float64)

save_path = "D:\\split_eegnn"
test_path = os.path.join(save_path, "test")
train_path = os.path.join(save_path, "train")
os.mkdir(test_path)
os.mkdir(train_path)

counter = 0
for xi, yi in zip(x_train, y_train):
    counter += 1
    with open(os.path.join(train_path, str(counter) + ".pkl"), "wb") as file:
        pickle.dump([xi, yi], file)

counter = 0
for xi, yi in zip(x_test, y_test_raw):
    counter += 1
    with open(os.path.join(test_path, str(counter) + ".pkl"), "wb") as file:
        pickle.dump([xi, yi], file)







