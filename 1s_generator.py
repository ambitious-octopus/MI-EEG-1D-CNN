import os
import sys
#print(os.getcwd())
#print(sys.path)
import numpy as np

from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import pickle

rand = np.random.default_rng()

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
#subjects = [1]
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
            if yi in ["R", "L", "F", "LR"]:
                final_x.append(xi[:, :160])
                final_x.append(xi[:, 160:320])
                final_x.append(xi[:, 320:480])
                final_x.append(xi[:, 480:640])
                for a in range(4):
                    final_y.append(yi)
            elif yi == "B" and rand.random() >= 0.65:
                final_x.append(xi[:, :160])
                final_x.append(xi[:, 160:320])
                final_x.append(xi[:, 320:480])
                final_x.append(xi[:, 480:640])
                for a in range(4):
                    final_y.append(yi)
            else:
                pass

x = np.stack(final_x)
print(np.unique(np.stack(final_y), return_counts=True))

y = Utils.to_one_hot(np.stack(final_y))

print(type(x))
print(type(y))

#%%

# x reshaped in 2 dimensions!

reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

#added the parameter "stratify = y" to train_test_split

x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(reshaped_x,y,test_size=0.15, stratify = y,random_state=42)

#minmaxscale for x_train & x_test

x_train_scaled_raw = minmax_scale(x_train_raw, axis=0)
x_test_valid_scaled_raw = minmax_scale(x_test_raw, axis=0)

#Reshape of x_test in 3 dimensions (from 2)

x_test = x_test_valid_scaled_raw.reshape(x_test_valid_scaled_raw.shape[0],
                                         int(x_test_valid_scaled_raw.shape[1]/2),2).astype(np.float64)


#apply smote to train data

# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=42)
# x_train_smote_raw, y_train = sm.fit_resample(x_train_scaled_raw, y_train_raw)

##Reshape of x_test in 3 dimensions (from 2) following mixmax_scaler and SMOTE application

x_train = x_train_scaled_raw.reshape(x_train_scaled_raw.shape[0],
                                               int(x_train_scaled_raw.shape[1]/2), 2).astype(np.float64)


save_path = "E:\\datasets\\eegnn\\1_second"

folder1 = os.path.join(save_path, "nosmote")
os.mkdir(folder1)

test_path = os.path.join(folder1, "test")
train_path = os.path.join(folder1, "train")

os.mkdir(test_path)
os.mkdir(train_path)

counter = 0
#here it was y_train, but changed in y_train_raw
for xi, yi in zip(x_train, y_train_raw):
    counter += 1
    with open(os.path.join(train_path, str(counter) + ".pkl"), "wb") as file:
        pickle.dump([xi, yi], file)

counter = 0
for xi, yi in zip(x_test, y_test_raw):
    counter += 1
    with open(os.path.join(test_path, str(counter) + ".pkl"), "wb") as file:
        pickle.dump([xi, yi], file)










