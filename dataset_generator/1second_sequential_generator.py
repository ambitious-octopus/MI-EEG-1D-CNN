import os
import sys
import numpy as np
from data_processing.general_processor import Utils
import pickle
import mne
from mne.io import read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.epochs import Epochs
import pandas as pd

WINDOW_LEGHT = 80
# Carefoul EPOCH_DIVISON should be int
EPOCH_DIVISON = int(640/WINDOW_LEGHT)

#Select channel combination
channels = Utils.combinations["e"]

#Create dictionary for save info
save_info = {"subject": [],
             "run": [],
             "channels": [],
             "filename": []}


map = {"B": np.array([1,0,0,0,0]),
       "L": np.array([0,1,0,0,0]),
       "R": np.array([0,0,1,0,0]),
       "LR": np.array([0,0,0,1,0]),
       "F": np.array([0,0,0,0,1])}

exclude = [38, 88, 89, 92, 100, 104]
subjects = [str(n) for n in np.arange(1, 110) if n not in exclude]
runs = [str(n) for n in [4, 6, 8, 10, 12, 14]]

data_path = "E:\\datasets\\eegbci"
save_path = "E:\\datasets\\eegnn\\seq"


task2 = [4, 8, 12] #(imagine opening and closing left or right fist)
task4 = [6, 10, 14] #(imagine opening and closing both fists or both feet)
# Get files
for subject in subjects:
    if len(subject) == 1:
        sub_name = "S" + "00" + subject
    elif len(subject) == 2:
        sub_name = "S" + "0" + subject
    else:
        sub_name = "S" + subject
    #Once we build the subject name adddres we join it with full path of the dataset
    sub_folder = os.path.join(data_path, sub_name)
    #At this point we should take the runs
    for run in runs:
        if len(run) == 1:
            path_run = os.path.join(sub_folder, sub_name + "R" + "0" + run + ".edf")
        else:
            path_run = os.path.join(sub_folder, sub_name + "R" + run + ".edf")
        # Load the raw edf of a single subject and a single run
        raw_run = read_raw_edf(path_run, preload=True)
        # Simply change the channels names ".C3" -> "C3"
        eegbci.standardize(raw_run)
        # Load the correct montage
        montage = make_standard_montage('standard_1005')
        # Apply correct montage to raw_edf file
        raw_run.set_montage(montage)
        # Check if the run is in task 2 or task 4
        # aka -> (imagine opening and closing both fists or both feet) or (imagine opening and closing left or right fist)
        if int(run) in task2:
            for index, an in enumerate(raw_run.annotations.description):
                if an == "T0":
                    raw_run.annotations.description[index] = "B"
                if an == "T1":
                    raw_run.annotations.description[index] = "L"
                if an == "T2":
                    raw_run.annotations.description[index] = "R"

        if int(run) in task4:
            for index, an in enumerate(raw_run.annotations.description):
                if an == "T0":
                    raw_run.annotations.description[index] = "B"
                if an == "T1":
                    raw_run.annotations.description[index] = "LR"
                if an == "T2":
                    raw_run.annotations.description[index] = "F"
        # For each couple of channels (e.g. ["FC1", "FC2"])
        for channel in channels:
            raw = raw_run.copy()
            if int(run) in task2:
                event_id = dict(B=1, L=2, R=3)
            elif int(run) in task4:
                event_id = dict(B=1, LR=2, F=3)
            else:
                raise Exception("Screeeeammmmm")

            tmin = 0
            tmax = 4
            events, _ = mne.events_from_annotations(raw, event_id=event_id)

            picks = mne.pick_types(raw.info, meg=False,
                                   eeg=True, stim=False, eog=False,
                                   exclude='bads', selection=channel)

            epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)

            finalx = list()
            finaly = list()
            for index, i in enumerate(epochs):
                counter = WINDOW_LEGHT
                for a in range(EPOCH_DIVISON):
                    if a == 0:
                        finalx.append(i[:, :counter])
                        counter += WINDOW_LEGHT
                    else:
                        finalx.append(i[:, counter - WINDOW_LEGHT :counter])
                finaly.append(epochs[index]._name)
            encoded = list()

            # Transform the categorical label (e.g. "B") to one-hot (e.g. [1,0,0,0,0])
            for i in finaly:
                encoded.append(map[i])
            # todo: save epoch by epoch in order to feed the tf.data.Dataset.from_generator
            

            x = np.stack(finalx)
            y = np.stack(encoded)
            filename = sub_name + "_" + run + "_" + channel[0] + "-" + channel[1] + ".pkl"

            save_info["subject"].append(sub_name)
            save_info["run"].append(run)
            save_info["channels"].append(channel[0] + "-" + channel[1])
            save_info["filename"].append(filename)

            with open(os.path.join(save_path, filename), "wb") as file:
                pickle.dump((x, y), file)

#save important data
tracked_data = pd.DataFrame.from_dict(save_info)
tracked_data.to_csv(os.path.join(save_path, "save_info.csv"))


