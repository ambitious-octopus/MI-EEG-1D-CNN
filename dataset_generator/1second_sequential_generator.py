import os
import sys
import numpy as np
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import pickle
import mne
from mne.io import read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.epochs import Epochs
import pandas as pd

WIN_LEN = 80

channels = Utils.combinations["e"]

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

data_path = "/home/kubasinska/datasets/eegbci/origin"
save_path = "/home/kubasinska/datasets/eegbci/seq"


task2 = [4, 8, 12]
task4 = [6, 10, 14]
for subject in subjects:
    if len(subject) == 1:
        sub_name = "S" + "00" + subject
    elif len(subject) == 2:
        sub_name = "S" + "0" + subject
    else:
        sub_name = "S" + subject
    sub_folder = os.path.join(data_path, sub_name)
    for run in runs:
        if len(run) == 1:
            path_run = os.path.join(sub_folder, sub_name + "R" + "0" + run + ".edf")
        else:
            path_run = os.path.join(sub_folder, sub_name + "R" + run + ".edf")
        raw_run = read_raw_edf(path_run, preload=True)  # Le carico
        eegbci.standardize(raw_run)  # Cambio n_epoch nomi dei canali
        montage = make_standard_montage('standard_1005')  # Caricare il montaggio
        raw_run.set_montage(montage)  # Setto il montaggio
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
        for channel in channels:
            # raw = raw_run.pick_channels(channel)
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

            picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                                   exclude='bads', selection=channel)

            epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)

            finalx = list()
            finaly = list()
            for index, i in enumerate(epochs):
                counter = 0
                for a in range(8):
                        finalx.append(i[:, counter:counter + WIN_LEN])
                        counter += WIN_LEN
                finaly.append(epochs[index]._name)
            encoded = list()
            # todo: salvare x e y in due cartelle separate per usare tf.records altrimenti si va in Out of memory

            for i in finaly:
                encoded.append(map[i])

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


