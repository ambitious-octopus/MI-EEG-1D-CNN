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

rand = np.random.default_rng()

channels = Utils.combinations["e"]

exclude = [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1, 110) if n not in exclude]
runs = [4, 6, 8, 10, 12, 14]

subject = 10
run = 4

data_path = "E:\\datasets\\eegbci"

final_x = list()
final_y = list()


sub_name = "S"+"0"+ str(subject)
sub_folder = os.path.join(data_path, sub_name)
path_run = os.path.join(sub_folder, sub_name+"R"+"0"+ str(run) + ".edf")
raw_run = read_raw_edf(path_run, preload=True)  # Le carico

# indexes = []
# for index, value in enumerate(raw_run.annotations.description):
#     if value == "BAD boundary" or value == "EDGE boundary":
#         indexes.append(index)
# raw_run.annotations.delete(indexes)

eegbci.standardize(raw_run)  # Cambio n_epoch nomi dei canali
montage = make_standard_montage('standard_1005')  # Caricare il montaggio
raw_run.set_montage(montage)  # Setto il montaggio

for index, an in enumerate(raw_run.annotations.description):
    if an == "T0":
        raw_run.annotations.description[index] = "B"
    if an == "T1":
        raw_run.annotations.description[index] = "L"
    if an == "T2":
        raw_run.annotations.description[index] = "R"

raw = raw_run.pick_channels(channels[0])

event_id = dict(B=1, L=2, R=3)

tmin=0
tmax=4

events, _ = mne.events_from_annotations(raw, event_id=event_id)

picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                                   exclude='bads')

epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)

import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")

data = np.hstack((epochs[0].get_data().reshape(2, 641), epochs[1].get_data().reshape(2, 641)))

plt.plot(data[0])
plt.axvline(x=641, color="blue", alpha=20)
plt.axvline(x=680, color="blue", alpha=20)
plt.show()