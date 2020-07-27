from pirate import Pirates
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from PIL import Image

cwd = os.getcwd()
p_path = "C:\\Users\\User\\OneDrive\\Documenti\\reco_raws"
#raw = mne.io.read_raw_fif(p_path + "S001.fif")

#Caricare tutti i reco_raw
#Epocare tutto
#Calcolare la differenza
#Mettere tutto in un file
result = list()
raws = list()
import os
for subdir, dirs, files in os.walk(p_path):
    for file in files:
        filepath = subdir + os.sep + file
        raw = mne.io.read_raw_fif(filepath,preload=True)
        raws.append(raw)

for index,raw in enumerate(raws):
    di = dict()
    di["S"] = index+1
    ch = ["C3", "C4"]
    for channel in ch:
        events, _ = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))
        picks = mne.pick_channels(raw.info["ch_names"], [channel])
        tmin, tmax = 0, 3
        event_ids = dict(base=1, left=2, right=3)
        epochs = mne.Epochs(raw, events, event_ids, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True)
        base_data = epochs["base"].average().data
        right_data = epochs["right"].average().data
        left_data = epochs["left"].average().data

        pxx_base, freqs_base = plt.psd(base_data[0], Fs=160)
        pxx_left, freqs_left = plt.psd(left_data[0], Fs=160)
        pxx_right, freqs_right = plt.psd(right_data[0], Fs=160)

        n_base = []
        n_left = []
        n_right = []

        for p, f in zip(pxx_base, freqs_base):
            if f >= 8 and f <= 13:
                n_base.append(p)
        np.mean(n_base)
        for p, f in zip(pxx_left, freqs_left):
            if f >= 8 and f <= 13:
                n_left.append(p)
        np.mean(n_left)
        for p, f in zip(pxx_right, freqs_right):
            if f >= 8 and f <= 13:
                n_right.append(p)
        np.mean(n_right)

        di[channel+"base-" + channel+"left"] = n_base[0]-n_left[0]
        di[channel + "base-" + channel + "right"] = n_base[0] - n_right[0]
    result.append(di)

import csv
csv_columns = ['S', 'C3base-C3left', 'C3base-C3right','C4base-C4left','C4base-C4right']
csv_file = "deltas.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in result:
            writer.writerow(data)
except IOError:
    print("I/O error")

epochs["base"].plot_psd(dB=False)
epochs["left"].plot_psd(dB=False)

# psds,freqs = mne.time_frequency.psd_array_multitaper(base.data[0],160)
# plt.plot(freqs,psds)
# psds,freqs = mne.time_frequency.psd_array_multitaper(left.data[0],160)
# plt.plot(freqs,psds)
# plt.show()

#%%
from pirate import Pirates
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from PIL import Image

cwd = os.getcwd()
p_path = "C:\\Users\\User\\OneDrive\\Documenti\\reco_raws"

raws= list()
for subdir, dirs, files in os.walk(p_path):
    for file in files:
        filepath = subdir + os.sep + file
        raw = mne.io.read_raw_fif(filepath,preload=True)
        raws.append(raw)

base_list3 = list()
left_list3 = list()
right_list3 = list()
base_list4 = list()
left_list4 = list()
right_list4 = list()
ch = ["C3", "C4"]
for index,raw in enumerate(raws):
    for index,channel in enumerate(ch):
        events, _ = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))
        picks = mne.pick_channels(raw.info["ch_names"], [channel])
        tmin, tmax = 0, 3
        event_ids = dict(base=1, left=2, right=3)
        epochs = mne.Epochs(raw, events, event_ids, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True)
        if index == 0:
            base_list3.append(epochs["base"].average())
            right_list3.append(epochs["right"].average())
            left_list3.append(epochs["left"].average())
        else:
            base_list4.append(epochs["base"].average())
            right_list4.append(epochs["right"].average())
            left_list4.append(epochs["left"].average())

grand = mne.grand_average()




