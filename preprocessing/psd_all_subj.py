from pirate import Pirates
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


cwd = os.getcwd()
p_path = 'C:\\Users\\admin\\REPOSITORY E DATASETS\\' +'reco_raws'
#raw = mne.io.read_raw_fif(p_path + "S001.fif")

#Caricare tutti i reco_raw
#Epocare tutto
#Calcolare la differenza
#Mettere tutto in un file
result = list()
raws = list()



for subdir, dirs, files in os.walk(p_path):
    for file in files:
        filepath = subdir + os.sep + file
        raw = mne.io.read_raw_fif(filepath,preload=True)
        raws.append(raw)

raws_copy = raws[:1]
list_di = []
di_psd = dict()

for index,raw in enumerate(raws_copy):
    di = dict()
    di["S"] = index+1
    ch = ["C3", "C4"]

    d_ch = dict()
    for channel in ch:
        events, _ = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))
        picks = mne.pick_channels(raw.info["ch_names"], [channel])
        tmin, tmax = 0, 3
        event_ids = dict(base=1, left=2, right=3)
        epochs = mne.Epochs(raw, events, event_ids, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True)
        # base_data = epochs["base"].average().data
        # right_data = epochs["right"].average().data
        # left_data = epochs["left"].average().data

        base_psd_dic = {'pxx_base': [], 'freqs_base': []}

        for e_b in epochs['base']:

            e_b_1D = e_b[0]
            pxx_base, freqs_base = plt.psd(e_b_1D, Fs=160)
            base_psd_dic['pxx_base'].append(pxx_base)
            base_psd_dic['freqs_base'].append(freqs_base)

        right_psd_dic = {'pxx_right': [], 'freqs_right': []}

        for e_r in epochs['right']:
            e_r_1D = e_r[0]
            pxx_right, freqs_right = plt.psd(e_r_1D, Fs=160)
            right_psd_dic['pxx_right'].append(pxx_right)
            right_psd_dic['freqs_right'].append(freqs_right)

        left_psd_dic = {'pxx_left': [], 'freqs_left': []}

        for e_l in epochs['left']:

            e_l_1D = e_l[0]
            pxx_left, freqs_left = plt.psd(e_l_1D, Fs=160)
            left_psd_dic['pxx_left'].append(pxx_left)
            left_psd_dic['freqs_left'].append(freqs_left)

    # average dei psd

        base_average_pxx = np.mean(base_psd_dic['pxx_base'], axis=0)
        base_average_freqs = np.mean(base_psd_dic['freqs_base'], axis=0)
        base_av_dic = {'pxx_base_av': base_average_pxx,'freqs_base_av':base_average_freqs}

        right_average_pxx = np.mean(right_psd_dic['pxx_right'], axis=0)
        right_average_freqs = np.mean(right_psd_dic['freqs_right'], axis=0)
        right_av_dic = {'pxx_right_av':right_average_pxx,'freqs_right_av': right_average_freqs}

        left_average_pxx= np.mean(left_psd_dic['pxx_left'], axis=0)
        left_average_freqs= np.mean(left_psd_dic['freqs_left'], axis=0)
        left_av_dic = {'pxx_left_av': left_average_pxx, 'freqs_left_av': left_average_freqs}



        n_base_pxx = []
        n_base_freqs= []

        n_left_pxx = []
        n_left_freqs = []

        n_right_pxx = []
        n_right_freqs = []

        for p, f in zip(base_average_pxx, base_average_freqs):
            if f >= 8 and f <= 13:
                n_base_pxx.append(p)
                n_base_freqs.append(f)
        for p2, f2 in zip(left_average_pxx, left_average_freqs):
            if f2 >= 8 and f2 <= 13:
                n_left_pxx.append(p2)
                n_left_freqs.append(f2)
        for p3, f3 in zip(right_average_pxx, right_average_freqs):
            if f3 >= 8 and f3 <= 13:
                n_right_pxx.append(p3)
                n_right_freqs.append(f3)

#Tobecontinued


        d_ch[channel] = {'n_base_pxx': list(n_base_pxx) ,'n_base_freqs': list(n_base_freqs),'n_left_pxx': list(n_left_pxx),'n_left_freqs':list(n_left_freqs),'n_right_pxx': list(n_right_pxx),'n_right_freqs': list(n_right_freqs)}
        #di_inlist = {'Soggetto': raw.__repr__()[7:11], 'Canale': channel,'n_base_pxx': list(n_base_pxx) ,'n_base_freqs': list(n_base_freqs),'n_left_pxx': list(n_left_pxx),'n_left_freqs':list(n_left_freqs),'n_right_pxx': list(n_right_pxx),'n_right_freqs': list(n_right_freqs)}
        #list_di.append(di_inlist)

    di_psd[raw.__repr__()[7:11]] = d_ch

    #     di[channel+"base-" + channel+"left"] = n_base_pxx[0]-n_left_pxx[0]
    #     di[channel + "base-" + channel + "right"] = n_base_pxx[0]- n_right_pxx[0]
    # result.append(di)

df = pd.DataFrame(list_di)
df.to_csv('Df_psd_all_subj.csv')

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




