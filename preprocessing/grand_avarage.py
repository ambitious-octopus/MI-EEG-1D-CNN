from pirate import Pirates
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from PIL import Image

cwd = os.getcwd()
p_path = "C:\\Users\\User\\OneDrive\\Documenti\\reco_raws"
#raw = mne.io.read_raw_fif(real_reco_raws + "S001.fif")

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

#%%
from pirate import Pirates
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap

cwd = os.getcwd()
p_path = "C:\\Users\\User\\OneDrive\\Documenti\\reco_raws"

raws= list()
for subdir, dirs, files in os.walk(p_path):
    for file in files:
        filepath = subdir + os.sep + file
        raw = mne.io.read_raw_fif(filepath,preload=True)
        raws.append(raw)


#raw = mne.concatenate_raws(raws[0])
raw = raws[0]


events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])

# epoch data ##################################################################
tmin, tmax = -1, 3  # define epochs around events (in s)
event_ids = dict(left=2, right=3)  # map event IDs to tasks

epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5,
                    picks=picks, baseline=None, preload=True)

# compute ERDS maps ###########################################################
freqs = np.arange(2, 35, 1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = [-1, 0]  # baseline interval (in s)
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None)  # for cluster test

# Run TF decomposition overall epochs
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                     use_fft=True, return_itc=False, average=False,
                     decim=2)
tfr.crop(tmin, tmax)
tfr.apply_baseline(baseline, mode="percent")
for event in event_ids:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1,
                                     **kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                              axes=ax, colorbar=False, show=False, mask=mask,
                              mask_style="mask")

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if not ax.is_first_col():
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1])
    fig.suptitle("ERDS ({})".format(event))
    fig.show()


#%%
from pirate import Pirates
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap

cwd = os.getcwd()
p_path = "C:\\Users\\User\\OneDrive\\Documenti\\reco_raws"

raws= list()
for subdir, dirs, files in os.walk(p_path):
    for file in files:
        filepath = subdir + os.sep + file
        raw = mne.io.read_raw_fif(filepath,preload=True)
        raws.append(raw)


#raw = mne.concatenate_raws(raws[0])
# raw = raws[0]

for raw in raws:
    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])

    # epoch data ##################################################################
    tmin, tmax = -1, 3  # define epochs around events (in s)
    event_ids = dict(left=2, right=3)  # map event IDs to tasks

    epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5,
                        picks=picks, baseline=None, preload=True)

    # compute ERDS maps ###########################################################
    freqs = np.arange(2, 35, 1)  # frequencies from 2-35Hz
    n_cycles = freqs  # use constant t/f resolution
    vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
    baseline = [-1, 0]  # baseline interval (in s)
    cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                  buffer_size=None)  # for cluster test

    # Run TF decomposition overall epochs
    tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                         use_fft=True, return_itc=False, average=False,
                         decim=2)
    tfr.crop(tmin, tmax)
    tfr.apply_baseline(baseline, mode="percent")
    for event in event_ids:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                                 gridspec_kw={"width_ratios": [10, 10, 10, 1]})
        for ch, ax in enumerate(axes[:-1]):  # for each channel
            # positive clusters
            _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
            # negative clusters
            _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1,
                                         **kwargs)

            # note that we keep clusters with p <= 0.05 from the combined clusters
            # of two independent tests; in this example, we do not correct for
            # these two comparisons
            c = np.stack(c1 + c2, axis=2)  # combined clusters
            p = np.concatenate((p1, p2))  # combined p-values
            mask = c[..., p <= 0.05].any(axis=-1)

            # plot TFR (ERDS map with masking)
            tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                                  axes=ax, colorbar=False, show=False, mask=mask,
                                  mask_style="mask")

            ax.set_title(epochs.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if not ax.is_first_col():
                ax.set_ylabel("")
                ax.set_yticklabels("")
        fig.colorbar(axes[0].images[-1], cax=axes[-1])
        fig.suptitle("ERDS ({})".format(event))
        fig.savefig("C:\\Users\\User\\OneDrive\\Documenti\\erd\\" + str(raw.__repr__()[7:11]) + str(event) + ".png", dpi="figure")
        plt.close("all")




