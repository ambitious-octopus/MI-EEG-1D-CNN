from pirate import Pirates
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import csv

cwd = os.getcwd()
real_reco_raws_path = "C:\\Users\\franc_pyl533c\\OneDrive\\Documenti\\real_reco_raws"
imagined_reco_raws_path = "C:\\Users\\franc_pyl533c\\OneDrive\\Documenti\\imagined_reco_raws"


deltas_real = "C:\\Users\\franc_pyl533c\\OneDrive\\Documenti\\deltas_real"
deltas_imagined = "C:\\Users\\franc_pyl533c\\OneDrive\\Documenti\\deltas_imagined"

#Caricare tutti i reco_raw
#Epocare tutto
#Calcolare la differenza
#Mettere tutto in un file
result = list()
raws_real = list()
raws_imagined = list()

#Load real data
for subdir, dirs, files in os.walk(real_reco_raws_path):
    for file in files:
        filepath = subdir + os.sep + file
        raw = mne.io.read_raw_fif(filepath, preload=True)
        raws_real.append(raw)

#Load imagined data
for subdir, dirs, files in os.walk(imagined_reco_raws_path):
    for file in files:
        filepath = subdir + os.sep + file
        raw = mne.io.read_raw_fif(filepath,preload=True)
        raws_imagined.append(raw)

#%%
#Generate data and graph for real
ch = ["C3", "C4"]
dic_3 = []
dic_4 = []
for channel in ch:
    for raw in raws_real:
        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
        picks = mne.pick_channels(raw.info["ch_names"], [channel])
        tmin, tmax = -3, 3
        event_ids = dict(left=2, right=3)
        epochs = mne.Epochs(raw, events, event_ids, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True)
        for i in range(len(epochs)):

            base = epochs[i].plot_psd(dB=False, tmin=-3, tmax=0)
            task = epochs[i].plot_psd(dB=False, tmin=0, tmax=3)
            plt.close("all")

            freq_base = base.gca().lines[2].get_xdata()
            amp_base = base.gca().lines[2].get_ydata()
            freq_task = task.gca().lines[2].get_xdata()
            amp_task = task.gca().lines[2].get_ydata()

            # Trovo il picco alhpa nella baseline
            base_freq_range = list()
            base_amp_range = list()
            result_base_freq = list()
            result_base_amp = list()
            result_task_amp = list()

            for f, a in zip(freq_base, amp_base):
                if f >= 8 and f <= 13:
                    base_freq_range.append(f)
                    base_amp_range.append(a)

            for f, a in zip(base_freq_range, base_amp_range):
                if a == max(base_amp_range):
                    freq_pick = f
                    amp_pick = a

            for f, a in zip(base_freq_range, base_amp_range):
                if f >= freq_pick - 1.5 and f <= freq_pick + 1.5:
                    result_base_freq.append(f)
                    result_base_amp.append(a)

            for f, a in zip(freq_task, amp_task):
                if f >= min(result_base_freq) and f <= max(result_base_freq):
                    result_task_amp.append(a)

            delta = np.mean(result_base_amp) - np.mean(result_task_amp)

            plt.plot(freq_base, amp_base, label="base")
            plt.plot(freq_task, amp_task, label="task")
            plt.title(raw.__repr__()[7:11] + " Channel: " + channel + " epoch: " + str(epochs[i]._name) + " \n Delta: " + str(delta))
            plt.legend()
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Aplitude(µV/sqrt (Hz)")
            plt.grid(True)
            plt.axvspan(freq_pick - 1.5, freq_pick + 1.5, color='red', alpha=0.2)
            #plt.annotate('pick', xy=(freq_pick, amp_pick), xytext=(20, max(amp_base - 50)), arrowprops=dict(arrowstyle="->"), )
            plt.savefig(deltas_real + raw.__repr__()[7:11] + "ch_" + channel + "epoch_" + str(i) + "_" + epochs[i]._name + ".png")
            plt.close("all")
            dic = {"S":raw.__repr__()[7:11], "n_epoch": i, "t_epoch": epochs[i]._name, "delta" : delta}
            if channel == "C3":
                dic_3.append(dic)
            else:
                dic_4.append(dic)


csv_columns = ["S","n_epoch","t_epoch","delta"]
csv_file = "real_C3_deltas.csv"

try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dic_3:
            writer.writerow(data)
except IOError:
    print("I/O error")

csv_file = "real_C4_deltas.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dic_4:
            writer.writerow(data)
except IOError:
    print("I/O error")

#%%
import pandas as pd
import numpy as np
import csv

c4 = pd.read_csv("real_C4_deltas.csv")
c3 = pd.read_csv("real_C3_deltas.csv")

c4_delta = list()
c3_delta = list()
for a in np.unique(c4.S):
    subj = c4[c4.S == a]
    subj_right = np.mean(subj[c4.t_epoch == "right"].delta)
    subj_left = np.mean(subj[c4.t_epoch == "left"].delta)
    di = {"S":a, "right": subj_right, "left": subj_left}
    c4_delta.append(di)

for a in np.unique(c3.S):
    subj = c3[c3.S == a]
    subj_right = np.mean(subj[c3.t_epoch == "right"].delta)
    subj_left = np.mean(subj[c3.t_epoch == "left"].delta)
    di = {"S":a, "right": subj_right, "left": subj_left}
    c3_delta.append(di)

csv_columns = ["S","right","left"]

csv_file = "real_C4_deltas_mean_sub.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in c4_delta:
            writer.writerow(data)
except IOError:
    print("I/O error")

csv_file = "real_C3_deltas_mean_sub.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in c3_delta:
            writer.writerow(data)
except IOError:
    print("I/O error")

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mne

c4 = pd.read_csv("real_C4_deltas_mean_sub.csv")
c3 = pd.read_csv("real_C3_deltas_mean_sub.csv")

c4_right_mean = np.mean(c4.right)
c4_left_mean = np.mean(c4.left)
c3_right_mean = np.mean(c3.right)
c3_left_mean = np.mean(c3.right)
means = [c4_right_mean, c4_left_mean, c3_right_mean, c3_left_mean]

c4_right_std = np.std(c4.right)
c4_left_std = np.std(c4.left)
c3_right_std = np.std(c3.right)
c3_left_std = np.std(c3.right)
stds = [c4_right_std, c4_left_std, c3_right_std, c3_left_std]

labels = ["C4_right", "C4_left", "C3_right", "C3_left"]
x = np.arange(len(labels))  # the label locations

# Build the plot
fig, ax = plt.subplots()
ax.bar(x, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('(base-task)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title('')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot.png')
plt.show()


#%%
#single df
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mne

c4_imagined = pd.read_csv("C4_deltas_mean_sub.csv")
c4_imagined.columns = ["S","c4_right","c4_left"]
c3_imagined = pd.read_csv("C3_deltas_mean_sub.csv")
c3_imagined.columns = ["S","c3_right","c3_left"]

c4_real = pd.read_csv("real_C4_deltas_mean_sub.csv")
c4_real.columns = ["S","c4_right","c4_left"]
c3_real = pd.read_csv("real_C3_deltas_mean_sub.csv")
c3_real.columns = ["S","c3_right","c3_left"]

c4_imagined_right_mean = np.mean(c4_imagined.c4_right)
c4_imagined_right_std = np.std(c4_imagined.c4_right)
c4_imagined_left_mean = np.mean(c4_imagined.c4_left)
c4_imagined_left_std = np.std(c4_imagined.c4_left)
c3_imagined_right_mean = np.mean(c3_imagined.c3_right)
c3_imagined_right_std = np.std(c3_imagined.c3_right)
c3_imagined_left_mean = np.mean(c3_imagined.c3_left)
c3_imagined_left_std = np.std(c3_imagined.c3_left)

c4_real_right_mean = np.mean(c4_real.c4_right)
c4_real_right_std = np.std(c4_real.c4_right)
c4_real_left_mean = np.mean(c4_real.c4_left)
c4_real_left_std = np.std(c4_real.c4_left)
c3_real_right_mean = np.mean(c3_real.c3_right)
c3_real_right_std = np.std(c3_real.c3_right)
c3_real_left_mean = np.mean(c3_real.c3_left)
c3_real_left_std = np.std(c3_real.c3_left)

x = np.arange(4)
#Plot c4 results
fig, ax = plt.subplots()
plt.bar(x, [c4_imagined_right_mean, c4_imagined_left_mean, c4_real_right_mean, c4_real_left_mean], yerr=[c4_imagined_right_std, c4_imagined_left_std, c4_real_right_std, c4_real_left_std], align='center', alpha=0.5, ecolor='black', capsize=10)
plt.xticks(x, ("c4_imagined_mov_right_mean = " + str(np.round(c4_imagined_right_mean, 2)),
               "c4_imagined_mov_left_mean = " + str(np.round(c4_imagined_left_mean, 2)),
               "c4_real_mov_right_mean = " + str(np.round(c4_real_right_mean, 2)),
               "c4_real_mov_left_mean = " + str(np.round(c4_real_left_mean, 2))))
# ax.set_xticklabels("c4_imagined_right_mean", "c4_imagined_left_mean", "c4_real_right_mean", "c4_real_left_mean")
plt.title('C4')

# Save the figure and show
plt.tight_layout()
plt.show()

#Plot c3 results
fig, ax = plt.subplots()
plt.bar(x, [c3_imagined_right_mean, c3_imagined_left_mean, c3_real_right_mean, c3_real_left_mean], yerr=[c3_imagined_right_std, c3_imagined_left_std, c3_real_right_std, c3_real_left_std], align='center', alpha=0.5, ecolor='black', capsize=10)
plt.xticks(x, ("c3_imagined_mov_right_mean" + str(np.round(c3_imagined_right_mean, 2)),
               "c3_imagined_mov_left_mean" + str(np.round(c3_imagined_left_mean, 2)),
               "c3_real_mov_right_mean" + str(np.round(c3_real_right_mean, 2)),
               "c3_real_mov_left_mean" + str(np.round(c3_real_left_mean, 2))))
# ax.set_xticklabels("c4_imagined_right_mean", "c4_imagined_left_mean", "c4_real_right_mean", "c4_real_left_mean")
plt.title('c3')

# Save the figure and show
plt.tight_layout()
plt.show()


best_c4_imagine = len([x for x in c4_imagined.c4_left.tolist() if x > c4_imagined_left_mean])
best_c4_imagine_mean = np.mean([x for x in c4_imagined.c4_left.tolist() if x > c4_imagined_left_mean])
best_c4_real = len([x for x in c4_real.c4_left.tolist() if x > c4_real_left_mean])
best_c4_real_mean = np.mean([x for x in c4_real.c4_left.tolist() if x > c4_real_left_mean])

best_c3_imagine = len([x for x in c3_imagined.c3_left.tolist() if x > c3_imagined_left_mean])
best_c3_imagine_mean = np.mean([x for x in c3_imagined.c3_left.tolist() if x > c3_imagined_left_mean])
best_c3_real = len([x for x in c3_real.c3_left.tolist() if x > c3_real_left_mean])
best_c3_real_mean = np.mean([x for x in c3_real.c3_left.tolist() if x > c3_real_left_mean])

x = np.arange(4)
#Plot best subject results
fig, ax = plt.subplots()
plt.bar(x, [best_c4_imagine_mean, best_c4_real_mean, best_c3_imagine_mean, best_c3_real_mean], align='center', alpha=0.5, ecolor='black', capsize=10)
plt.xticks(x, ("best_c4_imagine_mean = " + str(np.round(best_c4_imagine_mean, 2)) + "n= " + str(best_c4_imagine),
               "best_c4_real_mean = " + str(np.round(best_c4_real_mean, 2))+ "n= " + str(best_c4_real),
               "best_c3_imagine_mean = " + str(np.round(best_c3_imagine_mean, 2))+ "n= " + str(best_c4_imagine),
               "best_c3_real_mean = " + str(np.round(best_c3_real_mean, 2)) + "n= " + str(best_c3_real)))
# ax.set_xticklabels("c4_imagined_right_mean", "c4_imagined_left_mean", "c4_real_right_mean", "c4_real_left_mean")
plt.title('best subject means')

# Save the figure and show
plt.tight_layout()
plt.show()

