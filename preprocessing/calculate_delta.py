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

#Caricare tutti n_epoch reco_raw
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
        for n_epoch in range(len(epochs)):

            base = epochs[n_epoch].plot_psd(dB=False, tmin=-3, tmax=0)
            task = epochs[n_epoch].plot_psd(dB=False, tmin=0, tmax=3)
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
            plt.title(raw.__repr__()[7:11] + " Channel: " + channel + " epoch: " + str(epochs[n_epoch]._name) + " \n Delta: " + str(delta))
            plt.legend()
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Aplitude(µV/sqrt (Hz)")
            plt.grid(True)
            plt.axvspan(freq_pick - 1.5, freq_pick + 1.5, color='red', alpha=0.2)
            #plt.annotate('pick', xy=(freq_pick, amp_pick), xytext=(20, max(amp_base - 50)), arrowprops=dict(arrowstyle="->"), )
            plt.savefig(deltas_real + raw.__repr__()[7:11] + "ch_" + channel + "epoch_" + str(n_epoch) + "_" + epochs[n_epoch]._name + ".png")
            plt.close("all")
            dic = {"S":raw.__repr__()[7:11], "n_epoch": n_epoch, "t_epoch": epochs[n_epoch]._name, "delta" : delta}
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

data_path = "C:\\Users\\franc_pyl533c\\OneDrive\\Repository\\eeGNN\\data\\"

c4_imagined = pd.read_csv(data_path + "C4_deltas_mean_sub.csv")
c4_imagined.columns = ["S","c4_right","c4_left"]
c3_imagined = pd.read_csv(data_path +"C3_deltas_mean_sub.csv")
c3_imagined.columns = ["S","c3_right","c3_left"]

c4_real = pd.read_csv(data_path +"real_C4_deltas_mean_sub.csv")
c4_real.columns = ["S","c4_right","c4_left"]
c3_real = pd.read_csv(data_path +"real_C3_deltas_mean_sub.csv")
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

#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
import pickle

data_path = "C:\\Users\\franc_pyl533c\\OneDrive\\Repository\\eeGNN\\data\\"

c4_imagined = pd.read_csv(data_path + "C4_deltas_mean_sub.csv")
c4_imagined.columns = ["S","c4_right","c4_left"]
c3_imagined = pd.read_csv(data_path +"C3_deltas_mean_sub.csv")
c3_imagined.columns = ["S","c3_right","c3_left"]

c4_real = pd.read_csv(data_path +"real_C4_deltas_mean_sub.csv")
c4_real.columns = ["S","c4_right","c4_left"]
c3_real = pd.read_csv(data_path +"real_C3_deltas_mean_sub.csv")
c3_real.columns = ["S","c3_right","c3_left"]

#Take the "good" task for each channel
c4_img_good_task = c4_imagined[["S","c4_left"]]
c3_img_good_task = c3_imagined[["S","c3_right"]]

best_c4 = c4_img_good_task[c4_img_good_task["c4_left"] >= c4_img_good_task.c4_left.quantile(0.85)]
best_c3 = c3_img_good_task[c3_img_good_task["c3_right"] >= c3_img_good_task.c3_right.quantile(0.85)]

best_guy_c4= best_c4.S.to_list()
best_guy_c3= best_c4.S.to_list()

#Questi sono n_epoch soggetti che performano meglio sia su c3 che su c4
best_guys = list()
for guy in best_guy_c4:
    if guy in best_guy_c3:
        best_guys.append(guy)

# Here i have the best subj
# Now i slice all the subj

path_real_reco_data = "D:\\repo_data\\eegNN\\real_reco_raws"

#Load data
reco_data = list()
for subdir, dirs, files in os.walk(path_real_reco_data):
    for file in files:
        filepath = subdir + os.sep + file
        raw = mne.io.read_raw_fif(filepath, preload=True)
        reco_data.append(raw)

#Add to the info dic a key "S" with subject name
for raw in reco_data:
    s_name = raw.__repr__()[7:11]
    raw.info["S"] = s_name

#I take only the best guys
best_reco_data = [raw for raw in reco_data if raw.info["S"] in best_guys]


ch_plot_list = ["C4", "C3", "C1", "C2"]
save_dir = "D:\\repo_data\\eegNN\\deltas_mean_trial\\real\\"
#Estraggo tutti n_epoch delta
band_list = {"alpha": [[8,13],[1.5, 1.5]], "beta": [[14, 31],[6,6]], "alpha+beta":[8,31]}


for subj in best_reco_data:
    subj.info["alpha_deltas"] = dict()
    subj.info["beta_deltas"] = dict()
    subj.info["alpha+beta_deltas"] = dict()
    events, _ = mne.events_from_annotations(subj, event_id=dict(T1=2, T2=3))
    tmin, tmax = -3, 3
    event_ids = dict(left=2, right=3)
    for channel in subj.ch_names:
        epochs = mne.Epochs(subj, events, event_ids, tmin=tmin, tmax=tmax, picks=channel, baseline=None, preload=True)
        for epoch_type in epochs.event_id.keys():
            freq_base = list()
            amp_base = list()
            freq_task = list()
            amp_task = list()

            #Faccio prima solo movimento destro
            for n_epoch in range(len(epochs[epoch_type])):
                base = epochs[n_epoch].plot_psd(dB=False, tmin=-3, tmax=0)
                task = epochs[n_epoch].plot_psd(dB=False, tmin=0, tmax=3)
                plt.close("all")
                freq_base.append(base.gca().lines[2].get_xdata())
                amp_base.append(base.gca().lines[2].get_ydata())
                freq_task.append(task.gca().lines[2].get_xdata())
                amp_task.append(task.gca().lines[2].get_ydata())

            mean_freq_base = np.mean(freq_base, axis=0)
            mean_amp_base = np.mean(amp_base, axis=0)
            mean_freq_task = np.mean(freq_task, axis=0)
            mean_amp_task = np.mean(amp_task, axis=0)

            min_beta_and_alpha = 0
            max_beta_and_alpha = 0
            for band in band_list.keys():
                #A questo punto tratto tutto come una singola epoca
                base_freq_range = list()
                base_amp_range = list()
                result_base_freq = list()
                result_base_amp = list()
                result_task_amp = list()

                if band == "alpha+beta":
                    for f, a in zip(mean_freq_base, mean_amp_base):
                        if f >= min_beta_and_alpha and f <= max_beta_and_alpha:
                            base_freq_range.append(f)
                            base_amp_range.append(a)

                    for f, a in zip(base_freq_range, base_amp_range):
                        if a == max(base_amp_range):
                            freq_pick = f
                            amp_pick = a

                    for f, a in zip(base_freq_range, base_amp_range):
                        if f >= min_beta_and_alpha and f <= max_beta_and_alpha:
                            result_base_freq.append(f)
                            result_base_amp.append(a)

                    for f, a in zip(mean_freq_task, mean_amp_task):
                        if f >= min(result_base_freq) and f <= max(result_base_freq):
                            result_task_amp.append(a)
                    pass

                else:

                    for f, a in zip(mean_freq_base, mean_amp_base):
                        if f >= band_list[band][0][0] and f <= band_list[band][0][1]:
                            base_freq_range.append(f)
                            base_amp_range.append(a)

                    for f, a in zip(base_freq_range, base_amp_range):
                        if a == max(base_amp_range):
                            freq_pick = f
                            amp_pick = a

                    if band == "alpha":
                        min_beta_and_alpha = freq_pick - 1.5

                    if band == "beta":
                        max_beta_and_alpha = freq_pick + 6

                    for f, a in zip(base_freq_range, base_amp_range):
                        if f >= freq_pick - band_list[band][1][0] and f <= freq_pick + band_list[band][1][1]:
                            result_base_freq.append(f)
                            result_base_amp.append(a)

                    for f, a in zip(mean_freq_task, mean_amp_task):
                        if f >= min(result_base_freq) and f <= max(result_base_freq):
                            result_task_amp.append(a)

                delta = np.mean(result_base_amp) - np.mean(result_task_amp)

                if epoch_type not in subj.info[band + "_deltas"].keys():
                    subj.info[band + "_deltas"][epoch_type] = dict()
                if channel not in  subj.info[band + "_deltas"][epoch_type].keys():
                    subj.info[band + "_deltas"][epoch_type][channel] = 0
                subj.info[band + "_deltas"][epoch_type][channel] = delta

                if channel in ch_plot_list:
                    if band == "alpha+beta":
                        plt.plot(mean_freq_base, mean_amp_base, label="base")
                        plt.plot(mean_freq_task, mean_amp_task, label="task")
                        plt.title(subj.__repr__()[
                                  7:11] + " CH= " + channel + " BAND= " + band + " TYPE= " + epoch_type + " DELTA= " + str(
                            np.round(delta, 2)))
                        plt.legend()
                        plt.xlabel("Frequency (Hz)")
                        plt.ylabel("Aplitude(µV/sqrt (Hz)")
                        plt.grid(True)
                        plt.axvspan(min_beta_and_alpha, max_beta_and_alpha, color='red', alpha=0.2)
                        # plt.annotate('pick', xy=(freq_pick, amp_pick), xytext=(20, max(amp_base - 50)), arrowprops=dict(arrowstyle="->"), )
                        plt.savefig(save_dir + subj.__repr__()[
                                               7:11] + "ch_" + channel + "_band_" + band + "_type_" + epoch_type + ".png")
                        plt.close("all")

                    else:
                        plt.plot(mean_freq_base, mean_amp_base, label="base")
                        plt.plot(mean_freq_task, mean_amp_task, label="task")
                        plt.title(subj.__repr__()[
                                  7:11] + " CH= " + channel + " BAND= " + band + " TYPE= " + epoch_type + " DELTA= " + str(
                            np.round(delta, 2)))
                        plt.legend()
                        plt.xlabel("Frequency (Hz)")
                        plt.ylabel("Aplitude(µV/sqrt (Hz)")
                        plt.grid(True)
                        plt.axvspan(freq_pick - band_list[band][1][0], freq_pick + band_list[band][1][1], color='red',
                                    alpha=0.2)
                        # plt.annotate('pick', xy=(freq_pick, amp_pick), xytext=(20, max(amp_base - 50)), arrowprops=dict(arrowstyle="->"), )
                        plt.savefig(save_dir + subj.__repr__()[
                                               7:11] + "ch_" + channel + "_band_" + band + "_type_" + epoch_type + ".png")
                        plt.close("all")

def save_obj(obj, name):
    with open(save_subj_dir + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_subj_dir = "D:\\repo_data\\eegNN\\deltas_mean_trial\\best_subj_real\\"
for subj in best_reco_data:
    save_obj(subj, subj.info["S"])
    #subj.save(os.path.join(save_subj_dir, subj.__repr__()[7:11] + "raw.fif"), overwrite=True, picks=["eeg"])

result = dict()
for subj in best_reco_data:
    result[subj.info["S"]] = [subj.info["alpha+beta_deltas"]] + [subj.info["alpha_deltas"]] + [subj.info["beta_deltas"]]

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
import pickle

save_subj_dir = "D:\\repo_data\\eegNN\\deltas_mean_trial\\best_subj_real\\"

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)


post_data = list()
for subdir, dirs, files in os.walk(save_subj_dir):
    for file in files:
        filepath = subdir + os.sep + file
        raw = load_obj(filepath)
        post_data.append(raw)

topo_map_dir = "D:\\repo_data\\eegNN\\deltas_mean_trial\\real\\topo\\"
tasks = ["right","left"]
for subj in post_data:
    subj.info["alpha_deltas"]["right"] = np.array(list(subj.info["alpha_deltas"]["right"].values())).reshape(64,1)
    subj.info["alpha_deltas"]["left"] = np.array(list(subj.info["alpha_deltas"]["left"].values())).reshape(64, 1)
    subj.info["beta_deltas"]["right"] = np.array(list(subj.info["beta_deltas"]["right"].values())).reshape(64, 1)
    subj.info["beta_deltas"]["left"] = np.array(list(subj.info["beta_deltas"]["left"].values())).reshape(64, 1)
    subj.info["alpha+beta_deltas"]["right"] = np.array(list(subj.info["alpha+beta_deltas"]["right"].values())).reshape(64, 1)
    subj.info["alpha+beta_deltas"]["left"] = np.array(list(subj.info["alpha+beta_deltas"]["left"].values())).reshape(64, 1)

for subj in post_data:
    for task in tasks:

        fig, ax = plt.subplots(ncols=3, gridspec_kw=dict(top=0.9),
                                   sharex=True, sharey=True)
        fig.suptitle('Subject: ' + subj.info["S"] + " task:" + task, fontsize=16)

        mne.viz.plot_topomap(subj.info["alpha_deltas"][task][:, 0], subj.info, axes=ax[0],
                                 show=False)
        ax[0].set_title("alpha", fontweight='bold')
        mne.viz.plot_topomap(subj.info["beta_deltas"][task][:, 0], subj.info, axes=ax[1],
                                 show=False)
        ax[1].set_title("beta", fontweight='bold')
        mne.viz.plot_topomap(subj.info["alpha+beta_deltas"][task][:, 0], subj.info, axes=ax[2],
                                 show=False)
        ax[2].set_title("alpha+beta", fontweight='bold')
        fig.savefig(topo_map_dir + "topo_" + subj.info["S"] + "_task_" + task + ".png")
        plt.close("all")


