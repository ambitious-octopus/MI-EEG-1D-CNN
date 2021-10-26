import numpy as np
from data_processing.general_processor import Utils
import mne
from typing import List, Tuple
from mne.io import Raw

subj = [1,2]
runs = [4, 6, 8, 10, 12, 14]
data_path = "D:\\datasets\\eegbci"

#par
data = Utils.load_data(subj, runs=runs, data_path=data_path)


"""
THIS SCRIPT: |-------------| len = 640
"""

stride = 1.2
window_size = 4.2
channels = ["C3..", "C4.."]

#const
REAL_WIN_SIZE = window_size*160
REAL_STRIDE = stride*160

def win_slice(data, window_size, stride, threshold):
    """
    Questo assume, che la lunghezza massima della finestra è 4 secondi * 160

    :param data: una singola run
    :param window_size:
    :param stride:
    :return:
    """
    labels = list()
    for duration, label in zip(data.annotations.duration, data.annotations.description):
        labels += np.full((int(duration * 160)), label, dtype=str).tolist()
    labels = np.array(labels)
    result_data = list()
    result_label = list()
    real_data = data.get_data()
    P1, P2 = 0, window_size
    while P2 < real_data.shape[1]:
        result_data.append(real_data[:, P1:P2])
        win_label = labels[P1:P2]
        if len(np.unique(win_label)) == 1:
            result_label.append(np.unique(win_label)[0])
        else:
            count_label = sorted([[np.unique(win_label)[0], (win_label == np.unique(win_label)[0]).sum()],
                            [np.unique(win_label)[1], (win_label == np.unique(win_label)[1]).sum(
                            )]], key=lambda x: x[1], reverse=True)
            rapp = count_label[1][1]/count_label[0][1]
            if rapp == 1:
                result_label.append(win_label[-1])
            elif rapp >= threshold:
                result_label.append(count_label[0][0])
            else:
                result_label.append(count_label[1][0])
            # se è più di una, mi devi fare il rapporto tra quelle che ci sono
            # Quella che ha il rapporto maggiore alla threshold sarà la label
        P1 += stride
        P2 += stride
    return result_data, result_label

data, label = win_slice(data[0][0], window_size=3*160, stride=160, threshold=0.5)






#func
result = dict()
for subj in data:
    subj_name = repr(subj[0])[10:14]
    result[subj_name] = dict()
    for run in subj:
        run_name = repr(run)[14:17]
        result[subj_name][run_name] = list()
        #create window
        sliced_data = run.pick_channels(ch_names=channels, ordered=True).get_data()
        MAX = len(sliced_data[0])
        P1, P2 = 0, REAL_WIN_SIZE
        samples = list()
        labels = list()
        ON_SETS = run.annotations.onset
        MAX_ON_SETS = len(run.annotations.onset)
        n_label_slides = 0
        ANN = run.annotations.description

        while P2 <= MAX:
            P1 = int(np.round(P1))
            P2 = int(np.round(P2))
            samples.append(np.array([sliced_data[0][P1:P2], sliced_data[1][P1:P2]]))

            # print(P1 / 160)
            # print(P2/ 160)
            # create labels
            if P1 in ON_SETS and n_label_slides < MAX_ON_SETS:
                labels.append(ANN[n_label_slides])
                n_label_slides += 1
            else:
                labels.append("nan")

            P1 += REAL_STRIDE
            P2 += REAL_STRIDE

        result[subj_name][run_name].append(np.stack(samples))
        result[subj_name][run_name].append(np.array(labels))





#%%
import numpy as np
from data_processing.general_processor import Utils
import mne
from typing import List, Tuple
from mne.io import Raw
channels = ["C3..", "C4.."]
subj = [1,2]
runs = [4, 6, 8, 10, 12, 14]
data_path = "D:\\datasets\\eegbci"
#par
data = Utils.load_data(subj, runs=runs, data_path=data_path)

r = data[0][0]
data_real = data[0][0].pick_channels(ch_names=channels, ordered=True).get_data()

stride = 1 #in seconds
real_stride = stride*160
limit = len(data_real[0])
window_size = 4 #in seconds
real_window_size = 4*160
p1, p2 = 0, real_window_size
label_slide = 0
ann = r._annotations.description
sample = list()
label = list()
time = []
on_sets = list()
a = 0
for x in range(30):
    on_sets.append(a)
    a += real_window_size
while p2 <= limit:
    sample.append(np.array([data_real[0][p1:p2], data_real[1][p1:p2]]))

    if p1 in on_sets and label_slide < 30:
        label.append(ann[label_slide])
        time.append(dict(start=p1, end=p2, task=ann[label_slide]))
        label_slide+= 1
    else:
        label.append("nan")
        time.append(dict(start=p1, end=p2, task="nan"))


    # dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28', Resource="Alex"),

    p1 += real_stride
    p2 += real_stride
windowed = np.stack(sample)
labels = np.array(label)
import pandas as pd
t = pd.DataFrame(time)



