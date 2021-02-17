import numpy as np
from data_processing.general_processor import Utils
import mne
from mne.io import Raw

subj = [1]
runs = [4, 6, 8, 10, 12, 14]
data_path = "D:\\datasets\\eegbci"
channels = ["C3..", "C4.."]
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

on_sets = list()
a = 0
for x in range(30):
    on_sets.append(a)
    a += real_window_size

while p2 <= limit:
    sample.append(np.array([data_real[0][p1:p2], data_real[1][p1:p2]]))

    if p1 in on_sets and label_slide < 30:
        label.append(ann[label_slide])
        label_slide+= 1
    else:
        label.append("nan")

    p1 += real_stride
    p2 += real_stride

windowed = np.stack(sample)
labels = np.array(label)





