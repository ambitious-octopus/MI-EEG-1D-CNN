import mne
from mne.io import read_raw_edf
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

data_path = "E:\\datasets\\eegbci"

subject = 10
run = 6
sub_name = "S"+"0"+ str(subject)
sub_folder = os.path.join(data_path, sub_name)
path_run = os.path.join(sub_folder, sub_name+"R"+"0"+ str(run) + ".edf")
raw_run = read_raw_edf(path_run, preload=True)  # Le carico

raw_run.plot(duration=50)

