from pirate import Pirates
import os
import numpy as np
import mne
import matplotlib.pyplot as plt

cwd = os.getcwd()
preprocessing = os.path.join(cwd, "preprocessing")
dir_imagined = os.path.join(preprocessing, "imagined")

ch = ["C1", "C2", "C3", "C4"]
raw = mne.io.read_raw_fif(os.path.join(cwd, "S001" + ".fif"))
raw2 = mne.io.read_raw_fif(os.path.join(cwd, "S002" + ".fif"))

raws = []

raws.append(raw)
raws.append(raw2)

data = os.path.join(cwd, "data")

Pirates.image_generation(raws,data)


