from pirate import Pirates
import os
import numpy as np
import mne

cwd = os.getcwd()
preprocessing = os.path.join(cwd, "preprocessing")
dir_imagined = os.path.join(preprocessing, "imagined")


raw = mne.io.read_raw_fif(os.path.join(cwd, "S001" + "_raw_sss.fif"))
events, _ = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))
picks = mne.pick_channels(raw.info["ch_names"], ["C3", "C4"])
tmin, tmax = 0, 3
event_ids = dict(base=1, left=2, right=3)
#epochs = mne.Epochs(raw, events, event_ids, tmin, tmax, picks=picks, baseline=None, preload=True)
epochs = mne.Epochs(raw, events, event_ids, tmin=tmin,tmax=tmax, picks=picks, baseline=None, preload=True)

import matplotlib.pyplot as plt

freqs = np.arange(1,36,1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
#cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
e = epochs[57]
# Run TF decomposition overall epochs
tfr = mne.time_frequency.tfr_multitaper(e, freqs=freqs, n_cycles=n_cycles,
                     use_fft=True, return_itc=False, average=True,
                     decim=2)


a = tfr.plot(["C3"], cmap="binary")
a.savefig("test.jpg")
#tfr2.plot(["C4"], cmap="Greys")


from PIL import Image, ImageOps
img = Image.open("test.jpg").convert("LA")
w, h = img.size
border = (81, 59, 163, 53)
new = ImageOps.crop(img, border)
new.show()
new.save("test.jpg")

np_frame = np.array(new)


