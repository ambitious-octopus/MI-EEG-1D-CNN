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
events, _ = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))
picks = mne.pick_channels(raw.info["ch_names"], ch)
tmin, tmax = 0, 3
event_ids = dict(base=1, left=2, right=3)
epochs = mne.Epochs(raw, events, event_ids, tmin=tmin,tmax=tmax, picks=picks, baseline=None, preload=True)

freqs = np.arange(5,38,1)  # frequencies from 2-35Hz
n_cycles = freqs # use constant t/f resolution
#cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
e = epochs[10]
# Run TF decomposition overall epochs
tfr = mne.time_frequency.tfr_multitaper(e, freqs=freqs, n_cycles=n_cycles,
                     use_fft=True, return_itc=False, average=True,
                     decim=1)

# a = tfr.plot(["C3"], cmap="jet", vmin=0, vmax=0.000000009, colorbar=True)
lst_path_img = list()
for channel in ch:
    a = tfr.plot([channel], cmap="jet", vmin=0, vmax=0.000000009, colorbar=True)
    a.savefig(str(channel) + ".jpg")
    lst_path_img.append(os.path.join(cwd, str(channel) + ".jpg"))
    plt.close("all")

from PIL import Image, ImageOps
imgs = []
for im in lst_path_img:
    img = Image.open(im)
    imgs.append(img)

new_imgs = []
new_imgs_names = list()
for im, pa in zip(imgs,lst_path_img):
    w, h = im.size
    border = (81, 59, 163, 53)
    new = ImageOps.crop(im, border)
    new_imgs.append(new)
    new.save(pa[-6:])
    new_imgs_names.append(pa[-6:-4])

len_big_image = int(len(new_imgs) / 2)
width, height = new_imgs[0].size
real_width = width*len_big_image
real_height = height*len_big_image
mixed = Image.new('RGB', (real_width, real_height))

a1 = (0,0)
a2 = (new_imgs[0].size[0], 0)
a3 = (0, new_imgs[0].size[1])
a4 = (new_imgs[0].size[0], new_imgs[0].size[1])
position = [a1,a2,a3,a4]
for im, pos in zip(new_imgs, position):
    mixed.paste(im, pos)

img_path = os.path.join(cwd, "prova.jpg")
mixed.save(img_path)
#todo: creare funzione che genera tutte le immagini le concatena e le salva con i nomi appropriati (S49_i_e2_C1C2C3C4_label)

new = mixed.resize((int(np.floor(mixed.size[0]/2)), int(np.floor(mixed.size[1]/2))))
mixed_arr = np.array(new)

for e in epochs:
    print("ciao")