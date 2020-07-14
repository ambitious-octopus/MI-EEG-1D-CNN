from pirate import Pirates
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from PIL import Image

cwd = os.getcwd()
preprocessing = os.path.join(cwd, "preprocessing")
path = os.path.join(cwd,'C1_resize.jpg')
dir_imagined = os.path.join(preprocessing, "imagined")

ch = ["C1", "C2", "C3", "C4"]
raw = mne.io.read_raw_fif(os.path.join(cwd, "S001" + ".fif"))
raw2 = mne.io.read_raw_fif(os.path.join(cwd, "S002" + ".fif"))

raws = []

raws.append(raw)
raws.append(raw2)

cvd = Pirates.image_generation(raws,cwd)

Pirates.image_resize()

ls = []
immagine = Image.open('C1.jpg')
ls.append(immagine)
im2 = Image.open('C2.jpg')
ls.append(im2)
out = immagine.resize((128, 128))
im_save = out.save(path)

img_path = os.path.join(cwd, "prova.jpg")
mixed.save(img_path)
# #todo: creare funzione che genera tutte le immagini le concatena e le salva con i nomi appropriati (S49_i_e2_C1C2C3C4_label)
#
# new = mixed.resize((int(np.floor(mixed.size[0]/2)), int(np.floor(mixed.size[1]/2))))
# mixed_arr = np.array(new)
#
# for e in epochs:
#     print("ciao") 731323