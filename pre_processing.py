import numpy as np
import matplotlib.pyplot as plt

from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.preprocessing import ICA

#%% CARICO IL DATABASE
tmin, tmax = -1., 4.
event_id = dict(hands=2, feet=3)
subject = 1
runs = [2]

#Scarico i dati e mi ritorna il path locale
raw_fnames = eegbci.load_data(subject, runs)
#Concateno le path e le metto in un unico file
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
#Standardizzo la posizione degli elettrodi
eegbci.standardize(raw)
#Seleziono il montaggio
montage = make_standard_montage('standard_1005')
#Lo setto all'interno di raw
raw.set_montage(montage)
#Tolgo il punto alla fine
raw.rename_channels(lambda x: x.strip('.'))
raw.crop(tmax = 60).load_data()

#%% VISUALIZZO I DATI RAW
raw.plot_psd(area_mode=None, show=False, average=False, fmin =1.0, fmax=80.0, dB=False, n_fft=160)
raw.plot()
#Applico un filtro passa banda
raw.filter(1., 79., fir_design='firwin', skip_by_annotation='edge')
raw.plot_psd(area_mode=None, show=False, average=False, fmin=1.0, fmax=80.0, dB=False, n_fft=160)

#%% ICA
#Istanzio una Ica
ica = ICA(n_components=64, random_state=42, method="fastica", max_iter=1000)
#Faccio il fit
ica.fit(raw)
#Plotto le concentrazioni
ica.plot_sources(raw)

ica.plot_properties(raw, dB=False,plot_std=False, picks=[51])

exc = [0, 7, 9, 18, 25, 39, 40, 41, 42, 43, 44]
notch = [4, 12, 13, 14, 15, 17, 19, 26, 27, 28, 35, 49,50]

ica.plot_overlay(raw, exclude=exc, picks='eeg')

ica.exclude = exc

reconst_raw = raw.copy()

ica.apply(reconst_raw)

reconst_raw.plot_psd(area_mode=None, show=False, average=False, fmin=1.0, fmax=80.0, dB=False, n_fft=160)


ica.save("ica08_04")



