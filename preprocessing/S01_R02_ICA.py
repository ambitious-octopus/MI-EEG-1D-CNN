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
ax = plt.axes()
raw.plot_psd(area_mode=None, show=False, average=False, ax=plt.axes(ylim=(0,60)), fmin =1.0, fmax=80.0, dB=False, n_fft=160)
raw.plot()
#%%
#Applico un filtro passa banda
raw.filter(1., 79., fir_design='firwin', skip_by_annotation='edge')
raw.plot_psd(area_mode=None, show=False, average=False, fmin=1.0, fmax=80.0, dB=False, n_fft=160)
#Applico un NotchFilter
freqs = (60)
raw_notch = raw.notch_filter(freqs=freqs)

#%% ICA
#Istanzio una Ica
ica = ICA(n_components=64, random_state=10, method="fastica", max_iter=1000)
#Faccio il fit
ica.fit(raw)
#%%
#Plotto le concentrazioni
ica.plot_sources(raw)
#PLotto le proprietà della singola componente
ica.plot_properties(raw, dB=False,plot_std=False, picks=[61,57,51])
#%%
#Definisco delle componenti da escludere
exc = [0, 7, 9, 18, 25, 39, 40, 41, 42, 43, 44, 56, 58, 60,62]
#Definisco delle componenti da passare in un filtro
notch = [4, 12, 13, 14, 15, 17, 19, 26, 27, 28, 35, 49, 50, 51, 54, 57, 61]
#%%
reconst_raw = raw.copy()
ica.plot_overlay(reconst_raw, exclude=exc)
reconst_raw.plot_psd(area_mode=None, show=False, average=False,ax=plt.axes(ylim=(0,60)), fmin=1.0, fmax=80.0, dB=False, n_fft=160)
#%%
#Tolgo le componenti selezionate
sources = ica.get_sources(raw)
ica01 = sources.pick_channels(["ICA000"])
ica01_notch = ica01.notch_filter(freqs=freqs)
