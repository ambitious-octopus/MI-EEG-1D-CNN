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
runs = [4]
#Scarico i dati e mi ritorna il path locale
raw_fnames = eegbci.load_data(subject, runs)
#Concateno le path e le metto in un unico oggetto
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
raw.plot()
#%%
#Applico un filtro passa banda
raw.filter(1.0, 79.0, fir_design='firwin', skip_by_annotation='edge')
#Applico un NotchFilter
freqs = (60)
raw_notch = raw.notch_filter(freqs=freqs)
raw.plot_psd(area_mode=None, show=False, average=False, fmin=1.0, fmax=80.0, dB=False, n_fft=160)
#%% ICA
#Istanzio una Ica
ica = ICA(n_components=64, random_state=10, method="fastica", max_iter=1000) #Deve ritornare due tuple!
#Faccio il fit
ica.fit(raw)
#%%
eog_inds, eog_scores = ica.find_bads_eog(raw, ch_name='Fpz')
#Plotto le concentrazioni
ica.plot_sources(raw)
ica.plot_components()
#PLotto le proprietà della singola componente
ica.plot_properties(raw, dB=False,plot_std=False, picks=[0])
#%%
#Definisco delle componenti da escludere
exc = [1,0,12,11,18,29,28,36,34,49,47,45,63,51,52,53,56]
attesa = [4,23,49]
prot = [51,1,12, 36, 27]
#%%
reconst_raw = raw.copy()
#O questa
ica.plot_overlay(reconst_raw, exclude=exc)
#O questa
ica.apply(reconst_raw, exclude=exc)
reconst_raw.plot_psd(area_mode=None, show=False, average=False, fmin=1.0, fmax=80.0, dB=False, n_fft=160)