"""
Created by Francesco Mattioli
Francesco@nientepanico.org
www.nientepanico.org

References
----------

.. [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
       Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface
       (BCI) System. IEEE TBME 51(6):1034-1043.
.. [1] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
       Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank,
       PhysioToolkit, and PhysioNet: Components of a New Research Resource for
       Complex Physiologic Signals. Circulation 101(23):e215-e220.

IMPORTANTE: MNE-python cambia drasticamente da versione a versione
Versione utilizzata per questo script --> 0.19.2
Documentazione relativa a questa versione: https://mne.tools/stable/overview/index.html

"""
#%% Importo i moduli necessari
import numpy as np
import matplotlib.pyplot as plt
from mne import Epochs, pick_types, create_info, events_from_annotations
from mne.channels import read_layout, montage, read_custom_montage, make_standard_montage, find_layout, make_eeg_layout, make_dig_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,corrmap)
from mne.datasets import eegbci

#%%
#Numero del sogetto (va da 1 a 109)
soggetto = 1
#Offset evento
tmin, tmax = -1., 4.
#Prendo le run, per ora prendo solo quelle relative al movimento immaginato "task 4" in tutto sono 14 run e 4 task
runs = [4]
#Il database che staimo utilizzato è scaricabile direttamente da PhysioNet con una funzione di MNE-python
#Prendo il numero del sogetto e le rispettive run e scarico i dati (di default = All'interno della directory principale mne-python)
raw_fnames = eegbci.load_data(soggetto, runs)
#I blocchi del task sono di 120 secondi, quindi questo file raw sarà coposto di tanti blocchi quanti ne sono stati richiesti (variabile runs)
#In questo caso 3 relativi alla variabile runs
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])

#%%
#Mappo i canali
mapping = {
    'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCz', 'Fc2.': 'FC2',
    'Fc4.': 'FC4', 'Fc6.': 'FC6', 'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1',
    'Cz..': 'Cz', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6', 'Cp5.': 'CP5',
    'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPz', 'Cp2.': 'CP2', 'Cp4.': 'CP4',
    'Cp6.': 'CP6', 'Fp1.': 'Fp1', 'Fpz.': 'Fpz', 'Fp2.': 'Fp2', 'Af7.': 'AF7',
    'Af3.': 'AF3', 'Afz.': 'AFz', 'Af4.': 'AF4', 'Af8.': 'AF8', 'F7..': 'F7',
    'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'Fz', 'F2..': 'F2',
    'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8', 'Ft7.': 'FT7', 'Ft8.': 'FT8',
    'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10', 'Tp7.': 'TP7',
    'Tp8.': 'TP8', 'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1',
    'Pz..': 'Pz', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8',
    'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POz', 'Po4.': 'PO4', 'Po8.': 'PO8',
    'O1..': 'O1', 'Oz..': 'Oz', 'O2..': 'O2', 'Iz..': 'Iz'}

raw.rename_channels(mapping)
raw.set_montage('standard_1005')

#%%
#Estraggo le informazioni dal file edf
#Frequenza di campionamento
sfreq = raw.info['sfreq']
#Gli eventi
events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
#La tipologia
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,exclude='bads')
#%%
#Viasualizzo Il Power Spectral Density
raw.plot_psd(area_mode=None, tmax=10.0, show=False, average=False, fmin =1.0, fmax=80.0, dB=False)
# Applico un band-pass filter
raw.filter(l_freq=1.,h_freq=79.9, fir_design='firwin2', skip_by_annotation='edge')
#Visualizzo i dati raw, dopo il filtraggio
raw.plot(duration=5, n_channels=25)
#Visualizzo la PSD dopo il filtraggio
raw.plot_psd(area_mode='range', tmax=10.0, show=False, average=True)
#%%
#Instanzio un oggetto ica
ica = ICA(n_components=64, random_state=42, method="picard", max_iter=200)
#Applico la ica sul file raw
ica.fit(raw)
#Plotto le componenti indipendenti estratte in formato di onda
ica.plot_sources(raw)
#Plotto le componenti indipendenti estratte sullo scalpo
ica.plot_components()
#Questa funzione dovrebbe plottare la concentrazione delle componenti
ica.plot_properties(raw, picks=[1])
#Ricarico il raw originale
raw.load_data()
#Vedo la differenze togliendo alcune componenti
ica.plot_overlay(raw, exclude=[0,1,19,29], picks='eeg')
#Applico l'esclusione
ica.exclude = [0,1,19,29]
#Ricostrusco il tracciato
reconst_raw = raw.copy()
ica.apply(reconst_raw)
#Visualizzo il raw ricostruito
reconst_raw.plot(duration=5, n_channels=25)
#%%
#Divido le epoche
event_id = dict(left=2, right=3)
epochs = Epochs(reconst_raw, events, event_id, tmin, tmax, proj=True, picks=picks,baseline=None, preload=True)
labels = epochs.events[:, -1] - 2
epochs_data = epochs.get_data()




