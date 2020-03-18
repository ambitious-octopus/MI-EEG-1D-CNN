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
from mne.channels import read_layout, montage, read_custom_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,corrmap)
from mne.datasets import eegbci


#%%

#Numero del sogetto (va da 1 a 109)
soggetto = 1

#Epoche da prendere in esame per l'analisi prese dal paper [1]
tmin, tmax = -1., 4.

#Eventi mani/piedi
event_id = dict(hands=2, feet=3)

#Prendo le run, per ora prendo solo quelle relative al movimento immaginato "task 4" in tutto sono 14 run e 4 task
runs = [6 ]

#Il database che staimo utilizzato è scaricabile direttamente da PhysioNet con una funzione di MNE-python
#Prendo il numero del sogetto e le rispettive run e scarico i dati (di default = All'interno della directory principale mne-python)
raw_fnames = eegbci.load_data(soggetto, runs)

#I blocchi del task sono di 120 secondi, quindi questo file raw sarà coposto di tanti blocchi quanti ne sono stati richiesti (variabile runs)
#In questo caso 3 relativi alla variabile runs
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])

#%%

#NON RUNNARE Aggiustare i valori
raw.rename_channels(lambda a: a.strip('.'))
montage = read_custom_montage()
raw.set_montage(montage)

#%%
#Estraggo le informazioni dal file edf
#Frequenza di campionamento
sfreq = raw.info['sfreq']

#Gli eventi
events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

#La tipologia
raw.pick_types(meg=False, eeg=True, stim=False, eog=False, exclude='bads')

#Il nome dei canali contiene un "." finale che da problemi, lo elimino
raw.rename_channels(lambda a: a.strip('.'))

# Applico un band-pass filter
raw.filter(l_freq=1.,h_freq=79., fir_design='firwin2', skip_by_annotation='edge')

#Visualizzo i dati raw
raw.plot(duration=5, n_channels=25)


#%%
#Instanzio un oggetto ica
ica = ICA(n_components=32, random_state=42, method="fastica", max_iter=200)

#Applico la ica sul file raw
ica.fit(raw)

#Plotto le componenti indipendenti estratte
ica.plot_sources(raw)

#Questa funzione dovrebbe plottare la concentrazione delle componenti
#ica.plot_properties(raw, picks=[0, 1])

#Ricarico il raw originale
raw.load_data()

#Vedo la differenze togliendo alcune componenti
ica.plot_overlay(raw, exclude=[0,1,19,29], picks='eeg')

#Applico l'esclusione
ica.exclude = [0,1,19,29]

#
reconst_raw = raw.copy()
ica.apply(reconst_raw)

#Visualizzo il raw ricostruito
reconst_raw.plot(duration=5, n_channels=25)







