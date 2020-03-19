# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:20:48 2020

@author: xphid
"""


#importo le librerie 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf

import mne
from mne.preprocessing import ICA, create_ecg_epochs
from mne.datasets import sample

#scarico il database 
from mne.datasets import eegbci
from mne.decoding import CSP


#numero del soggetto (da 1 a 109)
subject= 1

tmin, tmax = -1.,4.
event_id=dict(handd=2, feet=3)

#le run che ci interessano (quindi quelle del task)
runs = [6, 10, 14]  # motor imagery: hands vs feet 


#metodo di mne : scarica i dati del soggetto in base al numero di indentificazione e le rispettive runs
raw_fnames =eegbci.load_data(subject,runs)

raw = concatenate_raws([read_raw_edf(f, preload = True ) for f in raw_fnames])

# il nome dei canali contiene un punto "." che dobbiamo togliere procedere con l'analisi
raw.rename_channels(lambda x: x.strip('.'))

#applico il filtro passa banda che permette il passaggio della banda di freq da 1 Hz a 90 Hz
# NB: in questo caso posso impostare il filtro solo a 79 Hz perchè in questo caso la  frequenza di Nyquist è settata a 80 Hz
raw.filter(1., 79., fir_design='firwin', skip_by_annotation='edge')

#evento = trigger dello stimolo (comando che do al soggetto)

events = find_events(raw, shortest_event=0, stim_channel='STI 014')

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Creo delle epoche di 1 - 2s 
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)
epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
labels = epochs.events[:, -1] - 2


#ICA: indipendent component analysis  (per togliere artefatti)
#provo ad usare il metodo della fast - ICA
method = 'fastica'

n_components = 25
decim = 3 
random_state = 23

# Definisco ICA 
ica = ICA(n_components=n_components, method=method, random_state=random_state)
print(ica)
