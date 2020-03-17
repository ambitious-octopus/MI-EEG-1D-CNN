"""
Created by Francesco Mattioli
Francesco@nientepanico.org
www.nientepanico.org

References
----------

.. [1] Zoltan J. Koles. The quantitative extraction and topographic mapping
       of the abnormal components in the clinical EEG. Electroencephalography
       and Clinical Neurophysiology, 79(6):440--447, December 1991.
.. [2] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
       Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface
       (BCI) System. IEEE TBME 51(6):1034-1043.
.. [3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
       Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank,
       PhysioToolkit, and PhysioNet: Components of a New Research Resource for
       Complex Physiologic Signals. Circulation 101(23):e215-e220.
"""
#%% IMPORTO TUTTO IL NECESSARIO

import numpy as np
import matplotlib.pyplot as plt

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf

from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

from mne.datasets import eegbci
from mne.decoding import CSP


#Documentazione mne Python https://mne.tools/0.16/documentation.html
#%%

#Numero del sogetto (1-109)
soggetto = 1

tmin, tmax = -1., 4. 
event_id = dict(hands=2, feet=3)

#le run che ci servono, per ora prendo solo quelle revalive al movimento immaginato "task 4"
runs = [6, 10, 14, ]

#Metodo di mne-Python, per il database in esame, prende il numero del sogetto e le rispettive runs e le scarica
raw_fnames = eegbci.load_data(soggetto, runs)

#I blocchi del task sono di 120 secondi, quindi questo file raw sarà coposto di tanti blocchi quanti ne sono stati richiesti, 
#In questo caso 3 relativi alla variabile runs
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])

#Il nome dei canali contiene un "." finale che da problemi, lo elimino
raw.rename_channels(lambda a: a.strip('.'))

# Applico un band-pass filter
raw.filter(l_freq=1.,h_freq=79., fir_design='firwin2', skip_by_annotation='edge')

events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

#Visualizzo i dati raw
raw.plot(duration=5, n_channels=25)

#%%
#applicazione della ICA

eog_evoked = create_eog_epochs(raw).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

















