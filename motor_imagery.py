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
#Scarico il database
from mne.datasets import eegbci
from mne.decoding import CSP

#Documentazione mne Python https://mne.tools/0.16/documentation.html
#%%

tmin, tmax = -1., 4.
event_id = dict(hands=2, feet=3)
sogetto = 2
runs = [6, 10, 14]  #le run che ci servono

#Funzione di mne-Python, prende il numero del sogetto e le rispettive run e le scarica
raw_fnames = eegbci.load_data(sogetto, runs)

#Questa funzione concatena i tre file precedentemente scaricati in un unico file
#Le registrazioni sono di 120 secondi troverete una scritta rossa dove inzia una e finisce l'altra
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])

#Visualizzo i dati raw
raw.plot()















