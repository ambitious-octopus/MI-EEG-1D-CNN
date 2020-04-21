import numpy as np
import matplotlib.pyplot as plt

from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.preprocessing import ICA, corrmap
#%%

raws = list()
icas = list()

for subj in range(4):
    fname = eegbci.load_data(subj + 1, runs=[3])[0]
    raw = read_raw_edf(fname)
    eegbci.standardize(raw)
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    # Ica
    ica = ICA(n_components=64, random_state=97, method="fastica", max_iter=1000)
    ica.fit(raw)
    raws.append(raw)
    icas.append(ica)

raw = raws[0]
ica = icas[0]
eog_inds, eog_scores = ica.find_bads_eog(raw, ch_name='Fp1')
corr_map = corrmap(icas, template=(0, eog_inds[0]))
