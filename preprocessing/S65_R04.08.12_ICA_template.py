import numpy as np
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.preprocessing import ICA, corrmap
import time
import os
#%%

#Lista dei file raw
raws = list()
#Lista degli oggetti ica
icas = list()
#Lista dei soggetti
subjects = [65,79,105,25,57]
#Lista delle runs
runs = [4,8,12]


for subj in subjects:
    ls_run = [] #Lista dove inseriamo le run
    for run in runs:
        fname = eegbci.load_data(subj, runs=run)[0] #Prendo le run
        raw_run = read_raw_edf(fname, preload=True) #Le carico
        len_run = np.sum((raw_run._annotations).duration) #Controllo la durata
        if len_run > 123:
            raw_run.crop(tmax=124.4) #Taglio la parte finale
        ls_run.append(raw_run) #Aggiungo la run alla lista delle run
    raw = concatenate_raws(ls_run)
    eegbci.standardize(raw) #Cambio i nomi dei canali
    montage = make_standard_montage('standard_1005') #Caricare il montaggio
    raw.set_montage(montage) #Setto il montaggio
    raw.filter(1.0, 79.0, fir_design='firwin', skip_by_annotation='edge') #Filtro
    raw_notch = raw.notch_filter(freqs=60) #Faccio un filtro passa banda
    raw.plot_psd(area_mode=None, show=False, average=False, fmin =1.0, fmax=80.0, dB=False, n_fft=160)
    # todo: qui salvare il plot psd
    # Ica
    ica = ICA(n_components=64, random_state=42, method="fastica", max_iter=1000)

    ind = []
    for index, value in enumerate((raw.annotations).description):
        if value == "BAD boundary" or value == "EDGE boundary":
            ind.append(index)
    (raw.annotations).delete(ind)

    ica.fit(raw)
    raws.append(raw)
    icas.append(ica)

#%%
eog_inds, eog_scores = icas[0].find_bads_eog(raws[0], ch_name='Fpz')
icas[0].plot_properties(raws[0], picks=[], dB=False)
exc_0 = [0,1,3,4,9,11,14,15,19,26,29,34,44,47,49,51,53,55,56,58,59,63,60,61]
maybe = [ 7,12,23,24,40]
reconst_raw = raws[0].copy()
icas[0].plot_overlay(reconst_raw, exclude=exc_65)
reconst_raw.plot_psd(area_mode=None, show=False, average=False, fmin=1.0, fmax=80.0, dB=False, n_fft=160)

#%%
exc_1 = [1,2,3]

ex_list = [exc_0,exc_1]

#todo: Creiamo due cicli per
index_sub_temp = [0]
for index in index_sub_temp:
    print(index)
    for list_comp in ex_list:
        print(list_comp)
        for comp in list_comp:
            print(comp)
            #corr_map = corrmap(icas, template=(index, comp), plot=False, threshold=0.85, label="artifact_" + str(index))

# todo: capire come fare questa corr map

#%%

#todo: fare parte che riscostruisce tutti raw sulla base delle componenti escluse attraverso corr_map

