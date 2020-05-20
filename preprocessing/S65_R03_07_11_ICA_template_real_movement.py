import numpy as np
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.preprocessing import ICA, corrmap
import os
#%% Creating directories for saving plots psd
#Checking current work directory
cwd = os.getcwd()
print(cwd)
#Verifying "Preprocessing" directory existence
#If not creating one
dir_preprocessing = os.path.join(cwd, 'preprocessing')
if os.path.isdir(dir_preprocessing):
    print('Preprocessing directory already exists')
else:
    print('Path not found, creating preprocessing directory...')
    os.mkdir(dir_preprocessing)
#Verifying dir_psd_imagined directory existence
#If not creating one
dir_psd_real = os.path.join(dir_preprocessing, 'psd_real')
if os.path.isdir(dir_psd_real):
    print('Psd_real directory already exists')
else:
    print('Path not found, creating psd_real directory...')
    os.mkdir(dir_psd_real)
#Verifying pre_psd directory existence
#If not creating one
dir_pre_psd = os.path.join(dir_psd_real,'pre_psd')
if os.path.isdir(dir_pre_psd):
    print('Pre_psd directory already exists')
else:
    print("Path not found, creating pre_psd directory...")
    os.mkdir(dir_pre_psd)
#Verifying post_psd directory existence
#If not creating one
dir_post_psd = os.path.join(dir_psd_real,'post_psd')
if os.path.isdir(dir_post_psd):
    print('Post_psd directory already exists')
else:
    print("Path not found, creating post_psd directory...")
    os.mkdir(dir_post_psd)
#%%
#Lista dei file raw
raws = list()

raws_filtered = list()
#Lista degli oggetti ica
icas = list()
#Lista dei soggetti
subjects = [66,23,31,46,109,95,38,102,77,12,57]
#Lista delle runs
runs = [3,7,11]

ica_true = True

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
    #todo: azzerare variabile
    plot_pre_psd = raw.plot_psd(area_mode=None, show=True, average=False, fmin =1.0, fmax=80.0, dB=False, n_fft=160)
    psd_name = os.path.join(dir_pre_psd,'S'+ str(subj) + '_real_pre.png')
    if os.path.exists(psd_name):
     raise Exception('Warning! This plot already exists! :(' )
    plot_pre_psd.savefig(psd_name)
    # Ica
    #del plot_pre_psd
    ica = ICA(n_components=64, random_state=42, method="fastica", max_iter=1000)

    ind = []
    for index, value in enumerate((raw.annotations).description):
        if value == "BAD boundary" or value == "EDGE boundary":
            ind.append(index)
    (raw.annotations).delete(ind)


    raws_filtered.append(raw)
    
    if ica_true == True:
        ica.fit(raw)
        raws.append(raw)
        icas.append(ica)
    else:
        continue

corrmap(icas, template=(0, 33), plot=True, threshold=0.70, label = 'artifact_')
#icas[0].plot_properties(raws[0], picks=[26,29,34,44,47,49,51,53,55,56,58,59,63,60,61], dB=False)

#%%
#eog_inds, eog_scores = icas[0].find_bads_eog(raws[0], ch_name='Fpz')
icas[10].plot_properties(raws[0], picks=[1,42,43,44,45,46,47,48,49,50,51,52], dB=False)


#components [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
#21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
#41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60
# 61,62,63]

#exc_0 = [0,8,9,10,13,15,18,28,31,32,49,43,44,51,57]
#maybe = [ ]
#eyes= [0,]
#ecg=
#movement=

#template from subject 66
exc_0_66 = [0,1,3,13,14,17,29,32,33,35,47,50,53]
eyes = [0,1,3,14,33,35,47]
ecg = []
movement =[17,32,] 
odd = [13]
electrode =[53]

#reconst_raw = raws[0].copy()
#icas[0].plot_overlay(reconst_raw, exclude=exc_0)
#reconst_raw.plot_psd(area_mode=None, show=False, average=False, fmin=1.0, fmax=80.0, dB=False, n_fft=160)

#%%
ex_list = [exc_0_66]
index_sub_temp = [0]

for index in index_sub_temp:
    for list_comp in ex_list:
        for comp in list_comp:
            corr_map = corrmap(icas, template=(index, comp), plot=False, threshold=0.85, label="artifact_" + str(index))
#print([ica.labels_ for ica in icas])
# todo: capire come fare questa corr map

#%%
reco_raws = []
for index, ica in enumerate(icas):
    reco_raw = raws[index].copy()
    icas[index].exclude = icas[index].labels_["artifact_0"]
    icas[index].apply(reco_raw)
    plot_post_psd = reco_raw.plot_psd(area_mode=None, show=False, average=False,fmin=1.0, fmax=80.0, dB=False, n_fft=160)
    psd_name_post = os.path.join(dir_post_psd, 'S' + str(subjects[index]) + '_real_post.png')
    if os.path.exists(psd_name_post):
        raise Exception('Warning! This plot already exists! :(')
    plot_post_psd.savefig(psd_name_post)
    reco_raws.append(reco_raw)
