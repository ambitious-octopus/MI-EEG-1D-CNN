import numpy as np
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.preprocessing import ICA, corrmap
import os
#%% Creating directories for saving plots psd

#returns folder directories for psd saving

def folders_psd ():
    
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
    
    
    #Verifying dir_psd_real directory existence
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
        
    return dir_preprocessing,dir_psd_real, dir_pre_psd,dir_post_psd

# returning folder directories 
        
dir_preprocessing,dir_psd_real, dir_pre_psd,dir_post_psd = folders_psd()

#%% Initializing: raws, icas, subjects, runs
    
#Lista dei file raw
raws = list()

#Lista degli oggetti ica
icas = list()

#Lista dei soggetti
subjects = [66,23,31]

#Lista delle runs
runs = [3,7,11]

#%% Load and crop raw data 

#returns list of lists of raw runs to be concatenated


def load_data ():
    
    ls_run_tot = [] #list of lists, contains lists of Raw files to be concatenated
                    # e.g. for subjects = [2, 45] 
                    #ls_run_tot = [[S2Raw03,S2Raw07,S2Raw11],
                    #              [S45Raw03,S45Raw07,S45Raw11]]
    
    for subj in subjects:
        ls_run = []#Lista dove inseriamo le run
        
        for run in runs:
            fname = eegbci.load_data(subj, runs=run)[0] #Prendo le run
            raw_run = read_raw_edf(fname, preload=True) #Le carico
            len_run = np.sum((raw_run._annotations).duration) #Controllo la durata
            
            if len_run > 123:
                raw_run.crop(tmax=124.4) #Taglio la parte finale
                
            ls_run.append(raw_run) 
            
        ls_run_tot.append(ls_run)
      
        
    return ls_run_tot

rawdata_loaded = load_data()
#list of lists of Raw objects of runs to be concatenated

#%% Concatenate raws 

# returns list of concatenated raw
# e.g. for subject = [2,45]
#returns list = [Raw_S2(concatenated), Raw_S45(concatenated)]


def concatenation():
    
    raw_conc_list = []
    
    for subj in range (len(subjects)):
          #print(subj)
          #print(rawdata_loaded[subj])
          raw_conc = concatenate_raws(rawdata_loaded[subj])
          
          raw_conc_list.append(raw_conc)
        
    return raw_conc_list

raw_conc_list = concatenation()

#%% Change channel names, set montage

#Returns list of raw setted with channel names and montage

def eeg_settings ():
    
    raw_setted = []
    
    for subj in range (len(subjects)):
        
        print(raw_conc_list[subj])  #control data
        eegbci.standardize(raw_conc_list[subj]) #Cambio i nomi dei canali
        montage = make_standard_montage('standard_1005') #Caricare il montaggio
        raw_conc_list[subj].set_montage(montage) #Setto il montaggio
        raw_setted.append(raw_conc_list[subj])
    
    return raw_setted

raw_setted = eeg_settings()
    
#%% Filtering raw data

#Returns list of raws filtered

def filtering ():
    
    #raw_filtered = []
    
    for subj in range (len(subjects)):
        
        raw_setted[subj].filter(1.0, 79.0, fir_design='firwin', skip_by_annotation='edge') #Filtro passabanda
        raw_setted[subj].notch_filter(freqs=60) #Faccio un filtro notch
                
        #todo: azzerare variabile
    return raw_setted

raw_filtered = filtering()

#%% Delete annotations

#returns raw objects with delete bad and useless annotations

def del_annotations():
    
    for subj in range (len(subjects)):
        ind = []
        
        for index, value in enumerate((raw_filtered[subj].annotations).description):            
            if value == "BAD boundary" or value == "EDGE boundary":
                ind.append(index)
        (raw_filtered[subj].annotations).delete(ind)
    
    return raw_filtered

del_annotations()


#%% Display and save plot psd of filtered raw data, real movement

def plot_pre_psd():
    
     for subj in range(len(subjects)): 
        
        #Plots psd of filtered raw data
        plot_pre = raw_filtered[subj].plot_psd(area_mode=None, show=True, average=False, fmin =1.0, fmax=80.0, dB=False, n_fft=160)
        
        #Creates plot's name
        psd_name = os.path.join(dir_pre_psd,'S'+ str(subjects[subj]) + '_real_pre.png')
        
        #Check if plot exists by checking its path
        if os.path.exists(psd_name):
            raise Exception('Warning! This plot already exists! :(' )           
        plot_pre.savefig(psd_name) #Saves plot
        
                
     return plot_pre

plot_pre_psd()

#Todos: Fix axes plot
#Todos: if fixed like before cumulates the same plot
#even del plot doesn't work
#Join pre psd and post psd plots 
        

#%%ICA

#Fitting Ica on raw_filtered data

def ica_function ():    
  
    # Ica
    #del plot_pre_psd
    for subj in range (len(subjects)):
        ica = ICA(n_components=64, random_state=42, method="fastica", max_iter=1000)    
        
        ica.fit(raw_filtered[subj])
        icas.append(ica)
    
    return icas 

icas = ica_function()
# Save icas?

#%% Ica plot properties
   
icas[0].plot_properties(raw_filtered[0], picks=[26], dB=False)

#%% Ica psd components instead of channels 

#Todos plot psd of ica components

#%% Correlational map



corrmap(icas, template=(0, 33), plot=True, threshold=0.70, label = 'artifact_')

#Todos
# Understand how to retrieve bad components found with corrmap

#%% Template for corrmap for real movement 


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
