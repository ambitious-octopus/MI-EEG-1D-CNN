import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import axes, image
from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations, make_ad_hoc_cov,compute_raw_covariance
from mne.cov import compute_whitener
from mne.simulation import add_noise
from mne.io import concatenate_raws, read_raw_edf,RawArray
from mne.channels import make_standard_montage
from mne.preprocessing import ICA, corrmap, read_ica
from scipy import stats
import os
#%% Creating directories for saving plots psd

#returns folder directories for psd saving

def folders ():
    
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
        
    
    dir_icas = os.path.join(dir_preprocessing,'icas')
    if os.path.isdir(dir_icas):
        print('Icas directory already exists')
    else:
        print("Path not found, creating icas directory...")
        os.mkdir(dir_icas)
        
    return dir_preprocessing,dir_psd_real, dir_pre_psd,dir_post_psd,dir_icas

# returning folder directories

folders()
        
dir_preprocessing,dir_psd_real, dir_pre_psd,dir_post_psd, dir_icas = folders()

#dir movimento immaginato 


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

raw_loaded = load_data()

m = raw_loaded[0][0]._data

m1 = raw_loaded[0][1]._data

m2 = raw_loaded[0][2]._data

varm = np.var(m)
varm1 = np.var(m1)
varm2 = np.var(m2)

#list of lists of Raw objects of runs to be concatenated

#%% Whitening data
#Non ho ben capito come fare

#a = raw_loaded[0][0]._data
#stand_a = stats.zscore(a)

#stand =  stats.zscore(raw_loaded[0][0]._data)

#raw_loaded[0][0]._data = stand


def whitening ():
    
    ls_z_tot = [] #lista di liste di run 
   
    
    for list in raw_loaded:
        
        ls_z = [] #lista di run di uno stesso soggetto 
           
        for run in range (len(list)):
            
                        
            raw_data = list[run]._data #prendo l'array dei dati raw
            
                                  
            #print(list[run])
            
            stand =  stats.zscore(raw_data) #standardizzo l'array dei dati raw
            
            list[run]._data = stand  #sostituisco n_epoch dati del raw originale con n_epoch dati standardizzati
           
            
            ls_z.append(list[run])           
            
              
        ls_z_tot.append(ls_z)
        

    return ls_z_tot #, ls_info_tot

raw_whitened = whitening() #list of raw data whitened
#raw_zetas, raws_infos = whitening() #list of raw data whitened      
#capire se è whitening



#%% Concatenate raws 

# returns list of concatenated raw
# e.g. for subject = [2,45]
#returns list = [Raw_S2(concatenated), Raw_S45(concatenated)]


#data = raw_loaded or raw_loaded_whitened, choose wheter you want whitened data

def concatenation(data):
    
    raw_conc_list = []
    
    for subj in range (len(subjects)):
          #print(subj)
          #print(rawdata_loaded[subj])
          #print(data[subj])
          raw_conc = concatenate_raws(data[subj])
          
          raw_conc_list.append(raw_conc)
        
    return raw_conc_list

raw_conc_list = concatenation(data = raw_whitened)
#return a list of raw data concatenated per subject

#%% Change channel names, set montage

#Returns list of raw setted with channel names and montage

def eeg_settings ():
    
    raw_setted = []
    
    for subj in range (len(subjects)):
        
        print(raw_conc_list[subj])  #control data
        eegbci.standardize(raw_conc_list[subj]) #Cambio n_epoch nomi dei canali
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

raws_filt = filtering()

#%% Delete annotations

# raw objects with delete bad and useless annotations

def del_annotations():
    
    for subj in range (len(subjects)):
        ind = []
        
        for index, value in enumerate((raws_filt[subj].annotations).description):            
            if value == "BAD boundary" or value == "EDGE boundary":
                ind.append(index)
        (raws_filt[subj].annotations).delete(ind)
    
    return raws_filt

del_annotations()


#%% Display and save plot psd of filtered raw data, real movement

#Choose wheter to overwrite existing plots

def plot_pre_psd(overwrite = True):
    
     for subj in range(len(subjects)): 
         
        plot_pre = plt.figure() #I create an empty figure so I can apply clear next line
           
        plot_pre.clear() #Clearing the plot each iteration I do not obtain overlapping plots
        
        #Plots psd of filtered raw data
        plot_pre = raws_filt[subj].plot_psd(area_mode=None, show=True, average=False, ax=plt.axes(ylim=(0,60)),fmin =1.0, fmax=80.0, dB=False, n_fft=160)
        
        #Creates plot's name
        psd_name = os.path.join(dir_pre_psd,'S'+ str(subjects[subj]) + '_real_pre.png')
        
        #Check if plot exists by checking its path
        if os.path.exists(psd_name) and overwrite == False: #stops only if overwrite == False
            raise Exception('Warning! This plot already exists! :(' ) 
            
        elif os.path.exists(psd_name) and overwrite == True:
            os.remove(psd_name) #removes existing saved plot
            
            
            if os.path.exists(psd_name):
                raise Exception('You did not remove existing plot!') #check if plot being deleted, not always os.remove works
            else:
                
                plot_pre.savefig(psd_name)
              
            
        else:
            plot_pre.savefig(psd_name)
            
                                                                         
     return None
plot_pre_psd(overwrite = True)


#Join pre psd and post psd plots 
        

#%%ICA

#Fitting Ica on raw_filtered dataicas
# Save = True to save icas 
# Overwrite = True to overwrite existing saved icas

def ica_function (save = False, overwrite = False):    
  
       
    icas_names = [] #Creating here empty list to clean it before using it, return ica saved paths for future loading
    icas = [] #return icas 
    
    for subj in range (len(subjects)):
        
        ica = ICA(n_components=64, random_state=42, method="fastica", max_iter=2000)    
        
        ica.fit(raws_filt[subj]) #fitting ica 
        
        if save == True:
            
            icas_name = os.path.join(dir_icas,'S'+ str(subjects[subj]) + '_ica.fif') #creating ica name path
            
            if os.path.exists(icas_name) and overwrite == False:
                raise Exception('Warning! This ica already exists! :( ' + str(icas_name)) #stops if you don't want to overwrite a saved ica
                
            elif os.path.exists(icas_name) and overwrite == True:
                
                os.remove(icas_name)  # to overwrite delets existing ica             
                                                            
                ica.save(icas_name)        
                icas_names.append(icas_name)
                print('Overwriting existing icas')
            
            else:
                ica.save(icas_name)        
                print('Creating ica files')
                icas_names.append(icas_name)
             
        icas.append(ica)
    
    return icas, icas_names 

icas,icas_names = ica_function(save = False, overwrite= False)

#icas[0]._pre_whiten(${1:raws_filt}, info, picks=64)
bm = icas[0].get_components()
#plot_psd(m)

#%%
b = icas[0].get_components()
raw = RawArray(b, raws_filt[0].info)
raw.plot_psd(area_mode=None, show=True, average=False,fmin =1.0, fmax=80.0, dB=False)
#%% Load saved icas from icas_names = paths
#BUG ICAS_NAMES not defined, you should run 

def loading_icas ():
    
    icas = []
    
    for subj in range (len(subjects)):
        
        icas_names = os.path.join(dir_icas,'S'+ str(subjects[subj]) + '_ica.fif')
        loaded_ica = read_ica(icas_names[subj])
        icas.append(loaded_ica)
    
    return icas

icas_load = loading_icas()
print(icas_load)


#%% Ica plot properties
   
icas[0].plot_properties(raws_filt[0], picks=[30,31,32,33,34,35,36,37,38,39,40], dB=False)
score_sources(self, inst, target=None, score_func='pearsonr', start=None, stop=None, l_freq=None, h_freq=None, reject_by_annotation=True, verbose=None)[source]
#%% Ica psd components instead of channels 

pre_whitener = np.empty([len(raws_filt[0]), 1])
pre_whitener_2 = np.std(raws_filt[0]
                        
                        )
 def _pre_whiten(self, data, info, picks):
        """Aux function."""
        has_pre_whitener = hasattr(self, 'pre_whitener_')
        if not has_pre_whitener and self.noise_cov is None:
            # use standardization as whitener
            # Scale (z-score) the data by channel type
            info = pick_info(info, picks)
            pre_whitener = np.empty([len(data), 1])
            for ch_type in _DATA_CH_TYPES_SPLIT + ('eog', "ref_meg"):
                if _contains_ch_type(info, ch_type):
                    if ch_type == 'seeg':
                        this_picks = pick_types(info, meg=False, seeg=True)
                    elif ch_type == 'ecog':
                        this_picks = pick_types(info, meg=False, ecog=True)
                    elif ch_type == 'eeg':
                        this_picks = pick_types(info, meg=False, eeg=True)
                    elif ch_type in ('mag', 'grad'):
                        this_picks = pick_types(info, meg=ch_type)
                    elif ch_type == 'eog':
                        this_picks = pick_types(info, meg=False, eog=True)
                    elif ch_type in ('hbo', 'hbr'):
                        this_picks = pick_types(info, meg=False, fnirs=ch_type)
                    elif ch_type == 'ref_meg':
                        this_picks = pick_types(info, meg=False, ref_meg=True)
                    else:
                        raise RuntimeError('Should not be reached.'
                                           'Unsupported channel {}'
                                           .format(ch_type))
                    pre_whitener[this_picks] = np.std(data[this_picks])
            data /= pre_whitener
        elif not has_pre_whitener and self.noise_cov is not None:
            pre_whitener, _ = compute_whitener(self.noise_cov, info, picks)
            assert data.shape[0] == pre_whitener.shape[1]
            data = np.dot(pre_whitener, data)
        elif has_pre_whitener and self.noise_cov is None:
            data /= self.pre_whitener_
            pre_whitener = self.pre_whitener_
        else:
            data = np.dot(self.pre_whitener_, data)
            pre_whitener = self.pre_whitener_

        return data, pre_whitener
#Todos plot psd of ica components

#%% Selection of components for template matching, real movement

#eog_inds, eog_scores = icas[0].find_bads_eog(raws[0], ch_name='Fpz')
icas_load[2].plot_properties(raws_filt[2], picks=[1], dB=False)
ica[0]._
#components displayed so you can past and copy them
#[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
#21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
#41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60
# 61,62,63]

exc_66 = [0,1,3,13,14,17,29,32,33,35,47,50,53]

#Here we report what kind of artifact
eyes = [0,1,3,14,33,35,47]
#ecg = []
#movement =[17,32,] 
#odd = [13]
#electrode =[53]


#%% Correlational map

#exc_template = 


corrmap(icas, template=(0, 33), plot=True, threshold=0.75, label = 'artifact Subject' )

for subj in range (len(subjects)):
    
    
    print(str(icas_load[subj].labels_ ))
    
    artifacts_dic = []
    
    #artifacts_dic.append(icas[subj].labels_.items())
    
    #artifacts_dic.append(icas[subj].labels_)
    print(artifacts_dic)
   # print(' Subject' + str(subjects[subj]) +' ' + str([icas[subj].labels_ for ica in icas]))
    
i = icas_load[1].labels_
#Todos
# Understand how to retrieve bad components found with corrmap

#%% Template for corrmap for real movement 

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

#%% Reconstruct raw files with excluded components
            
def reconstruct_raws ():
    
    reco_raws = []
    
    for index, ica in enumerate(icas):
        
        reco_raw = raws[index].copy()
        icas[index].exclude = icas[index].labels_["artifact_0"]
        icas[index].apply(reco_raw)
        reco_raws.append(reco_raw)
    
    return reco_raws

reco_raws = reconstruct_raws()

#%% Create plot post psd

def plot_post_psd(overwrite = True):
    
     for subj in range(len(subjects)): 
         
        plot_post = plt.figure() #I create an empty figure so I can apply clear next line
           
        plot_post.clear() #Clearing the plot each iteration I do not obtain overlapping plots
        
        #Plots psd of filtered raw data
        plot_post = reco_raws[subj].plot_psd(area_mode=None, show=True, average=False, ax=plt.axes(ylim=(0,60)),fmin =1.0, fmax=80.0, dB=False, n_fft=160)
        
        #Creates plot's name
        psd_name = os.path.join(dir_post_psd,'S'+ str(subjects[subj]) + '_real_post.png')
        
        #Check if plot exists by checking its path
        if os.path.exists(psd_name) and overwrite == False: #stops only if overwrite == False
            raise Exception('Warning! This plot already exists! :(' ) 
            
        elif os.path.exists(psd_name) and overwrite == True:
            os.remove(psd_name) #removes existing saved plot
            
            
            if os.path.exists(psd_name):
                raise Exception('You did not remove existing plot!') #check if plot being deleted, not always os.remove works
            else:
                
                plot_post.savefig(psd_name)
              
            
        else:
            plot_post.savefig(psd_name)
            
                                                                         
     return None
 
plot_post_psd(overwrite = True)
#%%  Tests to join psd

plot_post = raws_filt[0].plot_psd(area_mode=None, show=True, average=False, ax=plt.axes(ylim=(0,60)),fmin =1.0, fmax=80.0, dB=False, n_fft=160)

plot_pre = raws_filt[1].plot_psd(area_mode=None, show=True, average=False, ax=plt.axes(ylim=(0,60)),fmin =1.0, fmax=80.0, dB=False, n_fft=160)



path = "C:\\Users\\admin\\Desktop\\eeGNN\\preprocessing\\psd_real\\pre_psd\\S23_real_pre.png"
path2 = "C:\\Users\\admin\\Desktop\\eeGNN\\preprocessing\\psd_real\\pre_psd\\S66_real_pre.png"

b = image.FigureImage(plot_post)
c = image.FigureImage(plot_pre)
a = [b,c]

list_im = [path, path2]
new_im = Image.new('RGB', (444,95)) #creates a new empty image, RGB mode, and size 444 by 95

for elem in list_im:
    im=Image.open(elem)
    print(elem)
    #im.show()

    new_im.paste(im, (i,0))
new_im.save('test2.jpg')
    
#%% Other tests
    
images = [Image.open(x) for x in [path, path2]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save('test.jpg')
