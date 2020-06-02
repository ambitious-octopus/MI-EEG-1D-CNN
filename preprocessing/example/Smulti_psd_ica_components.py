from pirate import Pirates
import os
import numpy as np
from mne import find_events, EpochsArray
from matplotlib import pyplot as plt
cwd = os.getcwd()
preprocessing = os.path.join(cwd, "preprocessing")
imagined = os.path.join(preprocessing, "imagined")

#dir_dis, dir_psd, dir_pre_psd, dir_post_psd, dir_icas, dir_report, dir_templates = Pirates.setup_folders(imagined)  # setuppo le cartelle

chort = np.arange(2, 35).tolist()
temp = [1]
#sub = temp + chort
sub = [0]



# Ricordarsi di far passare il template come prima
runs = Pirates.load_data(sub, [3, 7, 11])  # carico i dati e croppo
# todo: whitening
raws = Pirates.concatenate_runs(runs)  # Concateno le runs
raws_set = Pirates.eeg_settings(raws)  # Standardizzo nomi ecc
raws_filtered = Pirates.filtering(raws_set)  # Filtro
raws_clean = Pirates.del_annotations(raws_filtered)  # Elimino annotazioni
Pirates.plot_pre_psd(raws_clean, dir_pre_psd, overwrite=True)
icas = Pirates.ica_function(raws_clean, dir_icas, save=True, overwrite = True)  # Applico una ica
icas = Pirates.load_saved_icas(dir_icas)

#raw_copy = raws_clean[0].copy()

#icas[0].plot_properties(raws_clean[0], picks=[2], dB=False)

#icas[0].plot_sources(raws_clean[0])

#%%
from mne.epochs import make_fixed_length_epochs # importo mne.epochs
from mne import create_info

inst = make_fixed_length_epochs(raws_clean[0], duration=2., verbose=False, preload=True)
#creo le epoche dal raw

data = icas[0].get_sources(inst).get_data() #prendo l'array delle componenti
#b = icas[0].get_sources(raws_clean[0])

ica_names = icas[0]._ica_names #mi prendo la lista di nomi di icas da usare al posto dei canali
info2 = raws_clean[0].info #mi copio le info dai dati raw, da questi mi servirà solo ['chs']


'''ALTERNATIVA con info3 creare una nuova info structure'''

#Bisogna creare un info object con il nome delle ica al posto dei canali
#Nell'oggetto info il nome dei canali deve essere modificato con le ica sia in 'ch_names'  
#sia in chs -> chs è una lista con ad ogni posto un dizionario per canale chs = [dict, dict ecc]
# chs[0] ha questa forma -> in cui ch_name va modificato

'''{'cal': 16184.0,
 'logno': 1,
 'scanno': 1,
 'range': 16184.0,
 'unit_mul': 0.0,
 'ch_name': 'FC5',
 'unit': 107 (FIFF_UNIT_V),
 'coord_frame': 4 (FIFFV_COORD_HEAD),
 'coil_type': 1 (FIFFV_COIL_EEG),
 'kind': 2 (FIFFV_EEG_CH),
 'loc': array([-0.07563298,  0.04304355,  0.06791385,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ])} 
'''

# per accedere a chs però non si accede tramite info.chs
#bisogna fare i procedimenti di cui sotto

chs = info2['chs'] #ho esportato la lista chs nella variabile chs
# acq_pars = info2['acq_pars']
# acq_stim = info2['acq_stim']
# custom_ref_applied = info2['custom_ref_applied']
# dev_head_t = info2['dev_head_t']
# dig =info2['dig']
# highpass = info2['highpass']
# lowpass = info2['lowpass']
# nchan = info2['nchan']


info3 = create_info(sfreq = 160, ch_names = ica_names, ch_types ='eeg')

#info3['chs'] = chs #aggiungo alla nuova info structure questo parametro con i canali, ci vuole altrimenti da errore

#altri parametri che si potrebbero aggiungere volendo
#info3['acq_pars']=acq_pars
#info3['acq_stim'] = acq_stim
# info3['custom_ref_applied'] =custom_ref_applied
# info3['dev_head_t']=dev_head_t
# info3['dig']=dig
# info3['highpass']=highpass
# info3['lowpass']=lowpass
# info3['nchan']= nchan


#Sostituisco ad ogni nome canale in chs il nome delle ica

# for index, dic in enumerate(chs):
#     print(ch)
#     chs[index]['ch_name'] = ica_names[index] #sostituisco al nome dei canali quello delle ica
    
# info3['chs'] = chs



comp_epocate = EpochsArray(data,info3) #creo un EpochsArray con le epoche calcolate prima(data) e
#la info structure creata prima

comp_epocate.plot_psd(dB= False,area_mode=None, average=False,fmin=1.0,fmax=80.0, picks = 'all', spatial_colors = True)
#questo plot psd fa parte di EpochsArray

icas[0].plot_properties(raws_clean[0],picks = [16] ,dB= False)


#%% I estimate psd to have psd and freqs array

from mne.time_frequency import psd_multitaper,psd_welch

psdw, freqw = psd_welch(comp_epocate, n_fft = 160, n_overlap = 60)
plt.plot(freqw, psdw[0][0])

psd, freqs = psd_multitaper(comp_epocate)

plt.plot(freqs,psd[0][0])


#%%
#Ignorare da qui in poi

sources = icas[0].get_sources(raws_clean[0])
data = sources.get_data()

data_m = np.mean(data)

ps, fr, lin = plt.magnitude_spectrum(data[34], Fs=160)


line2dx = lin.get_xdata()
line2dy = lin.get_ydata()
plt.plot(line2dx,line2dy)



plt.plot(lin)
plt.show()
plt.close("all")
icas[0].plot_properties(raws_clean[0],picks = 34, dB= False)

plt.close("all")
plt.plot(freqs,psds)

    
events = find_events(raws_clean[0])



sources =icas[0].get_sources(raws_clean[0])
print(sources._data)
#print(raws_clean[0]._data)
sources.plot()
sources.plot_psd(area_mode=None, show=False, average=False, ax=plt.axes(ylim=(0, 30)), fmin=1.0,fmax=80.0, dB=False, n_fft=160)

eye = [5,0, 32]
other = [1,2]
mov = [33, 42]
nb = [21, 40, 44, 48]
comp_template = eye + other + mov + nb
not_found = [23, 46, 62]

Pirates.corr_map(icas, 0, comp_template, dir_templates, threshold=0.80, label="artifact")
reco_raws = Pirates.reconstruct_raws(icas, raws_clean, "artifact")
Pirates.plot_post_psd(reco_raws, dir_post_psd, overwrite=True)
Pirates.create_report_psd(dir_pre_psd, dir_post_psd, dir_report)

a = 3/6
