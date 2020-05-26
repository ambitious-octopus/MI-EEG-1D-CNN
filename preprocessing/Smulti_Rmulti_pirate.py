from pirate import Pirates
import os
import matplotlib.pyplot as plt

cwd = os.getcwd()
preprocessing = os.path.join(cwd,"preprocessing")
imagined = os.path.join(preprocessing, "imagined")

dir_psd, dir_pre_psd, dir_post_psd, dir_icas, dir_report, dir_templates = Pirates.setup_folders(imagined) #setuppo le cartelle

#Ricordarsi di far passare il template come prima
runs = Pirates.load_data([65, 1, 2, 3, 4, 5, 6, 7], [4, 8, 12]) #carico i dati e croppo
#todo: whitening
raws = Pirates.concatenate_runs(runs) #Concateno le runs
raws_set = Pirates.eeg_settings(raws) #Standardizzo nomi ecc
raws_filtered = Pirates.filtering(raws_set) #Filtro
raws_clean = Pirates.del_annotations(raws_filtered) #Elimino annotazioni
Pirates.plot_pre_psd(raws_clean, dir_pre_psd,overwrite=True)
icas = Pirates.ica_function(raws_clean,dir_icas,save=False) #Applico una ica
comp_temp = [0,3,9,11,14,15,19,26,29,34,44,47,49,51,53,55,56,58,59,63,60,61]
Pirates.corr_map(icas, 0, comp_temp,dir_templates,threshold=0.85,label="artifact")
reco_raws = Pirates.reconstruct_raws(icas,raws_clean,"artifact")
Pirates.plot_post_psd(reco_raws,dir_post_psd,overwrite=True)
Pirates.create_report_psd(dir_pre_psd,dir_post_psd,dir_report)


# todo: template matching
# Per creare il template(icas, ica_template, comp_template) salvare le immagini.
# todo: L'idea è di insseririe all'interno della pipeline un check per il template


#Dowbnload data
import numpy as np
a = np.arange(1,110).tolist()
Pirates.load_data(a, [4, 8, 12])
