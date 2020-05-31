from pirate import Pirates
import os
import numpy as np

cwd = os.getcwd()
preprocessing = os.path.join(cwd, "preprocessing")
imagined = os.path.join(preprocessing, "imagined")

dir_dis, dir_psd, dir_pre_psd, dir_post_psd, dir_icas, dir_report, dir_templates = Pirates.setup_folders(imagined)  # setuppo le cartelle

#chort = np.arange(34).tolist()
temp = [34]
sub = temp

# Ricordarsi di far passare il template come prima
runs = Pirates.load_data(sub, [4, 8, 12])  # carico i dati e croppo
# todo: whitening
raws = Pirates.concatenate_runs(runs)  # Concateno le runs
raws_set = Pirates.eeg_settings(raws)  # Standardizzo nomi ecc
raws_filtered = Pirates.filtering(raws_set)  # Filtro
raws_clean = Pirates.del_annotations(raws_filtered)  # Elimino annotazioni
Pirates.plot_pre_psd(raws_clean, dir_pre_psd, overwrite=True)
#icas = Pirates.ica_function(raws_clean, dir_icas, save=True)  # Applico una ica
icas = Pirates.load_saved_icas(dir_icas)

icas[0].plot_properties(raws_clean[0], picks=[2], dB=False)

icas[0].plot_sources(raws_clean[0])

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

icas[0].plot_properties(raws_clean[0], picks=[0])


import matplotlib.pyplot as plt

sources = icas[0].get_sources(raws_clean[0])

data_sources = sources.get_data()

import numpy as np

freq = np.arange(1,81)
amp, fre =plt.psd(data_sources[0], Fs=160.0, scale_by_freq=False)

plt.close("all")

amp_s = amp


from mne.time_frequency import psd_multitaper

