from pirate import Pirates
import os
import numpy as np

cwd = os.getcwd()
preprocessing = os.path.join(cwd, "preprocessing")
imagined = os.path.join(preprocessing, "imagined")

dir_dis, dir_psd, dir_pre_psd, dir_post_psd, dir_icas, dir_report, dir_templates = Pirates.setup_folders(imagined)  # setuppo le cartelle

chort = np.arange(2, 35).tolist()
temp = [1]
sub = temp + chort

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