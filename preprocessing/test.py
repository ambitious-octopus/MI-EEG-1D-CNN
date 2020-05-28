from pirate import Pirates
import os
import numpy as np

cwd = os.getcwd()
preprocessing = os.path.join(cwd, "preprocessing")
example = os.path.join(preprocessing, "example")

dir_dis, dir_psd, dir_pre_psd, dir_post_psd, dir_icas, dir_report, dir_templates = Pirates.setup_folders(example)  # setuppo le cartelle

chort = np.arange(2, 4).tolist()
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

eye = [5,0, 32]
other = [1,2]
mov = [33, 42]
nb = [21, 40, 44, 48]
comp_template = eye + other + mov + nb
not_found = [23, 46, 62]

Pirates.corr_map(icas, 0, comp_template, dir_templates, threshold=0.80, label="artifact")
reco_raws = Pirates.reconstruct_raws(icas, raws_clean, "artifact")
Pirates.plot_post_psd(reco_raws, dir_post_psd, overwrite=True)

Pirates.discrepancy(raws_clean,reco_raws,dir_dis)
Pirates.create_report_psd(dir_pre_psd, dir_post_psd, dir_dis, dir_report)

from mne.preprocessing import ICA

ica = ICA()


a = ["Francesco","Laura","Sara"]
b = [25, 28, 27]

for nome, eta in zip(a,b):
    print(nome,eta)
    print()




