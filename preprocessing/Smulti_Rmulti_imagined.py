from pirate import Pirates
import os
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
preprocessing = os.path.join(cwd, "preprocessing")
dir_imagined = os.path.join(preprocessing, "imagined")

dir_psd_icas, dir_dis, dir_psd, dir_pre_psd, dir_post_psd, dir_icas, dir_report, dir_templates, dir_psd_topo_map = Pirates.setup_folders(dir_imagined)  # setuppo le cartelle

temp = [1]
chort = np.arange(2, 110).tolist()
sub = temp + chort

# Ricordarsi di far passare il template come prima
runs = Pirates.load_data(sub, [4, 8, 12])  # carico i dati e croppo
raws = Pirates.concatenate_runs(runs)  # Concateno le runs
raws_set = Pirates.eeg_settings(raws)  # Standardizzo nomi ecc
raws_filtered = Pirates.filtering(raws_set)  # Filtro
raws_clean = Pirates.del_annotations(raws_filtered)  # Elimino annotazioni
Pirates.plot_pre_psd(raws_clean, dir_pre_psd, overwrite=True)
#icas = Pirates.ica_function(raws_clean, dir_icas, save=True, overwrite=True)  # Applico una ica
icas = Pirates.load_saved_icas(dir_icas, 1, 109)
#list_psd = Pirates.get_ica_psd(raws_clean, icas, dir_psd_icas)
path_exclusions = os.path.join(dir_imagined, "_exclusion.npy")
Pirates.load_exclusion(icas, path_exclusions)

#Selecting components
eye = [5, 0]
mov = [33]
nb = [21, 40, 44, 48]
comp_template = eye + mov + nb

Pirates.select_components(icas, raws_clean, "artifact")
#corr = Pirates.corr_map(icas, 0, comp_template, dir_templates, "artifact", threshold=0.80)
reco_raws = Pirates.reconstruct_raws(icas, raws_clean, "artifact")
#Interpolation
Pirates.interpolate(reco_raws[3], "FT8") #Remember that the real subj is x + 1
Pirates.interpolate(reco_raws[8], "AF7")
Pirates.interpolate(reco_raws[8], "AF8")
Pirates.interpolate(reco_raws[20], "C5")
Pirates.interpolate(reco_raws[47], "C1")
Pirates.interpolate(reco_raws[78], "P1")
Pirates.interpolate(reco_raws[84], "O1")
Pirates.interpolate(reco_raws[94], "AF4")

Pirates.psd_topo_map(icas, raws_clean, "artifact", "models")
Pirates.plot_post_psd(reco_raws, dir_post_psd, overwrite=True)
Pirates.discrepancy(raws_clean, reco_raws, dir_dis)
Pirates.create_report_psd(dir_pre_psd, dir_post_psd, dir_dis, dir_report)
