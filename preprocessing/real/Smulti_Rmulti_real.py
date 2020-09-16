from pirate import Pirates
import os
import numpy as np
import mne

cwd = os.getcwd()
preprocessing = os.path.join(cwd, "preprocessing")
dir_real = os.path.join(preprocessing, "real")
#Setup Folders
dir_psd_icas, dir_dis, dir_psd, dir_pre_psd, dir_post_psd, dir_icas, dir_report, dir_templates, dir_psd_topo_map = Pirates.setup_folders(dir_real)
temp = [1]
chort = np.arange(2, 110).tolist()
sub = temp + chort
#Loading data
runs = Pirates.load_data(sub, [3, 7, 11])
#Concatenate runs
raws = Pirates.concatenate_runs(runs)
#Standardize
raws_set = Pirates.eeg_settings(raws)
#Filtering 
raws_filtered = Pirates.filtering(raws_set)
#Deleting annotations
raws_clean = Pirates.del_annotations(raws_filtered)
#Plot pre psd
Pirates.plot_pre_psd(raws_clean, dir_pre_psd, overwrite=True)
icas = Pirates.ica_function(raws_clean, dir_icas, save=True, overwrite=True)  # Applico una ica
icas = Pirates.load_saved_icas(dir_icas, 1, 109)
list_psd = Pirates.get_ica_psd(raws_clean, icas, dir_psd_icas)


# path_exclusions = os.path.join(dir_imagined, "_exclusion.npy")
# Pirates.load_exclusion(icas, path_exclusions)
#Selecting components
eye = [0,36,35,18,6]
mov = [4]
nb = [1, 57, 54, 53, 49, 48, 47,44,40,39,28,20]
comp_template = eye + mov + nb
corr = Pirates.corr_map(icas, 0, comp_template, dir_templates, "artifact", threshold=0.80)
Pirates.psd_topo_map(icas, raws_clean, "artifact", dir_psd_topo_map)

Pirates.select_components(icas, raws_clean, "artifact")
Pirates.save_exclusion_file(icas, "real_exclusion")

exc = np.load("real_exclusion.npy", allow_pickle=True)


reco_raws = Pirates.reconstruct_raws(icas, raws_clean, "artifact")


#reco_raws[108].plot_psd(area_mode=None, show=False, average=False, ax=plt.axes(ylim=(0, 30)), fmin=1.0, fmax=80.0, dB=False, n_fft=160)
#Interpolation
Pirates.interpolate(reco_raws[3], "FT8")  #Remember that the real subj is x + 1
Pirates.interpolate(reco_raws[6], "T10")
Pirates.interpolate(reco_raws[8], "AF7")
Pirates.interpolate(reco_raws[8], "AF8")
Pirates.interpolate(reco_raws[12], "T10")
Pirates.interpolate(reco_raws[20], "C5")
Pirates.interpolate(reco_raws[42], "T9")
Pirates.interpolate(reco_raws[47], "C1")
Pirates.interpolate(reco_raws[78], "P1")
Pirates.interpolate(reco_raws[84], "O1")
Pirates.interpolate(reco_raws[94], "AF4")
Pirates.interpolate(reco_raws[24], "C6")
Pirates.interpolate(reco_raws[52], "P7")
Pirates.interpolate(reco_raws[52], "T10")
Pirates.interpolate(reco_raws[56], "FT7")
Pirates.interpolate(reco_raws[78], "Fp1")
Pirates.interpolate(reco_raws[82], "AF4")
Pirates.interpolate(reco_raws[96], "AF7")
Pirates.interpolate(reco_raws[98], "T10")
Pirates.interpolate(reco_raws[99], "T10")
Pirates.interpolate(reco_raws[99], "FC5")
Pirates.interpolate(reco_raws[102], "T10")
Pirates.interpolate(reco_raws[108], "F6")


Pirates.plot_post_psd(reco_raws, dir_post_psd, overwrite=True)
Pirates.discrepancy(raws_clean, reco_raws, dir_dis)
Pirates.create_report_psd(dir_pre_psd, dir_post_psd, dir_dis, dir_report)
dir_to_save = "C:\\Users\\franc_pyl533c\\OneDrive\\Documenti\\real_reco_raws"
Pirates.save_fif(reco_raws, dir_to_save)


