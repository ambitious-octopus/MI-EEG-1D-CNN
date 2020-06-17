from pirate import Pirates
import os
import numpy as np
import mne

cwd = os.getcwd()
preprocessing = os.path.join(cwd, "preprocessing")
dir_imagined = os.path.join(preprocessing, "imagined")
#Setup Folders
dir_psd_icas, dir_dis, dir_psd, dir_pre_psd, dir_post_psd, dir_icas, dir_report, dir_templates, dir_psd_topo_map = Pirates.setup_folders(dir_imagined)
temp = [1]
chort = np.arange(2, 110).tolist()
sub = temp + chort
#Loading data
runs = Pirates.load_data(sub, [4, 8, 12])
#Concatenate runs
raws = Pirates.concatenate_runs(runs)
#Standardize
raws_set = Pirates.eeg_settings(raws)
#Filtering 
raws_filtered = Pirates.filtering(raws_set)
#Deleting annotations
raws_clean = Pirates.del_annotations(raws_filtered)
#Plot pre psd
#Pirates.plot_pre_psd(raws_clean, dir_pre_psd, overwrite=True)
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
#corr = Pirates.corr_map(icas, 0, comp_template, dir_templates, "artifact", threshold=0.80)
#Pirates.select_components(icas, raws_clean, "artifact")
#Pirates.save_exclusion_file(icas, "new")

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

#Pirates.psd_topo_map(icas, raws_clean, "artifact", "models")
#Pirates.plot_post_psd(reco_raws, dir_post_psd, overwrite=True)
#Pirates.discrepancy(raws_clean, reco_raws, dir_dis)
#Pirates.create_report_psd(dir_pre_psd, dir_post_psd, dir_dis, dir_report)




#reco_raws[0].save(os.path.join(cwd, reco_raws[0].__repr__()[10:14] + "_raw_sss.fif"), overwrite=True)
#raw = mne.io.read_raw_fif(os.path.join(cwd, reco_raws[0].__repr__()[10:14] + "_raw_sss.fif"))


#%% Pre and post peaks in freq domain

#1. Filter data between 8 and 13 Hz
#2. Epochs data
#3. Divide by events
#4. For right events -> 
    # - get peaks in baseline
    # - Average baseline peaks
    # - get peaks in task
    # - Average task peaks
# 5. Same for left events
# 6. Average of 8-13 Hz values?


def freq_analysis():
    
    


