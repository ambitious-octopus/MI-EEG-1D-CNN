from pirate import Pirates
import os
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
preprocessing = os.path.join(cwd, "preprocessing")
example = os.path.join(preprocessing, "example")

dir_dis, dir_psd, dir_pre_psd, dir_post_psd, dir_icas, dir_report, dir_templates, dir_psd_topo_map = Pirates.setup_folders(example)  # setuppo le cartelle

chort = np.arange(2, 4).tolist()
temp = [1]
sub = temp + chort

# Ricordarsi di far passare il template come prima
runs = Pirates.load_data(sub, [4, 8, 12])  # carico i dati e croppo
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


corr = Pirates.corr_map(icas, 0, comp_template, dir_templates, "arti", threshold=0.85)
Pirates.psd_topo_map(icas, raws_clean, "arti", dir_psd_topo_map)

#reco_raws = Pirates.reconstruct_raws(icas, raws_clean, "artifact")
#Pirates.plot_post_psd(reco_raws, dir_post_psd, overwrite=True)

#Pirates.discrepancy(raws_clean,reco_raws,dir_dis)
#Pirates.create_report_psd(dir_pre_psd, dir_post_psd, dir_dis, dir_report)

from PIL import Image, ImageDraw, ImageFont
from mne.time_frequency import psd_multitaper, psd_welch




#%%

sources = icas[1].get_sources(raws_clean[1])
data = sources.get_data(picks=34)
psds, freqs = plt.psd(data[0], Fs=160)
plt.close("all")
plt.plot(freqs,psds)

log_psds = np.sqrt(freqs)
plt.close("all")
#dio_aiutaci.reverse()
a = psds**2/np.sqrt(freqs)
plt.plot(freqs,a)



sources = icas[0].get_sources(raws_clean[0])
data = sources.get_data(picks=34)
p = []
f = []
for comp in n_comp:
    data = sources.get_data(picks=comp)
    psds, freqs = plt.psd(data[0], Fs=160, NFFT=160, noverlap=0.6)
    p.append(psds)
    f.append(freqs)
plt.close("all")

for a,b in zip(f,p):
    #plt.ylim(0, 0.3)
    plt.plot(a,b)



















