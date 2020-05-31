from pirate import Pirates
import os
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
preprocessing = os.path.join(cwd, "preprocessing")
example = os.path.join(preprocessing, "example")

dir_dis, dir_psd, dir_pre_psd, dir_post_psd, dir_icas, dir_report, dir_templates = Pirates.setup_folders(example)  # setuppo le cartelle

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
#reco_raws = Pirates.reconstruct_raws(icas, raws_clean, "artifact")
#Pirates.plot_post_psd(reco_raws, dir_post_psd, overwrite=True)

#Pirates.discrepancy(raws_clean,reco_raws,dir_dis)
#Pirates.create_report_psd(dir_pre_psd, dir_post_psd, dir_dis, dir_report)


import matplotlib.pyplot as plt
from PIL import Image
from mne.time_frequency import psd_multitaper, psd_welch

icas = icas
label = "arti"
dir_topo_psd = os.getcwd()
list_sub = sub
raws = raws_clean

#Qui defnire
subjs = list()
for ica,subj in zip(icas[1:],raws[1:]):
    img_paths = list()
    lab = ica.labels_[label]
    for n in lab:
        ax1 = ica.plot_components(picks=n, title=subj.__repr__()[10:14])
        path_comp = os.path.join(dir_topo_psd, "TOPO"  + subj.__repr__()[10:14] + "C" + str(n) + ".png")
        ax1.savefig(path_comp)
        plt.close(ax1)
        sources = ica.get_sources(subj)
        psds, freqs = psd_welch(sources, picks=[n])
        psds = 10 * np.log10(psds)
        psds_mean = psds.mean(0)
        plt.figure(figsize=(2.5, 2))
        plt.plot(freqs, psds_mean)
        path_psd = os.path.join(dir_topo_psd, "PSD" + subj.__repr__()[10:14] + "C" + str(n) + ".png")
        plt.savefig(path_psd)
        plt.close("all")
        psd = Image.open(path_comp)
        topo = Image.open(path_psd)
        imgs = [psd, topo]
        width_0, height_0 = imgs[0].size
        width_1, height_1 = imgs[1].size
        real_width = width_0 + width_1
        real_height = height_1 + 15
        new_im = Image.new('RGBA', (real_width, real_height))
        x_offset = 0
        for im in imgs:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        img_path = os.path.join(dir_topo_psd, subj.__repr__()[10:14] + "C" + str(n) + ".png")
        new_im.save(img_path)
        img_paths.append(img_path)
        os.remove(path_psd)
        os.remove(path_comp)
        plt.close('all')
    subjs.append(img_paths)
        














