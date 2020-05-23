from pirate import Pirate
import os

raws = Pirate.load_data([1, 2], [3, 4])
raw = Pirate.concatenate_runs(raws)
raw_no = Pirate.del_annotations(raw)
dir_preprocessing, dir_psd_real, dir_pre_psd, dir_post_psd, dir_icas = Pirate.create_folders_psd()

raw_no = Pirate.eeg_settings(raw_no)


raw_fill = Pirate.filtering(raw_no)

icas, icas_names = Pirate.ica_function(raw_fill,dir_icas)