from data_processing.general_processor import Utils
import numpy as np
import os
exclude = [] #[38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1, 110) if n not in exclude]
runs = [4, 6, 8, 10, 12, 14]
channels = ["FC3", "FC4"]
data_path = "D:\\datasets\\eegbci"
save_path = "D:\\datasets\\eeg_dataset\\FC3_FC4_no_filter"

for sub in subjects:
    x, y = Utils.epoch(Utils.select_channels
                       (Utils.eeg_settings(Utils.del_annotations(Utils.concatenate_runs(
        Utils.load_data(subjects=[sub], runs=runs, data_path=data_path)))), channels), exclude_base=False)
    np.save(os.path.join(save_path, "x_FC3_FC4_sub_" + str(sub)), x, allow_pickle=True)
    np.save(os.path.join(save_path, "y_FC3_FC4_sub_" + str(sub)), y, allow_pickle=True)

#%%
from data_processing.general_processor import Utils
import numpy as np
import os

channels = [["FC1", "FC2"],
            ["FC3", "FC4"],
            ["FC5", "FC8"],
            ["C5", "C6"],
            ["C3", "C4"],
            ["C1", "C2"],
            ["CP1", "CP2"],
            ["CP3", "CP4"],
            ["CP5", "CP6"]]

exclude = [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1, 110) if n not in exclude]
runs = [4, 6, 8, 10, 12, 14]
data_path = "D:\\datasets\\eegbci"
for couple in channels:
    base_path = "D:\\datasets\\eeg_dataset\\n_ch_base"
    save_path = os.path.join(base_path, couple[0] + couple[1])
    os.mkdir(save_path)
    for sub in subjects:
        x, y = Utils.epoch(Utils.select_channels
            (Utils.eeg_settings(Utils.del_annotations(Utils.concatenate_runs(
            Utils.load_data(subjects=[sub], runs=runs, data_path=data_path)))), couple),
            exclude_base=False)
        np.save(os.path.join(save_path, "x_sub_" + str(sub)), x, allow_pickle=True)
        np.save(os.path.join(save_path, "y_sub_" + str(sub)), y, allow_pickle=True)