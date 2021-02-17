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