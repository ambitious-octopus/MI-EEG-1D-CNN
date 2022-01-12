"""
A 1D CNN for high accuracy classiﬁcation in motor imagery EEG-based brain-computer interface
Journal of Neural Engineering (https://doi.org/10.1088/1741-2552/ac4430)
Copyright (C) 2022  Francesco Mattioli, Gianluca Baldassarre, Camillo Porcaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
sys.path.insert(0,'/workspace/')
from data_processing.general_processor import Utils
import numpy as np
import os

channels = [["FC1", "FC2"],
            ["FC3", "FC4"],
            ["FC5", "FC6"],
            ["C5", "C6"],
            ["C3", "C4"],
            ["C1", "C2"],
            ["CP1", "CP2"],
            ["CP3", "CP4"],
            ["CP5", "CP6"]]

exclude = [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1, 110) if n not in exclude]
runs = [4, 6, 8, 10, 12, 14]
data_path = "/dataset/original"
for couple in channels:
    base_path = "/dataset/paper"
    save_path = os.path.join(base_path, couple[0] + couple[1])
    os.mkdir(save_path)
    for sub in subjects:
        x, y = Utils.epoch(Utils.select_channels
            (Utils.eeg_settings(Utils.del_annotations(Utils.concatenate_runs(
            Utils.load_data(subjects=[sub], runs=runs, data_path=data_path)))), couple),
            exclude_base=False)

        np.save(os.path.join(save_path, "x_sub_" + str(sub)), x, allow_pickle=True)
        np.save(os.path.join(save_path, "y_sub_" + str(sub)), y, allow_pickle=True)

