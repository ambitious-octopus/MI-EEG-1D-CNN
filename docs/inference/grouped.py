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
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import pickle

models = ["a", "b", "c", "d", "e", "f"]
PATH = "/dataset/saved_models"


fig, axs = plt.subplots(2,3)
for index, roi_path in enumerate(os.scandir(PATH)):
    if len(roi_path.name) == 5:
        with open(os.path.join(roi_path.path, "hist.pkl"), "rb") as file:
            hist = pickle.load(file)

        axs[index].subplot(1, 2, 1, title="train accuracy")
        axs[index].plot(hist["accuracy"], label="Train")
        axs[index].plot(hist["val_accuracy"], label="Test")
        axs[index].legend(loc='lower right')
        axs[index].subplot(1, 2, 2, title="train loss")
        axs[index].plot(hist["val_loss"], label="Test")
        axs[index].plot(hist["loss"], label="Train")
        axs[index].legend(loc='upper right')


