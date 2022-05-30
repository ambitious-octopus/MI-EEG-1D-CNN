import pickle
import numpy as np
import os
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.epochs import Epochs
import mne

"""
This is a new generator that allows the data to be saved correctly.
Run this script and then load the data with load.py
"""


def load_subject_data(subject: int, data_path: str, exclude_base: bool = False):
    """
    Given a subject number (@subject) and the original dataset
    path (@data_path), this function returns:
        xs: The time series; a numpy array of shape (n_sample, 64, 641)
        y: The labels, a list of length n_samples
        ch_names: The 64 channels order in the xs array
    """
    runs = [4, 6, 8, 10, 12, 14]
    task2 = [4, 8, 12]
    task4 = [6, 10, 14]
    if len(str(subject)) == 1:
        sub_name = "S" + "00" + str(subject)
    elif len(str(subject)) == 2:
        sub_name = "S" + "0" + str(subject)
    else:
        sub_name = "S" + str(subject)
    sub_folder = os.path.join(data_path, sub_name)
    subject_runs = []
    for run in runs:
        if len(str(run)) == 1:
            path_run = os.path.join(sub_folder,
                                    sub_name + "R" + "0" + str(run) + ".edf")
        else:
            path_run = os.path.join(sub_folder,
                                    sub_name + "R" + str(run) + ".edf")
        raw_run = read_raw_edf(path_run, preload=True)
        len_run = np.sum(
            raw_run._annotations.duration)
        if len_run > 124:
            raw_run.crop(tmax=124)

        """
        B indicates baseline
        L indicates motor imagination of opening and closing left fist;
        R indicates motor imagination of opening and closing right fist;
        LR indicates motor imagination of opening and closing both fists;
        F indicates motor imagination of opening and closing both feet.
        """

        if int(run) in task2:
            for index, an in enumerate(raw_run.annotations.description):
                if an == "T0":
                    raw_run.annotations.description[index] = "B"
                if an == "T1":
                    raw_run.annotations.description[index] = "L"
                if an == "T2":
                    raw_run.annotations.description[index] = "R"
        if int(run) in task4:
            for index, an in enumerate(raw_run.annotations.description):
                if an == "T0":
                    raw_run.annotations.description[index] = "B"
                if an == "T1":
                    raw_run.annotations.description[index] = "LR"
                if an == "T2":
                    raw_run.annotations.description[index] = "F"
        subject_runs.append(raw_run)
    raw_conc = concatenate_raws(subject_runs)
    indexes = []
    for index, value in enumerate(raw_conc.annotations.description):
        if value == "BAD boundary" or value == "EDGE boundary":
            indexes.append(index)
    raw_conc.annotations.delete(indexes)

    eegbci.standardize(raw_conc)
    montage = make_standard_montage('standard_1005')
    raw_conc.set_montage(montage)
    tmin = 0
    tmax = 4
    if exclude_base:
        event_id = dict(F=2, L=3, LR=4, R=5)
    else:
        event_id = dict(B=1, F=2, L=3, LR=4, R=5)

    events, _ = mne.events_from_annotations(raw_conc, event_id=event_id)

    picks = mne.pick_types(raw_conc.info, meg=False, eeg=True, stim=False,
                           eog=False, exclude='bads')
    epochs = Epochs(raw_conc, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)

    print(epochs[0].ch_names)

    y = list()
    for index, data in enumerate(epochs):
        y.append(epochs[index]._name)

    xs = np.array([epoch for epoch in epochs])

    return xs, y, raw_conc.ch_names


exclude = [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1, 110) if n not in exclude]
save_path = "/dataset/sub_by_sub_motor_imagery"
if not os.path.exists(save_path):
    os.mkdir(save_path)
for subject in subjects:
    x, y, ch_names = load_subject_data(subject, "/dataset/original")
    with open(os.path.join(save_path, str(subject) + ".pkl"), "wb") as file:
        pickle.dump([x, y, ch_names], file)