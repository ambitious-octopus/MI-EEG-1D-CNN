from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.epochs import Epochs
import mne
import numpy as np
from typing import List, TYPE_CHECKING


def load_data(subjects: List[int], runs: List[int]):
    """Load data from eegbci dataset
    :param subjects: list of integer
    :param runs: list of integer
    :return: list of list of raw objects
    """
    ls_run_tot = []  # list of lists, contains lists of Raw files to be concatenated
    # e.g. for subjects = [2, 45]
    # ls_run_tot = [[S2Raw03,S2Raw07,S2Raw11],
    #              [S45Raw03,S45Raw07,S45Raw11]]
    for subj in subjects:
        ls_run = []  # Lista dove inseriamo le run
        for run in runs:
            fname = eegbci.load_data(subj, runs=run)[0]  # Prendo le run
            raw_run = read_raw_edf(fname, preload=True)  # Le carico
            len_run = np.sum(raw_run._annotations.duration)  # Controllo la durata
            if len_run > 124:
                raw_run.crop(tmax=124)  # Taglio la parte finale
            ls_run.append(raw_run)
        ls_run_tot.append(ls_run)
    return ls_run_tot

def concatenate_runs(list_runs):
    """ Concatenate a list of runs
    :param list_runs: list of raw
    :return: list of concatenate raw
    """
    raw_conc_list = []
    for subj in list_runs:
        raw_conc = concatenate_raws(subj)
        raw_conc_list.append(raw_conc)
    return raw_conc_list

def del_annotations(list_of_subraw):
    """
    Delete "BAD boundary" and "EDGE boundary" from raws
    :param list_of_subraw: list of raw
    :return: list of raw
    """
    list_raw = []
    for subj in list_of_subraw:
        indexes = []
        for index, value in enumerate(subj.annotations.description):
            if value == "BAD boundary" or value == "EDGE boundary":
                indexes.append(index)
        subj.annotations.delete(indexes)
        list_raw.append(subj)
    return list_raw

def eeg_settings(raws):
    """
    Standardize montage of the raws
    :param raws: list of raws
    :return: list of standardize raws
    """
    raw_setted = []
    for subj in raws:
        eegbci.standardize(subj)  # Cambio n_epoch nomi dei canali
        montage = make_standard_montage('standard_1005')  # Caricare il montaggio
        subj.set_montage(montage)  # Setto il montaggio
        raw_setted.append(subj)

    return raw_setted


def filtering(list_of_raws):
    """
    Perform a band_pass and a notch filtering on raws
    :param list_of_raws:  list of raws
    :return: list of filtered raws
    """
    raw_filtered = []
    for subj in list_of_raws:
        if subj.info["sfreq"] == 160.0:
            subj.filter(1.0, 79.0, fir_design='firwin', skip_by_annotation='edge')  # Filtro passabanda
            subj.notch_filter(freqs=60)  # Faccio un filtro notch
            raw_filtered.append(subj)
        else:
            subj.filter(1.0, (subj.info["sfreq"]/2)-1, fir_design='firwin', skip_by_annotation='edge')  # Filtro passabanda
            subj.notch_filter(freqs=60)  # Faccio un filtro notch
            raw_filtered.append(subj)

    return raw_filtered


def epoch(raws, tmin = -1, tmax = 4):
    xs = list()
    ys = list()
    for raw in raws:
        event_id = dict(base=1, hands=2, feet=3)
        tmin, tmax = tmin, tmax
        events, _ = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))

        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')

        # Read epochs (train will be done only between 1 and 2s)
        # Testing will be done with a running classifier
        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                        baseline=None, preload=True)

        y = list()
        for index, data in enumerate(epochs):
                y.append(epochs[index]._name)

        xs.append(np.array([epoch for epoch in epochs]))
        ys.append(y)


    return np.concatenate(tuple(xs), axis=0), [item for sublist in ys for item in sublist]

def select_channels(raws, ch_list):
    s_list = []
    for raw in raws:
        s_list.append(raw.pick_channels(ch_list))
    
    return s_list
    
        


def get_data(subjects, runs, tmin, tmax, channels):
    return epoch(select_channels(filtering(eeg_settings(del_annotations(concatenate_runs(load_data(subjects=subjects, runs=runs))))), channels), tmin=tmin, tmax=tmax)


if __name__ == "__main__":
    subjects = list(np.arange(1,10))
    runs = [4,6,8,10,12,14]
    channels = ["C3", "C4"]

    raws = select_channels(filtering(eeg_settings(del_annotations(concatenate_runs(load_data(subjects=subjects, runs=runs))))),channels)

    xs = list()
    ys = list()
    for raw in raws:
        event_id = dict(base=1, hands=2, feet=3)
        tmin, tmax = 0, 4
        events, _ = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))

        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                               exclude='bads')

        # Read epochs (train will be done only between 1 and 2s)
        # Testing will be done with a running classifier
        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                        baseline=None, preload=True)

        y = list()
        for index, data in enumerate(epochs):
            y.append(epochs[index]._name)

        xs.append(np.array([epoch for epoch in epochs]))
        ys.append(y)



