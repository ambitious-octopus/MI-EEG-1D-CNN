import mne
import numpy as np
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.epochs import Epochs


runs = list()
# This is my data path
data_path = "D:\datasets\eegbci"

def load_data(subjects, runs):
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


raw = eeg_settings(del_annotations(concatenate_runs(load_data([1], [3, 7, 11]))))[0]
event_id = dict(base=1, hands=2, feet=3)
tmin, tmax = -1., 4.
events, _ = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))

picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)

y = list()
for index, data in enumerate(epochs):
    if epochs[index]._name == "base":
        y.append(1)
    if epochs[index]._name == "hands":
        y.append(2)
    if epochs[index]._name == "feet":
        y.append(3)

x = np.array([epoch for epoch in epochs])
y = np.array(y)


"""
First Simple model
"""



import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

tf.keras.utils.normalize(x, axis=-1, order=2)

# Creo un modello
# Inizializzo un ogetto sequential
model = keras.models.Sequential()
# Primo layer di input
model.add(keras.layers.Flatten(input_shape=[64, 801]))
# Creo tre layers
model.add(keras.layers.Dense(1000, activation="relu"))
model.add(keras.layers.Dense(1000, activation="relu"))
model.add(keras.layers.Dense(3, activation="softmax"))
# Guardo il modello
model.summary()
# Compilo il modello
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=["accuracy"])

# Faccio partire il training
history = model.fit(x, y, epochs=80)

import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
