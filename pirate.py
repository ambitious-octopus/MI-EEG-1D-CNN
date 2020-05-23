import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import axes, image
from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations, make_ad_hoc_cov, compute_raw_covariance
from mne.simulation import add_noise
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.preprocessing import ICA, corrmap, read_ica
import os

class Pipe:
    @staticmethod

    def load(subjects):
        runs = [2]
    # Scarico i dati e mi ritorna il path locale
        raw_fnames = eegbci.load_data(subjects, runs)
    # Concateno le path e le metto in un unico file
        raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        return raw



