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

class Pirates:
    """
    Class Pirates
    """
    @staticmethod
    def load_data(subjects, runs):
        """Load data from eegbci dataset
            :param subjects: list of integer
            :param runs: list of integer
            :return: list of list of raw object
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
                if len_run > 123:
                    raw_run.crop(tmax=124.4)  # Taglio la parte finale
                ls_run.append(raw_run)
            ls_run_tot.append(ls_run)
        return ls_run_tot

    @staticmethod
    def concatenate_runs(list_runs):
        """ Concatenate a list of runs
        :param list_runs: list of raw
        :return: list of concatenate raw
        """
        raw_conc_list = []
        for subj in range(len(list_runs)):
            raw_conc = concatenate_raws(list_runs[subj])
            raw_conc_list.append(raw_conc)
        return raw_conc_list

    @staticmethod
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

    @staticmethod
    def create_folders_psd():
        """
        Create folders in the current working directory
        :return: 5 folders dir_preprocessing, dir_psd_real, dir_pre_psd, dir_post_psd, dir_icas
        """
        # Checking current work directory
        cwd = os.getcwd()
        # Verifying "Preprocessing" directory existence
        # If not creating one
        dir_preprocessing = os.path.join(cwd, 'preprocessing')
        if os.path.isdir(dir_preprocessing):
            print('Preprocessing directory already exists')
        else:
            print('Path not found, creating preprocessing directory...')
            os.mkdir(dir_preprocessing)
        # Verifying dir_psd_real directory existence
        # If not creating one
        dir_psd_real = os.path.join(dir_preprocessing, 'psd_real')
        if os.path.isdir(dir_psd_real):
            print('Psd_real directory already exists')
        else:
            print('Path not found, creating psd_real directory...')
            os.mkdir(dir_psd_real)
        # Verifying pre_psd directory existence
        # If not creating one
        dir_pre_psd = os.path.join(dir_psd_real, 'pre_psd')
        if os.path.isdir(dir_pre_psd):
            print('Pre_psd directory already exists')
        else:
            print("Path not found, creating pre_psd directory...")
            os.mkdir(dir_pre_psd)
        # Verifying post_psd directory existence
        # If not creating one
        dir_post_psd = os.path.join(dir_psd_real, 'post_psd')
        if os.path.isdir(dir_post_psd):
            print('Post_psd directory already exists')
        else:
            print("Path not found, creating post_psd directory...")
            os.mkdir(dir_post_psd)
        dir_icas = os.path.join(dir_preprocessing, 'icas')
        if os.path.isdir(dir_icas):
            print('Icas directory already exists')
        else:
            print("Path not found, creating icas directory...")
            os.mkdir(dir_icas)

        return dir_preprocessing, dir_psd_real, dir_pre_psd, dir_post_psd, dir_icas

    @staticmethod
    def eeg_settings(raws):
        """
        Standardize montage of the raws
        :param raws: list of raws
        :return: list of standardize raws
        """
        raw_setted = []
        for subj in raws:
            eegbci.standardize(subj)  # Cambio i nomi dei canali
            montage = make_standard_montage('standard_1005')  # Caricare il montaggio
            subj.set_montage(montage)  # Setto il montaggio
            raw_setted.append(subj)

        return raw_setted

    @staticmethod
    def plot_pre_psd(list_of_raws, dir_pre_psd, overwrite=False):
        """
        Save psd of raws in dir_pre_psd
        :param list_of_raws: list of raws
        :param dir_pre_psd: absolute path of dir_pre_psd
        :param overwrite: bolean overwrite
        :return: None
        """
        for subj in list_of_raws:
            plot_pre = plt.figure()  # I create an empty figure so I can apply clear next line
            plot_pre.clear()  # Clearing the plot each iteration I do not obtain overlapping plots
            # Plots psd of filtered raw data
            plot_pre = subj.plot_psd(area_mode=None, show=True, average=False, ax=plt.axes(ylim=(0, 60)), fmin=1.0,
                                     fmax=80.0, dB=False, n_fft=160)
            # Creates plot's name
            psd_name = os.path.join(dir_pre_psd, subj.__repr__()[10:14] + '_real_pre.png')
            # Check if plot exists by checking its path
            if os.path.exists(psd_name) and overwrite == False:  # stops only if overwrite == False
                raise Exception('Warning! This plot already exists! :(')
            elif os.path.exists(psd_name) and overwrite == True:
                os.remove(psd_name)  # removes existing saved plot
                if os.path.exists(psd_name):
                    raise Exception(
                        'You did not remove existing plot!')  # check if plot being deleted, not always os.remove works
                else:
                    plot_pre.savefig(psd_name)
            else:
                plot_pre.savefig(psd_name)
        return None


    @staticmethod
    def filtering(list_of_raws):
        """
        Perform a band_pass and a notch filtering on raws
        :param list_of_raws:  list of raws
        :return: list of filtered raws
        """
        raw_filtered = []
        for subj in list_of_raws:
            subj.filter(1.0, 79.0, fir_design='firwin', skip_by_annotation='edge')  # Filtro passabanda
            subj.notch_filter(freqs=60)  # Faccio un filtro notch
            raw_filtered.append(subj)

        return raw_filtered


    @staticmethod
    def ica_function(list_of_raws,dir_icas, save=True, overwrite=True):
        """

        :param list_of_raws: list of raws
        :param dir_icas: dir to save icas
        :param save: boolean
        :param overwrite: boolean
        :return: list of icas objects
        """

        icas_names = []  # Creating here empty list to clean it before using it, return ica saved paths for future loading
        icas = []  # return icas

        for subj in list_of_raws:
            ica = ICA(n_components=64, random_state=42, method="fastica", max_iter=1000)
            ica.fit(subj)  # fitting ica
            if save == True:
                icas_name = os.path.join(dir_icas, subj.__repr__()[10:14] + '_ica.fif')  # creating ica name path
                if os.path.exists(icas_name) and overwrite == False:
                    raise Exception('Warning! This ica already exists! :( ' + str(
                        icas_name))  # stops if you don't want to overwrite a saved ica
                elif os.path.exists(icas_name) and overwrite == True:
                    os.remove(icas_name)  # to overwrite delets existing ica
                    ica.save(icas_name)
                    icas_names.append(icas_name)
                    print('Overwriting existing icas')
                else:
                    ica.save(icas_name)
                    print('Creating ica files')
                    icas_names.append(icas_name)
            icas.append(ica)
        return icas

    @staticmethod
    def plot_post_psd(list_of_raws, dir_post_psd, overwrite=False):
        """
        Save post psd
        :param list_of_raws: list of raws
        :param dir_post_psd: dir to save post_psd
        :param overwrite: boolean
        :return: None
        """
        for subj in list_of_raws:
            plot_post = plt.figure()  # I create an empty figure so I can apply clear next line
            plot_post.clear()  # Clearing the plot each iteration I do not obtain overlapping plots
            # Plots psd of filtered raw data
            plot_post = subj.plot_psd(area_mode=None, show=True, average=False, ax=plt.axes(ylim=(0, 60)), fmin=1.0, fmax=80.0, dB=False, n_fft=160)
            # Creates plot's name
            psd_name = os.path.join(dir_post_psd, subj.__repr__()[10:14] + '_real_post.png')
            # Check if plot exists by checking its path
            if os.path.exists(psd_name) and overwrite == False:  # stops only if overwrite == False
                raise Exception('Warning! This plot already exists! :(')
            elif os.path.exists(psd_name) and overwrite == True:
                os.remove(psd_name)  # removes existing saved plot
                if os.path.exists(psd_name):
                    raise Exception(
                        'You did not remove existing plot!')  # check if plot being deleted, not always os.remove works
                else:
                    plot_post.savefig(psd_name)
            else:
                plot_post.savefig(psd_name)
        return None

