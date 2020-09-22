import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps
from matplotlib import axes, image
import mne
from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations, make_ad_hoc_cov, compute_raw_covariance
from mne.simulation import add_noise
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.preprocessing import ICA, corrmap, read_ica
import os
from mne.time_frequency import psd_multitaper, psd_welch
from mne.epochs import make_fixed_length_epochs
from mne import create_info
from mne import find_events, EpochsArray


class Pirates:
    """
    Class Pirates
    """
    print("Seems that your are using Pirates, have fun!")

    @staticmethod
    def load_data(subjects, runs):
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

    @staticmethod
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
    def setup_folders(path_to_save):
        """
        Create folders in the current working directory
        :path_to_save: string, path to save
        :return: dir_psd, dir_pre_psd, dir_post_psd, dir_icas, dir_report, dir_templates
        """
        # Checking current work directory
        path = path_to_save
        # Verifying "Preprocessing" directory existence
        # If not creating one
        # Verifying dir_psd_real directory existence
        # If not creating one
        dir_psd = os.path.join(path, 'psd')
        if os.path.isdir(dir_psd):
            print('Psd_real directory already exists')
        else:
            print('Path not found, creating psd_real directory...')
            os.mkdir(dir_psd)
        # Verifying pre_psd directory existence
        # If not creating one
        dir_pre_psd = os.path.join(dir_psd, 'pre_psd')
        if os.path.isdir(dir_pre_psd):
            print('Pre_psd directory already exists')
        else:
            print("Path not found, creating pre_psd directory...")
            os.mkdir(dir_pre_psd)
        # Verifying post_psd directory existence
        # If not creating one
        dir_post_psd = os.path.join(dir_psd, 'post_psd')
        if os.path.isdir(dir_post_psd):
            print('Post_psd directory already exists')
        else:
            print("Path not found, creating post_psd directory...")
            os.mkdir(dir_post_psd)

        dir_dis = os.path.join(dir_psd, 'discrepancy')
        if os.path.isdir(dir_dis):
            print('discrepancy directory already exists')
        else:
            print("Path not found, creating discrepancy directory...")
            os.mkdir(dir_dis)

        dir_icas = os.path.join(path, 'icas')
        if os.path.isdir(dir_icas):
            print('Icas directory already exists')
        else:
            print("Path not found, creating icas directory...")
            os.mkdir(dir_icas)

        dir_report = os.path.join(path, 'report_psd')
        if os.path.isdir(dir_report):
            print('report_psd directory already exists')
        else:
            print('Path not found, creating report_psd directory...')
            os.mkdir(dir_report)

        dir_templates = os.path.join(path, 'template')
        if os.path.isdir(dir_templates):
            print('Icas directory already exists')
        else:
            print("Path not found, creating icas directory...")
            os.mkdir(dir_templates)

        dir_psd_topo_map = os.path.join(path, 'psd_topo_map')
        if os.path.isdir(dir_psd_topo_map):
            print('dir_psd_topo_map directory already exists')
        else:
            print('Path not found, creating dir_psd_topo_map directory...')
            os.mkdir(dir_psd_topo_map)

        dir_psd_icas = os.path.join(path, 'dir_psd_icas')
        if os.path.isdir(dir_psd_icas):
            print('dir_psd_icas directory already exists')
        else:
            print('Path not found, creating dir_psd_icas directory...')
            os.mkdir(dir_psd_icas)

        return dir_psd_icas, dir_dis, dir_psd, dir_pre_psd, dir_post_psd, dir_icas, dir_report, dir_templates, dir_psd_topo_map

    @staticmethod
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
            if subj.info["sfreq"] == 160.0:
                plot_pre = subj.plot_psd(area_mode=None, show=False, average=False, ax=plt.axes(ylim=(0, 30)), fmin=1.0,
                                     fmax=80.0, dB=False, n_fft=160)
            else:
                plot_pre = subj.plot_psd(area_mode=None, show=False, average=False, ax=plt.axes(ylim=(0, 30)), fmin=1.0,
                                         fmax=64.0, dB=False, n_fft=160)
            # Creates plot's name
            psd_name = os.path.join(dir_pre_psd, subj.__repr__()[10:14] + '.png')
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
            plt.close('all')
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
            if subj.info["sfreq"] == 160.0:
                subj.filter(1.0, 79.0, fir_design='firwin', skip_by_annotation='edge')  # Filtro passabanda
                subj.notch_filter(freqs=60)  # Faccio un filtro notch
                raw_filtered.append(subj)
            else:
                subj.filter(1.0, (subj.info["sfreq"]/2)-1, fir_design='firwin', skip_by_annotation='edge')  # Filtro passabanda
                subj.notch_filter(freqs=60)  # Faccio un filtro notch
                raw_filtered.append(subj)

        return raw_filtered

    @staticmethod
    def ica_function(list_of_raws, dir_icas=None, save=False, overwrite=False):
        """
            Perform an ica
        :param list_of_raws: list of raws objects
        :param dir_icas: dir to save icas
        :param save: boolean
        :param overwrite: boolean
        :return: list of icas objects
        """
        icas_names = []  # Creating here empty list to clean it before using it, return ica saved paths for future loading
        icas = []  # return icas

        if type(list_of_raws) == list:
            pass
        else:
            list(list_of_raws)

        for subj in list_of_raws:
            ica = ICA(n_components=64, random_state=10, method="fastica", max_iter=1000)
            ica.fit(subj)
            ica.info["S"] = subj.__repr__()[10:14]
            if save == True:
                icas_name = os.path.join(dir_icas, subj.__repr__()[10:14] + '_ica.fif')
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
            if subj.info["sfreq"] == 160.0:
                plot_post = subj.plot_psd(area_mode=None, show=False, average=False, ax=plt.axes(ylim=(0, 30)), fmin=1.0,
                                        fmax=80.0, dB=False, n_fft=160)
            else:
                plot_post = subj.plot_psd(area_mode=None, show=False, average=False, ax=plt.axes(ylim=(0, 30)),
                                          fmin=1.0, fmax=64.0, dB=False, n_fft=160)
            # Creates plot's name
            psd_name = os.path.join(dir_post_psd, subj.__repr__()[10:14] + '.png')
            # Check if plot exists by checking its path
            if os.path.exists(psd_name) and overwrite == False:
                raise Exception('Warning! This plot already exists! :(')
            elif os.path.exists(psd_name) and overwrite == True:
                os.remove(psd_name)
                if os.path.exists(psd_name):
                    raise Exception(
                        'You did not remove existing plot!')  # check if plot being deleted, not always os.remove works
                else:
                    plot_post.savefig(psd_name)
            else:
                plot_post.savefig(psd_name)
            plt.close('all')
        return None

    @staticmethod
    def reconstruct_raws(list_icas, list_raws, labels):
        """
        Appply a recostruction icas
        :param list_icas: list of icas
        :param list_raws: list of raws
        :param labels: string, label to exclude
        :return: list of recostruited raws
        """
        reco_raws = []

        for index, ica in enumerate(list_icas):
            reco_raw = list_raws[index].copy()
            ica.exclude = ica.labels_[labels]
            ica.apply(reco_raw)
            reco_raws.append(reco_raw)

        return reco_raws

    @staticmethod
    def create_report_psd(directory_pre_psd, directory_post_psd, dir_dis, dir_report):
        """
        Take pre_psd and post_psd and merge in one image located in the dir_report
        :param directory_pre_psd: string, directory of pre_psd
        :param directory_post_psd: string, directory of post_psd
        :param dir_dis: string, directory that contains discrepancy
        :param dir_report: string, directory to save the merged images
        :return: None
        """
        pre_psd = []
        post_psd = []
        dis_psd = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(directory_pre_psd):
            for file in f:
                # if '.png' in file:
                pre_psd.append(os.path.join(r, file))
        for r, d, f in os.walk(directory_post_psd):
            for file in f:
                # if '.png' in file:
                post_psd.append(os.path.join(r, file))
        for r, d, f in os.walk(dir_dis):
            for file in f:
                # if '.png' in file:
                dis_psd.append(os.path.join(r, file))

        pre_images = [Image.open(x) for x in pre_psd]
        post_images = [Image.open(x) for x in post_psd]
        dis_images = [Image.open(x) for x in dis_psd]
        text = ["PRE", "POST", "DISCREPANCY"]
        font = ImageFont.truetype("arial.ttf", 20)
        for ind, image in enumerate(pre_images):
            img = [pre_images[ind], post_images[ind], dis_images[ind]]
            width, height = image.size
            real_width = width * len(img)
            real_height = height
            new_im = Image.new('RGB', (real_width, real_height))
            x_offset = 0
            for index, im in enumerate(img):
                d = ImageDraw.Draw(im)
                d.text((150, 2), text[index], fill=(0, 0, 0), font=font)
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]
            new_im.save(os.path.join(dir_report, pre_psd[ind][-8:-4] + ".png"))

        return None

    @staticmethod
    def corr_map(list_icas, ica_template, comp_template, dir_templates, label, threshold=0.85):
        """
        apply a corr map modify objects list_icas inplace
        :param list_icas: list of icas objects
        :param ica_template: int positional int of the ica to use as template
        :param comp_template: list of int, list of components
        :param dir_templates: string, diretory to save templates
        :param threshold: float64, set the threshold
        :param label: string, label
        """
        for comp in comp_template:
            corr = corrmap(list_icas, template=(ica_template, comp), plot=True, show=False, threshold=threshold,
                           label=label)
            path_topos = os.path.join(dir_templates, str(comp) + ".png")
            corr[0].savefig(path_topos)
            topos = Image.open(path_topos)
            if type(corr[1]) == list:
                lst = [topos]
                paths = [path_topos]
                total_width = 0
                total_height = 0
                for ind, im in enumerate(corr[1]):
                    path_im = os.path.join(dir_templates, str(comp) + str(ind) + ".png")
                    paths.append(path_im)
                    im.savefig(path_im)
                    scalps = Image.open(path_im)
                    lst.append(scalps)
                    width, height = scalps.size
                    total_width += width
                    if height > total_height:
                        total_height = height
                new_im = Image.new('RGBA', (total_width, total_height))
                x_offset = 0
                for scalp in lst:
                    new_im.paste(scalp, (x_offset, 0))
                    x_offset += scalp.size[0]
                new_im.save(os.path.join(dir_templates, "component" + str(comp) + ".png"))
                # for n_epoch in lst:
                # n_epoch.close()
                for p in paths:
                    os.remove(p)
                plt.close('all')

            else:
                path_found = os.path.join(dir_templates, str(comp) + "a" + ".png")
                corr[1].savefig(path_found)
                topos = Image.open(path_topos)
                found = Image.open(path_found)
                img = [topos, found]
                width_0, height_0 = img[0].size
                width_1, height_1 = img[1].size
                real_width = width_0 + width_1
                real_height = height_1
                new_im = Image.new('RGBA', (real_width, real_height))
                x_offset = 0
                for im in img:
                    new_im.paste(im, (x_offset, 0))
                    x_offset += im.size[0]
                new_im.save(os.path.join(dir_templates, "component" + str(comp) + ".png"))
                os.remove(path_topos)
                os.remove(path_found)
                plt.close('all')

    @staticmethod
    def load_saved_icas(dir_icas, from_ica, to_ica):
        """
        Load saved icas
        :param dir_icas: str, directory of the icas
        :param from_ica: int, from ica number
        :param to_ica: int, to ica number
        :return: list of icas object
        """
        icas = []
        selection = np.arange(from_ica, to_ica + 1)
        for dir, _, file in os.walk(dir_icas):
            for fi in file:
                if fi[1] == "0":
                    subj = int(fi[2:4])
                else:
                    subj = int(fi[1:4])

                if subj in selection:
                    ica = read_ica(os.path.join(dir, fi))
                    ica.info["S"] = fi[1:4]
                    icas.append(ica)
        return icas

    @staticmethod
    def discrepancy(raws, clean_raws, dir_discrepancy, return_raw=False):
        """
        Calculate and plot discrepacy
        :param raws: list, of raws objects
        :param clean_raws: list, of raws objects
        :param dir_discrepancy: str, directory to save the plot of the discrepacy
        :param return_raw: bool
        :return: if return_raw return a lst of raw with the discrepacy
        """
        lst = list()
        for raw, c_raw in zip(list(raws), list(clean_raws)):
            dis = np.subtract(raw._data, c_raw._data)
            raw_cop = raw.copy()
            raw_cop._data = dis
            if raw_cop.info["sfreq"] == 160.0:
                psd = raw_cop.plot_psd(area_mode=None, show=False, average=False, ax=plt.axes(ylim=(0, 30)), fmin=1.0,
                                    fmax=80.0, dB=False, n_fft=160)
            else:
                psd = raw_cop.plot_psd(area_mode=None, show=False, average=False, ax=plt.axes(ylim=(0, 30)), fmin=1.0,
                                       fmax=64.0, dB=False, n_fft=160)
            psd_name = os.path.join(dir_discrepancy, raw_cop.__repr__()[10:14] + '.png')
            psd.savefig(psd_name)
            plt.close('all')
            if return_raw:
                lst.append(raw_cop)
        if return_raw:
            return lst

    @staticmethod
    def psd_topo_map(icas, raws, label, dir_topo_psd):
        """
        Save an image composed by psd and topomap of all the icas component
        :param icas: list of icas
        :param raws: list, of raws
        :param label: str, label to set as excluded
        :param dir_topo_psd: str, directory to save
        :return: None
        """
        n_comp = np.arange(0, 64)
        subjs = list()
        paths_to_delete = []
        for ica, subj in zip(icas, raws):
            img_paths = list()
            lab = ica.labels_[label]
            inst = make_fixed_length_epochs(subj.copy(), duration=2., verbose=False, preload=True)
            data = ica.get_sources(inst).get_data()  # prendo l'array delle componenti
            ica_names = ica._ica_names  # mi prendo la lista di nomi di icas da usare al posto dei canali
            info2 = subj.info  # mi copio le info dai dati raw, da questi mi servirà solo ['chs']
            # chs = info2['chs']
            info3 = create_info(sfreq=160, ch_names=ica_names, ch_types='eeg')
            comp_epocate = EpochsArray(data, info3)  # creo un EpochsArray con le epoche calcolate prima(data) e
            # la info structure creata prima
            for n in n_comp:
                psd = comp_epocate.plot_psd(dB=False, area_mode=None, average=False,
                                            ax=plt.axes(yticks=([]), title="Sub"),
                                            fmin=1.0, fmax=80.0, picks=n, spatial_colors=False, show=False)
                if n in lab:
                    ax1 = ica.plot_components(picks=n, title="Excluded")
                else:
                    ax1 = ica.plot_components(picks=n, title="")
                path_comp = os.path.join(dir_topo_psd, "TOPO" + subj.__repr__()[10:14] + "C" + str(n) + ".png")
                ax1.savefig(path_comp)
                plt.close(ax1)
                path_psd = os.path.join(dir_topo_psd, "PSD" + subj.__repr__()[10:14] + "C" + str(n) + ".png")
                psd.set_size_inches(3, 2.5)
                psd.savefig(path_psd)
                plt.close(psd)
                psd = Image.open(path_comp)
                topo = Image.open(path_psd)
                imgs = [psd, topo]
                width_0, height_0 = imgs[0].size
                width_1, height_1 = imgs[1].size
                real_width = width_0 + width_1
                real_height = height_1 + 15
                new_im = Image.new('RGBA', (real_width, real_height))
                x_offset = 0
                for im in imgs:
                    new_im.paste(im, (x_offset, 0))
                    x_offset += im.size[0]
                img_path = os.path.join(dir_topo_psd, subj.__repr__()[10:14] + "C" + str(n) + ".png")
                new_im.save(img_path)
                img_paths.append(img_path)
                paths_to_delete.append(path_comp)
                paths_to_delete.append(path_psd)
                plt.close('all')
            subjs.append(img_paths)
        for indi, l in enumerate(subjs):
            imgs = [Image.open(x) for x in l]
            width, height = imgs[0].size
            real_width = width * 8 + 4
            real_height = height * 8 + 4
            new_im = Image.new('RGBA', (real_width, real_height))
            x_offset = 0
            y_offset = 0
            count = -1
            for ind, image in enumerate(imgs):
                count += 1
                if count != 0 and count % 8 == 0:
                    y_offset += image.size[1]
                    x_offset = 0
                new_im.paste(image, (x_offset, y_offset))
                x_offset += image.size[0]
            new_im.save(os.path.join(dir_topo_psd, "total" + raws[indi].__repr__()[10:14] + ".png"))

        for l in subjs:
            for p in l:
                os.remove(p)
        for y in paths_to_delete:
            os.remove(y)

    @staticmethod
    def get_ica_psd(raws, icas, dir_icas_psd=None, return_figure=False):
        """
        Save the psd of the icas
        :param raws: list, of raw objects
        :param icas: list, of icas object
        :param dir_icas_psd: str, directory to save
        :param return_figure: bool, if return the figures or to save only
        :return: if return_figure is True return a list of matplolib figures
        """
        psdl = list()
        for raw, ica in zip(raws, icas):
            inst = make_fixed_length_epochs(raw.copy(), duration=2., verbose=False, preload=True)
            data = ica.get_sources(inst).get_data()  # prendo l'array delle componenti
            ica_names = ica._ica_names  # mi prendo la lista di nomi di icas da usare al posto dei canali
            info2 = raw.info  # mi copio le info dai dati raw, da questi mi servirà solo ['chs']
            chs = info2['chs']
            info3 = create_info(sfreq=160, ch_names=ica_names, ch_types='eeg')
            comp_epocate = EpochsArray(data, info3)  # creo un EpochsArray con le epoche calcolate prima(data) e
            # la info structure creata prima
            psd = comp_epocate.plot_psd(dB=False, area_mode=None, average=False, ax=plt.axes(yticks=([]), title="Sub"),
                                        fmin=1.0, fmax=80.0, picks="all", spatial_colors=False, show=False)
            psdl.append(psd)

            if dir_icas_psd is not None:
                psd.savefig(os.path.join(dir_icas_psd, "ICA-psd" + raw.__repr__()[10:14] + ".png"))
            plt.close(psd)

        if return_figure:
            return psdl

    @staticmethod
    def select_components(icas, raws, label):
        """
        Activate an interactive shell to exclude or remove components
        :param icas: list, of icas onjects
        :param raws: list, of raws object
        :param label: str, name of the label
        :return: list, of icas labeled with exclusion
        """
        if type(icas) == list and type(raws) == list:
            pass
        else:
            icas = [icas]
            raws = [raws]

        for ica, raw in zip(icas, raws):
            comand = ""
            while comand != "next":
                print("Your are working on subject: " + raw.__repr__()[11:14])
                comand = input("Inserisci un comando: ")
                print()
                if comand == "exclude":
                    number_comp_to_exclude = int(input("Which component do you want to exclude? "))
                    ica.labels_[label].append(number_comp_to_exclude)
                if comand == "show":
                    print("Subject: " + raw.__repr__()[11:14] + " Excluded component:")
                    print(ica.labels_[label])
                    print()
                if comand == "remove":
                    number_comp_remove = int(input("Which component do you want to remove from exclusion list? "))
                    if number_comp_remove in ica.labels_[label]:
                        ica.labels_[label] = [x for x in ica.labels_[label] if x != number_comp_remove]
                    else:
                        print("you didn't exclude this component")
                if comand == "help":
                    print("The available commands are: ")
                    print()
                    print("remove: remove a component from the list")
                    print("exclude: set a component to exclude")
                    print("show: show the actual components selected")
        return icas

    @staticmethod
    def save_exclusion(icas, file_path, is_there_template_also=False):
        """
        Add exclusions to the master file exclusion
        :param icas:
        :param file_path:
        :param is_there_template_also:
        :return:
        """
        excluded = []
        file = np.load(file_path, allow_pickle=True)
        if is_there_template_also:
            for ica in icas:
                excluded.append(ica.labels_)
            new = np.concatenate((file, excluded))
            np.save(file_path, new)
        else:
            for ica in icas[1:]:
                excluded.append(ica.labels_)
            new = np.concatenate((file, excluded))
            np.save(file_path, new)

    @staticmethod
    def load_exclusion(icas, path):
        """
        Merge a list of icas with corresponding label of excluded components
        icas are modified in-place
        :param icas: list, of icas object
        :param path: str, the master exclusion file
        :return: None
        """
        di = np.load(path, allow_pickle=True)
        for value, ica in zip(di, icas):
            ica.labels_ = value
        return None

    @staticmethod
    def check_exclusion(file_path):
        """
        Visualize the exclusion for each subject
        :param file_path: str, file path of the exclusions
        :return: a dictionary
        """
        dic = np.load(file_path)
        for index, subj in enumerate(dic):
            subj["Subject: " + str(index + 1)] = subj.pop("artifact")
        return dic

    @staticmethod
    def interpolate(raw_data, channels):
        """
        Interpolate chanel in-place
        :param raw_data: raw, a raw object
        :param channels: str, name of the channel
        :return: None
        """
        raw_data.info['bads'].append(channels)
        raw_data.interpolate_bads()

    @staticmethod
    def save_exclusion_file(icas, file_name):
        """
        Save a file with the exclusion
        :param icas: list, of icas
        :param name: str, path
        :return: None
        """
        file = []
        for ica in icas:
            file.append(ica.labels_)
        np.save(file_name + ".npy", file)

    @staticmethod
    def save_fif(raws, dir_to_save):
        for raw in raws:
            name = raw.__repr__()[10:14]
            raw.save(os.path.join(dir_to_save, name + ".fif"), overwrite=True)


    @staticmethod

    def image_generation (list_of_raws, dir_tfr_imagined):
        ch = ["C1", "C2", "C3", "C4"]
        for raw in list_of_raws:
            events, _ = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))
            picks = mne.pick_channels(raw.info["ch_names"], ch)
            tmin, tmax = 0, 3
            event_ids = dict(base=1, left=2, right=3)
            epochs = mne.Epochs(raw, events, event_ids, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True)
            freqs = np.arange(5, 38, 1)  # frequencies from 2-35Hz
            n_cycles = freqs  # use constant t/f resolution
            # Run TF decomposition overall epochs
            for index,e in enumerate(epochs):
                tfr = mne.time_frequency.tfr_multitaper(epochs[index], freqs=freqs, n_cycles=n_cycles,use_fft=True, return_itc=False, average=True,decim=1)
                lst_path_img = list()
                for channel in ch:
                    a = tfr.plot([channel], cmap="jet", vmin=0, vmax=0.000000009, colorbar=True)
                    a.savefig(os.path.join(dir_tfr_imagined, str(channel) + ".jpg"))
                    lst_path_img.append(os.path.join(dir_tfr_imagined, str(channel) + ".jpg"))
                plt.close("all")
                imgs = []
                for im in lst_path_img:
                    img = Image.open(im)
                    imgs.append(img)
                new_imgs = []
                new_imgs_names = list()
                for im, path in zip(imgs, lst_path_img):
                    border = (81, 59, 163, 53)
                    new = ImageOps.crop(im, border)
                    new_imgs.append(new)
                    new.save(path)
                    new_imgs_names.append(path)
                len_big_image = int(len(new_imgs) / 2)
                width, height = new_imgs[0].size
                real_width = width * len_big_image
                real_height = height * len_big_image
                mixed = Image.new('RGB', (real_width, real_height))
                c1 = (0, 0)
                c2 = (new_imgs[0].size[0], 0)
                c3 = (0, new_imgs[0].size[1])
                c4 = (new_imgs[0].size[0], new_imgs[0].size[1])
                position = [c1, c2, c3, c4]
                channel_dict = {c1: "1", c2: "2", c3: "3", c4:"4" }
                channel_list = []
                for ind, pos in enumerate(position):
                    channel_list.append(channel_dict.pop(position[ind]))
                    channel_str = ''.join(channel_list)

                final_im_list = []
                img_path_list = []
                for im, pos in zip(new_imgs, position):
                    final_im = mixed.paste(im, pos)
                    final_im_list.append(final_im)
                img_path = os.path.join(dir_tfr_imagined, raw._filenames[0][-8:-4] + "_i_"+ "e"+ str(epochs[index].selection[0]) + "C" + channel_str + "_" + epochs[index].event_id.__str__()[2]+".jpg")
                img_path_list.append(im)
                mixed.save(img_path)
                print(str(raw._filenames[0][-8:-4]) + "epoch: " + str(epochs[index].selection[0]))

                return final_im_list,img_path_list


    @staticmethod

    def image_resize(img_path_list,image_resized_path):

        """
              Resize images from list of paths of previous images
              :param img_path_list: list, of paths of previous images
              :param imgage_resized_path: path to save resized images
              :param name: list of path, path
              :return: None
              """
        if img_path_list != list:
            raise Exception


        image_list_resized = []

        for image in img_path_list:

            im= Image.open(image)
            image_resized = im.resize((32,32))
            image_list_resized.append(image_resized)
            image_resized.save(image_resized_path)

        return image_list_resized




if __name__ == "__main__":
    pass


