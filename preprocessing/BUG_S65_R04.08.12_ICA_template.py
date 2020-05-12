import numpy as np
import matplotlib.pyplot as plt

from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations, pick_channels
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.preprocessing import ICA
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap

#%% CARICO IL DATABASE
subject = [65]
runs = [4]
for subj in subject:
    ls_run = [] #Lista dove inseriamo le run
    for run in runs:
        fname = eegbci.load_data(subj, runs=run)[0] #Prendo le run
        raw_run = read_raw_edf(fname, preload=True) #Le carico
        len_run = np.sum((raw_run._annotations).duration) #Controllo la durata
        if len_run > 123:
            raw_run.crop(tmax=124.4) #Taglio la parte finale
        ls_run.append(raw_run) #Aggiungo la run alla lista delle run
#Concateno le path e le metto in un unico file
raw = concatenate_raws(ls_run)
#Standardizzo la posizione degli elettrodi
eegbci.standardize(raw)
#Seleziono il montaggio
montage = make_standard_montage('standard_1005')
#Lo setto all'interno di raw
raw.set_montage(montage)
#Tolgo il punto alla fine
raw.rename_channels(lambda x: x.strip('.'))
#%% VISUALIZZO I DATI RAW
raw.plot_psd(area_mode=None, show=False, average=False, fmin =1.0, fmax=80.0, dB=False, n_fft=160)
raw.plot()
#%%
#Applico un filtro passa banda
raw.filter(1., 79., fir_design='firwin', skip_by_annotation='edge')
raw.plot_psd(area_mode=None, show=False, average=False, fmin=1.0, fmax=80.0, dB=False, n_fft=160)
#Applico un NotchFilter
freqs = (60)
raw_notch = raw.notch_filter(freqs=freqs)

#%% ICA
#Istanzio una Ica
ica = ICA(n_components=64, random_state=10, method="fastica", max_iter=1000)
#Faccio il fit
ica.fit(raw)
#%%
#Plotto le concentrazioni
ica.plot_sources(raw)
#PLotto le proprietà della singola componente
ica.plot_properties(raw, dB=False,plot_std=False, picks=[61,57,51])



#Scarico i dati e mi ritorna il path locale
raw_fnames = eegbci.load_data(subject, runs)
#Concateno le path e le metto in un unico oggetto
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
#Standardizzo la posizione degli elettrodi
eegbci.standardize(raw)
#Seleziono il montaggio
montage = make_standard_montage('standard_1005')
#Lo setto all'interno di raw
raw.set_montage(montage)
#Tolgo il punto alla fine
raw.rename_channels(lambda x: x.strip('.'))
raw.crop(tmax = 60).load_data()
#%% VISUALIZZO I DATI RAW
raw.plot()
#%% FILTRAGGIO
#Applico un filtro passa banda
raw.filter(1.0, 79.0, fir_design='firwin', skip_by_annotation='edge')
#Applico un NotchFilter
freqs = (60)
raw_notch = raw.notch_filter(freqs=freqs)
raw.plot_psd(area_mode=None, show=False, average=False, fmin=1.0, fmax=80.0, dB=False, n_fft=160)
#%% ICA
#Istanzio una Ica
ica = ICA(n_components=64, random_state=10, method="fastica", max_iter=1000) #Deve ritornare due tuple!
#Faccio il fit
ica.fit(raw)
#%% ANALIZZO LE COMPONENTI ICA
#Elimino automaticamente le componenti che somigliano ad un artefatto oculare
eog_inds, eog_scores = ica.find_bads_eog(raw, ch_name='Fpz')
#%%
#Plotto le concentrazioni
ica.plot_sources(raw)
ica.plot_components()
#PLotto le proprietà della singola componente
ica.plot_properties(raw, dB=False,plot_std=False, picks=[0])
#%% DEFINIZIONE COMPONENTI
#Definisco delle componenti da escludere
exc = [1,0,12,11,18,29,28,36,34,49,47,45,63,51,52,53,56]
attesa = [4,23,49]
prot = [51,1,12, 36, 27]
#%% RICOSTRUZIONE
reconst_raw = raw.copy()
ica.plot_overlay(reconst_raw, exclude=exc)
ica.apply(reconst_raw, exclude=exc)
reconst_raw.plot_psd(area_mode=None, show=False, average=False, fmin=1.0, fmax=80.0, dB=False, n_fft=160)
#%% SPLITTO IL SEGNALE IN EPOCHE
#Carico gli eventi dal canale annotations
events, _ = events_from_annotations(reconst_raw, event_id=dict(T1=2, T2=3))
#Seleziono solo alcuni canali
picks = pick_channels(reconst_raw.info["ch_names"], ["C3", "Cz", "C2"])
# Definisco onset e offset delle epoche (secondi)
tmin, tmax = -1, 4
#Mappo i nomi degli eventi
event_ids = dict(left=2, right=3)
#Divido il tracciato in epoche
epochs = Epochs(reconst_raw, events, event_ids, tmin - 0.5, tmax + 0.5, picks=picks, baseline=None, preload=True)
#%% ERD
#Selezione un range di frequenza
freqs = np.arange(2, 50, 1)
n_cycles = freqs
#Onset e offset dei grafici
vmin, vmax = -1, 1.5
#Lunghezza della baseline
baseline = [-1, 0]
#Mappo i colori con una scala
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)

kwargs = dict(n_permutations=100, step_down_p=0.05, seed=10, buffer_size=None)
#Instazio una time-frequency representation
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=False, decim=2)
#tfr.crop(tmin, tmax)
#Applico la sottrazione della baseline
tfr.apply_baseline(baseline, mode="percent")

for event in event_ids:
    #Per ogni evento
    tfr_ev = tfr[event]
    #Faccio i subplot
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):  #Per ogni canale, indice
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1, **kwargs)
        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False), axes=ax, colorbar=False, show=False, mask=mask, mask_style="mask")

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if not ax.is_first_col():
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1])
    fig.suptitle("ERDS ({})".format(event))
    fig.show()
