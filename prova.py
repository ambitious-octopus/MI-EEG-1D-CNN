import mne
import matplotlib.pyplot as plt
import numpy as np
import neurokit as nk

path = "D:\\datasets\\DEAP\\data_original\\s16.bdf"
data = mne.io.read_raw_bdf(path, preload=True)

gsr1 = data.get_data(picks=["GSR1"])

#Preprocessing eda
processed_eda = nk.eda_process(
        gsr1[0][0 : 512*60*10],
        sampling_rate=data.info["sfreq"],
        alpha=0.018, #cvxEDA penalization for the sparse SMNA driver.
        gamma=0.01, #cvxEDA penalization for the tonic spline coefficients.
    # , # Can be Butterworth filter ("butter"), Finite Impulse Response filter ("FIR"), Chebyshev filters ("cheby1" and "cheby2"), Elliptic filter ("ellip") or Bessel filter ("bessel"). Set to None to skip filtering.
        scr_method="m")

data.crop(tmin=0, tmax=10*60)
SCR_data = processed_eda["EDA"]
EDA_data = processed_eda["df"]

eda_raw = EDA_data['EDA_Raw']
eda_filtered = EDA_data['EDA_Filtered']
eda_phasic = EDA_data['EDA_Phasic']
eda_tonic = EDA_data['EDA_Tonic']

fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(eda_raw)
ax[0].plot(eda_filtered)
ax[1].plot(eda_phasic)
ax[2].plot(eda_tonic)
plt.show()

data._data[-8] = np.concatenate([eda_phasic.to_numpy(), np.array([0])], axis=0)


event = data.get_data(picks=['Status'])

"""
1 (First occurence)	N/A	start of experiment (participant pressed key to start)
1 (Second occurence)	120000 ms	start of baseline recording
1 (Further occurences)	N/A	start of a rating screen
2	1000 ms	Video synchronization screen (before first trial, before and after break, after last trial)
3	5000 ms	Fixation screen before beginning of trial
4	60000 ms	Start of music video playback
5	3000 ms	Fixation screen after music video playback
7	N/A	End of experiment
"""

map_events = {"start_of_experiment": 1 ,
            "video_sinc_screen": 2 ,
            "fixation_screen_before_trial": 3 ,
            "begin_music_video": 4 ,
            "after_video_fixation": 5 }
            # "end_experiment": 7 }

events = mne.find_events(data, stim_channel='Status')

# fig = mne.viz.plot_events(events, event_id=map_events, sfreq=data.info["sfreq"])


epochs = mne.Epochs(data,
                    events,
                    event_id=map_events,
                    preload=True,
                    picks=["GSR1"],
                    tmin=-2,
                    tmax=3)


conds_we_care_about = ["after_video_fixation", "fixation_screen_before_trial"]
epochs.equalize_event_counts(conds_we_care_about)  # this operates in-place

avg_dict = dict()
for cond in conds_we_care_about:
    avg_dict[cond] = epochs[cond].average()#.detrend(order=0)




mne.viz.plot_compare_evokeds(avg_dict, legend=3)



