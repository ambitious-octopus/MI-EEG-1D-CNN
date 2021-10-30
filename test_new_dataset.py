import mne
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
"""
SOURCE: http://gigadb.org/dataset/view/id/100295
DATASET STRUCTURE: 
The MATLAB structure of the EEG (1st to 64th channel) and EMG (65th to 68th channel) data (“∗.mat”) for each subject 
is shown below:

 rest: resting state with eyes-open condition
 noise:
- eye blinking, 5 seconds × 2
- eyeball movement up/down, 5 seconds × 2
- eyeball movement left/right, 5 seconds × 2
- jaw clenching, 5 seconds × 2
- head movement left/right, 5 seconds × 2  imagery left: 100 or 120 trials of left hand MI  imagery right: 100 or 120 trials of right hand MI  n imagery trials: 100 or 120 trials for each MI class  imagery event: value “1” represents onset for each MI trial  movement left: 20 trials of real left hand movement  movement right: 20 trials of real right hand movement  n movement trials: 20 trials for each real hand movement
class
 movement event: value “1” represents onset for each movement trial
 frame: temporal range of a trial in milliseconds
 srate: sampling rate
 senloc: 3D sensor locations
 psenloc: sensor location projected to unit sphere
"""

data_path = "/home/kubasinska/Desktop/test_dataset/s01.mat"
info_path = "/home/kubasinska/Desktop/test_dataset/trial_sequence/s1_trial_sequence_v1.mat"

from mne.externals.pymatreader import read_mat
mat_data = read_mat(data_path)
info = read_mat(info_path)

"""
mat_data has the following keys:
    dict_keys(['noise', 'rest', 'srate', 'movement_left', 'movement_right', 'movement_event', 'n_movement_trials', 
    'imagery_left', 'imagery_right', 'n_imagery_trials', 'frame', 'imagery_event', 'comment', 'subject', 
    'bad_trial_indices', 'psenloc', 'senloc'])

mat_data['eeg']["noise"]
    list len 5: 
        1. eye blinking ->  5 seconds * 2
        2. eyeball movement up/down ->  5 seconds * 2
        3. eyeball movement left/right ->  5 seconds  * 2
        4. jaw clenching ->  5 seconds * 2
        5. head movement left/right ->  5 seconds * 2
    TEST: print([i.shape[1]/512 for i in mat_data['eeg']["noise"]])

mat_data["eeg"]["rest"]
    INFO: resting state with eyes-open condition
    np.array:
        1. rest -> 60 seconds
    TEST: print(mat_data["eeg"]["rest"].shape[1]/512)
    
mat_data["eeg"]['movement_left']
    INFO: 20 trials of real left hand movement
    np.array
        1. the monitor showed a black screen with a fixation cross for 2 seconds; “left hand” appeared on the screen 
        for 3 seconds, and subjects move the appropriate hand depending on the instruction given. After the movement, 
        blank screen reappeared, the subject was given a break for a random 4.1 to 4.8 seconds. These processes were 
        repeated 20 times for one class (one run), and one run was performed.
    
mat_data["eeg"]['movement_right']
    INFO: 20 trials of real right hand movement
    np.array
        1. the monitor showed a black screen with a fixation cross for 2 seconds; “right hand” appeared randomly on 
        the screen for 3 seconds, and subjects move the appropriate hand depending on the instruction given. After the 
        movement, blank screen reappeared, the subject was given a break for a random 4.1 to 4.8 seconds. These 
        processes were repeated 20 times for one class (one run), and one run was performed.
        
mat_data["eeg"]["n_movement_trials"]
    INFO: 20 trials for each real hand movement class
    int
    
mat_data["eeg"]["srate"]
    INFO: sampling rate
    int

mat_data["eeg"]["frame"]
    INFO: temporal range of a trial in milliseconds is 7000 ms 
    |__|__|__|__|__|__|__| = 7000ms (7 s) or (3584 sampling point)
     rest |  task  | rest
     
    |__| = 1s (1000 ms) or (512 sampling point)
    np.array

"""


slice = list()
channels = [9, 10, 12, 11, 17, 18, 45, 44, 48, 49, 55, 54]
for ch in channels:
    slice.append(mat_data["eeg"]["imagery_left"][ch])

data_slice = np.stack(slice) # SHAPE = CHANNELS * TIME

tasks = list()
task_time = 7*512

offsets = np.zeros(mat_data["eeg"]["imagery_event"].shape[0] + 512*7)

for index, event in enumerate(mat_data["eeg"]["imagery_event"]):
    if event == 1:
        tasks.append(data_slice[:, index:task_time+index])
        offsets[task_time+index] = 1


if not mat_data["eeg"]["n_imagery_trials"] == len(tasks):
    raise Exception("Abbiamo sbagliato tutto!")

y_raw = list()
x_raw = list()

for t in tasks:
    x_raw.append(t[:, :512])
    y_raw.append("B")

    x_raw.append(t[:, 512:1024])
    y_raw.append("B")

    x_raw.append(t[:, 1024:1536])
    y_raw.append("L")

    x_raw.append(t[:, 1536:2048])
    y_raw.append("L")

    x_raw.append(t[:, 2048:2560])
    y_raw.append("L")

    x_raw.append(t[:, 2560:3072])
    y_raw.append("B")

    x_raw.append(t[:, 3072:3584])
    y_raw.append("B")

for i in x_raw:
    for a in range(7):
        print(i.shape)
    print()

plt.plot(mat_data["eeg"]["imagery_event"])
plt.plot(offsets, c="r")
plt.show()

# x = np.stack(x_raw)