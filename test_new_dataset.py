import mne

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

mat_data['eeg']["imagery_left"].shape