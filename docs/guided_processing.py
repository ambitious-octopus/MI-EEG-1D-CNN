from data_processing.general_processor import Utils
import numpy as np
import os

exclude = [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1, 110) if n not in exclude]
runs = [4, 6, 8, 10, 12, 14]

channels = [["FC1", "FC2"],
            ["FC3", "FC4"],
            ["FC5", "FC6"],
            ["C5", "C6"],
            ["C3", "C4"],
            ["C1", "C2"],
            ["CP1", "CP2"],
            ["CP3", "CP4"],
            ["CP5", "CP6"]]

data_path = "D:\\datasets\\eegbci"

datas = Utils.load_data(subjects=subjects, runs=runs, data_path=data_path)

"""
Datas is a List of List of RawEdf -> List[List[RawEdf] where the first list is for each subjects
and the second list is for each run. At this point we concatenate the runs of each subject in one 
unique RawEdf.
"""

datas_concat = Utils.concatenate_runs(datas)

"""
When 2 runs are concatenated, mne-python tells us that the junction is not physiologically 
labeled with a red line and a caption in the annotations attribute. Since we are going to split 
the continuos signal into epochs, we don't care about that annotion, conseguentlly we totaly 
erase that label. 
"""

datas_no_bad_edges = Utils.del_annotations(datas_concat)

"""
todo: standardize datas
"""





