import tensorflow as tf
import numpy as np

inp = np.array([[[0],[1],[2],[3],[4]],[[5],[4],[3],[2],[1]]]).astype(np.float32)
kernel = tf.Variable(tf.initializers.glorot_uniform()([5, 1, 4]), dtype=tf.float32)
out = tf.nn.conv1d(inp, kernel, stride=1, padding='same')
print(out)



from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
subject= 1
runs = [4]
raw_fnames = eegbci.load_data(subject, runs)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raws)

raw.annotations.duration