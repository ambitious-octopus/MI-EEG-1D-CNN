#Importing stuff
import os
import sys
import matplotlib
matplotlib.use('Qt5Agg')
# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
import matplotlib.pyplot as plt
print(os.getcwd())
print(sys.path)
sys.path.append("/home/kubasinska/data/repos/eeGNN")
from model_set.models import HopefullNet
import numpy as np
import tensorflow as tf
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)
from sklearn.preprocessing import minmax_scale
tf.autograph.set_verbosity(0)
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


PATH = "/home/kubasinska/datasets/eegbci/paper"

channels = [["CP3", "CP4"]]

exclude =  [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1,109) if n not in exclude]
#Load data
x, y = Utils.load(channels, subjects, base_path=PATH)

datas = {key: list() for key in np.unique(y)}

for xi, yi in zip(x,y):
    datas[yi].append(xi)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, axs = plt.subplots(5, 1)

for index, k in enumerate(datas.keys()):
    mean  = np.mean(np.stack(datas[k])**2, axis=0)
    std = np.std(np.stack(datas[k])**2, axis=0)
    axs[index].plot(mean.T[:,0], c="r", label=channels[0][0])
    # axs[index].fill_between(range(640), mean[0] - std[0], mean[0] + std[0], alpha=0.5, color="r")
    axs[index].plot(mean.T[:,1], c="b", label=channels[0][1])
    # axs[index].fill_between(range(640), mean[1] - std[1], mean[1] + std[1], alpha=0.5, color="b")
    axs[index].xaxis.set_label_position('top')
    axs[index].set_xlabel("class " + k)
fig.set_size_inches(18.5, 10.5)
plt.legend()
plt.show()








