import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import pickle

models = ["a", "b", "c", "d", "e", "f"]
PATH = "E:\\rois"


fig, axs = plt.subplots(2,3)
for index, roi_path in enumerate(os.scandir(PATH)):
    with open(os.path.join(roi_path.path, "hist.pkl"), "rb") as file:
        hist = pickle.load(file)

    axs[index].subplot(1, 2, 1, title="train accuracy")
    axs[index].plot(hist["accuracy"], label="Train")
    axs[index].plot(hist["val_accuracy"], label="Test")
    axs[index].legend(loc='lower right')
    axs[index].subplot(1, 2, 2, title="train loss")
    axs[index].plot(hist["val_loss"], label="Test")
    axs[index].plot(hist["loss"], label="Train")
    axs[index].legend(loc='upper right')

#%%
