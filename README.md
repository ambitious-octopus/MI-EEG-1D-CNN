<h1 align="center" > A 1D CNN for high accuracy classification and transfer learning in motor 
imagery EEG-based brain-computer interface </h1>


<p align="center">
<a href="https://github.com/Kubasinska/MI-EEG-1D-CNN/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/Kubasinska/MI-EEG-1D-CNN"></a>
<a href="https://github.com/Kubasinska/MI-EEG-1D-CNN/issues"><img alt="GitHub issues" src="https://img.shields.io/github/forks/kubasinska/MI-EEG-1D-CNN?style=social"></a>
<a href="https://github.com/Kubasinska/MI-EEG-1D-CNN/issues"><img alt="GitHub issues" src="https://img.shields.io/github/forks/kubasinska/MI-EEG-1D-CNN?style=social"></a>
<a href="https://github.com/Kubasinska/MI-EEG-1D-CNN/issues"><img alt="GitHub issues" src="https://img.shields.io/github/stars/kubasinska/MI-EEG-1D-CNN?style=social"></a>
<a href="https://github.com/Kubasinska/MI-EEG-1D-CNN/issues"><img alt="GitHub issues" src="https://img.shields.io/github/watchers/kubasinska/MI-EEG-1D-CNN?style=social"></a>
<a href="https://github.com/Kubasinska/MI-EEG-1D-CNN/blob/master/LICENSE.txt"><img alt="GitHub issues" src="https://img.shields.io/github/license/Kubasinska/MI-EEG-1D-CNN"></a>
<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FKubasinska%2FMI-EEG-1D-CNN"><img alt="GitHub issues" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2FKubasinska%2FMI-EEG-1D-CNN"></a>
<a href="https://twitter.com/FrancescoMattio"><img alt="GitHub issues" src="https://img.shields.io/twitter/follow/FrancescoMattio?style=social"></a>
<a href=""><img alt="GitHub issues" src="https://img.shields.io/github/languages/top/kubasinska/MI-EEG-1D-CNN"></a>
<a href=""><img alt="GitHub issues" src="https://img.shields.io/github/languages/code-size/kubasinska/MI-EEG-1D-CNN"></a>
<a href=""><img alt="GitHub issues" src="https://img.shields.io/github/repo-size/kubasinska/MI-EEG-1D-CNN"></a>
<a href=""><img alt="GitHub issues" src="https://img.shields.io/github/v/release/kubasinska/MI-EEG-1D-CNN"></a>
<a href=""><img alt="GitHub issues" src="https://img.shields.io/github/commit-activity/y/kubasinska/MI-EEG-1D-CNN"></a>
<a href=""><img alt="GitHub issues" src="https://img.shields.io/github/last-commit/kubasinska/MI-EEG-1D-CNN"></a>
<a href=""><img alt="GitHub issues" src="https://img.shields.io/github/release-date/kubasinska/MI-EEG-1D-CNN"></a>
<a href="https://doi.org/10.1088/1741-2552/ac4430"><img alt="GitHub issues" src="https://badges.aleen42.com/src/docker.svg"></a>
<a href=""><img alt="GitHub issues" src="https://img.shields.io/badge/Tensorflow%20Version-2.3.0-blueviolet"></a>
<a href="https://doi.org/10.1088/1741-2552/ac4430"><img alt="GitHub issues" src="https://badges.aleen42.com/src/tensorflow.svg"></a>
<a href=""><img alt="GitHub issues" src="https://img.shields.io/badge/Python%20Version-3.8.3-blueviolet"></a>
<a href="https://doi.org/10.1088/1741-2552/ac4430"><img alt="GitHub issues" src="https://badges.aleen42.com/src/python.svg"></a>
<a href="https://doi.org/10.1088/1741-2552/ac4430"><img alt="GitHub issues" src="https://img.shields.io/badge/DOI-https%3A%2F%2Fdoi.org%2F10.1088%2F1741--2552%2Fac4430-success"></a>

<h3 align="center" > What is this? </h3>
<div> A new Brain Computer Interface approach based on a 10-layer one-dimensional convolution 
neural 
network (1D-CNN) to classify five brain states (four Motor Imagery classes plus a 'baseline' class) 
using a data augmentation algorithm and a limited number of EEG channels. The model tested with 
the 'EEG Motor Movement/Imagery Dataset' outperforms the current state-of-the-art models by 
achieving a 99.38% accuracy at the group level. In addition, the transfer learning approach we 
present achieves an average accuracy of 99.46%.</div> 

<h3 align="center" > Reference paper </h3>
 <div> Mattioli F, Porcaro C, Baldassarre G. A 1D CNN for high accuracy classification and transfer learning in motor imagery EEG-based brain-computer interface. J Neural Eng. 2022 Jan 6;18(6). doi: 10.1088/1741-2552/ac4430. PMID: 34920443. </div> 

<h3 align="center" > Usage </h3>
<div align="center">
<img src="docs/hopefullnet.png" alt="HopefullNet">
</div>

<h4 align="center" > Directory structure </h4>

```
.
└── MI-EEG-1D-CNN/
    ├── data_processing/ # A module with useful functions
    │   └── general_processor.py
    ├── dataset_generator/ # Script that generates the dataset
    │   └── generator.py
    ├── docker/ # All the useful things to build the contanier
    │   ├── Dockerfile
    │   ├── environment.yml
    │   └── generator.py
    ├── docs/ # Scripts for inference and plotting
    │   ├── inference
    │   └── environment.yml
    ├── models/ # Scripts for training networks
    │   ├── hand_test
    │   └── transfer
    └── model_set/ # A module with all models
        └── models.py
```

<h4 align="center" > Install the dependencies </h4>
In order to train the network you need to install dependencies. There are two ways to install dependencies: (1) Using a docker container (recommended), (2) 
Using a python environment. An NVIDIA GPU with at least 6 GB is also recommended. The network was trained with an NVIDIA RTX 2060 and an NVIDIA TESLA P100, the former taking about 30 minutes to train, the latter about 15 minutes.

<h5 align="center" > Using a docker container (recommended) </h5>

<a href="https://www.docker.com/resources/what-container">What is a docker contanier? </a>
<div>
In addition to docker we also recommend installing <a href="https://developer.nvidia.
com/nvidia-container-runtime#:~:text=NVIDIA%20Container%20Runtime%20is%20a,
desktop%2C%20cloud%20or%20data%20centers.">NVIDIA Container Runtime (v2) </a> which allows you 
to create contaniers that use your nvidia GPU. <a href="https://docs.nvidia.
com/datacenter/cloud-native/container-toolkit/install-guide.html">NVIDIA Container Runtime (v2) 
installation guide.
 </a>
The contanier building process has been tested on Ubuntu 20.04. If you have a Windows machine you can try Windows Linux Subsystem but it has not been tested yet (In case someone tests it please tell us about your experience by opening an issue).
</div>

Assuming you have docker installed. Building the contanier is simple. 
1. Clone this repository `git clone https://github.com/Kubasinska/MI-EEG-1D-CNN.git`
2. Open a shell and `cd` into the docker folder `MI-EEG-1D-CNN/docker`
3. Run `docker build -t eegcnn .` Depending on the permissions set on your machine you 
   may need to run this command as administrator by adding `sudo` at the beginning. This command 
   will build a new contanier called eegbci. This may take a few minutes. During the build 
   process the original dataset is also downloaded and the dataset is also generated as described within the methods section of the original paper. Once the contanier is built you already have everything ready.
4. Run the contanier to check that everything has been installed correctly. To launch the 
   contanier go to the working directory of the `MI-EEG-1D-CNN` repository. You can launch the 
   contanier in several ways, here we recommend 2, one that allows you to plot graphs and one that 
   does not.
   1. I want to see graphs. If you want to see the graphs you must allow the contanier to access your screen. To do this run `xhost +local:root`, this basically expose your xhost so that container can render to the correct display by reading and writing though the X11 unix socket. Then, from the working directory of the repository launch: `docker run -it --gpus all -v $(pwd):/workspace -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=unix$DISPLAY --device /dev/dri --privileged -v /home/$USER/. Xauthority:/root/.Xauthority eegcnn bash`. This will open a bash shell inside the contanier and you are good to go! If you don't have an NVIDIA gpu or you don't have NVIDIA Container Runtime (v2) installed, omit `--gpus all`. When you close the contanier remember to launch `xhost -local:root`.
   2. I don't care about graphs. In this case it is much easier! Just run: `docker run -it 
      --gpus all -v $(pwd):/workspace eegcnn bash` and you're in.
5. All code paths are made so that using the container you don't have to change anything. Inside 
   the directory `/dataset` in the contanier you find 3 folders `/dataset/original`, the 
   original dataset data, `/dataset/paper` the data generated through the method described in the paper with the script `dataset_generator/generator.py`, and a third folder 
   `/dataset/saved_models`, where the trained models are automatically saved.

<h5 align="center" > Using a python environment. </h5>

This procedure is simpler but can create dependency issues based on your machine or operating 
system. Using this procedure you must also manually download the original dataset and generate 
the dataset that is used in the paper. You also need Anaconda or Miniconda to create a separate 
python environment. The following guide assumes you have Anaconda or Miniconda installed on your system.
1. Open a terminal and `cd` into the folder `MI-EEG-1D-CNN/docs`. Run `conda env create -f 
   environment.yml`. These create a new python environment called `eeg` that contains all 
   necessary dependencies. The only dependencies that are missing are CUDA and cuDNN which are 
   needed by tensorflow to use your GPU. If you don't have an NVIDIA GPU, go ahead. If you have 
   an NVIDIA GPU you need to install CUDA 10.1 and cuDNN 7.6 (be careful with the version, CUDA 
   10.2 or 10.0 is not good, you need 10.1, same for cuDNN). Please refer to the [official NVIDIA 
   website for installation](https://developer.nvidia.com/cudnn), [here is a guide for windows](https://www.nientepanico.org/2021/02/20/humans-guide-to-tensorflow-gpu/).
2. Download the EEG Motor Movement/Imagery Dataset [here](https://physionet.org/content/eegmmidb/1.0.0/). The dataset is quite large (3.4 GB), it will take a while. Once downloaded, extract it. If 
   you have wget you can download it from the terminal with the command. `wget -r -N -c -np 
   https://physionet.org/files/eegmmidb/1.0.0/`.
3. Generate the dataset, this procedure simply takes the raw data and breaks it into the input dimension of the neural network. 
   Use the script `MI-EEG-1D-CNN/dataset_generator/generator.py`. Change the dataset path to the 
   path of the dataset you downloaded and you are ready! Don't forget to run the script with the new conda environment you downloaded.























