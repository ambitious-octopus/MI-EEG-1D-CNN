# tensorflow base image (ubuntu 10.04, tensorflow 2.3.0, cuda 10.1, cudaNN 7.6)
FROM tensorflow/tensorflow:2.3.0-gpu
RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt install git vim wget unzip -y
RUN DEBIAN_FRONTEND="noninteractive" pip install mne wget
RUN mkdir data
WORKDIR data/
RUN touch prova.txt
RUN wget -r -N -c -np --no-verbose --show-progress --progress=bar:force:noscroll https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip
WORKDIR physionet.org/static/published-projects/eegmmidb/
RUN unzip eeg-motor-movementimagery-dataset-1.0.0.zip -d ../../../../../../data

#RUN rm eeg-motor-movementimagery-dataset-1.0.0.zip
#WORKDIR ../eegNN/
# Build: docker build -t eegnn:lstm .
# Run:

## We need git and also vim
#RUN DEBIAN_FRONTEND="noninteractive" apt install vim git -y
## wget and ca-certificates are used to get the key for ros repo
#RUN apt install -y ca-certificates wget
## Get key for ros
#RUN bash -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
#RUN wget -qO -  https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
#RUN apt update
## This ros version is pretty huge, we should evaluate if we need all these things; if not here the other versions -> http://wiki.ros.org/melodic/Installation/Ubuntu
#RUN DEBIAN_FRONTEND="noninteractive" apt install ros-melodic-desktop -y
## Add ros setup to ~/.bashrc
#RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
#RUN source ~/.bashrc
## Important lib for ros
#RUN DEBIAN_FRONTEND="noninteractive" apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential -y
#RUN apt install python-rosdep -y
#RUN rosdep init
#RUN rosdep update
## Dependecies for python
#RUN pip install real_robots pyyaml opencv-python rospkg
#
## legacy OpenGL (GL); OpenGL X extension (GLX); native platform graphics interface (EGL); ibraries for X interfacing
## source: https://medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1
#RUN apt update && apt install -y -qq --no-install-recommends libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 && rm -rf /var/lib/apt/lists/*
## Env vars for the nvidia-container-runtime.
#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
#RUN export LIBGL_ALWAYS_INDIRECT=1
#
## Run this contanier with:
## CAUTION:
##   1. change the grow-hbp path based on your needs
##   2. put --gpus all if you have a nvidia-gpu
#
## Build with:
##   grow-hbp >> docker build -t  grow-hbp:main .
## Run with:
##   >> docker run -it --net host -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=unix$DISPLAY --device /dev/dri --privileged -v /home/$USER/.Xauthority:/root/.Xauthority --gpus all -v /home/$USER/repos/grow-hbp:/grow-hbp grow-hbp:main bash
