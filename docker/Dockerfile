FROM ubuntu:20.04
RUN apt-get update
RUN apt-get update && apt-get install -y apt-transport-https
RUN DEBIAN_FRONTEND="noninteractive" apt-get install vim wget nano curl tk-dev git git-lfs ca-certificates -y
WORKDIR /root/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN /root/Miniconda3-py38_4.10.3-Linux-x86_64.sh -b && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && /root/miniconda3/bin/conda clean -afy
RUN /root/miniconda3/bin/conda init
COPY environment.yml /root/environment.yml
RUN /root/miniconda3/bin/conda env create -f /root/environment.yml && /root/miniconda3/bin/conda clean -afy
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
WORKDIR /
RUN mkdir dataset
RUN mkdir workspace
WORKDIR /dataset

RUN wget -O output.tar.gz "https://onedrive.live.com/download?cid=AF37652B85ECAA2E&resid=AF37652B85ECAA2E%21271719&authkey=AGh4toRof0SHV0M"

RUN tar -xf output.tar.gz
RUN rm output.tar.gz

WORKDIR /dataset
RUN mkdir paper
RUN mkdir saved_models
COPY data_processing /home/data_processing
COPY generator.py /home/generator.py

RUN echo 'conda activate eeg' >> ~/.bashrc
WORKDIR /home
RUN /root/miniconda3/bin/conda run -n eeg python generator.py
WORKDIR /workspace

# Bulild -> docker build -t eegcnn .
# Run with -> docker run -it --gpus all -v $(pwd):/workspace eegcnn bash