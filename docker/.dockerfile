FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

WORKDIR /home

ENV DEBIAN_FRONTEND=noninteractive
ENV MOUNT_DIR=/home/mount

# install python3.12 from ppa repository

RUN apt update -y || true && \ 
    apt upgrade -y || true && \
    apt install curl -y && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt update -y || true && \
    apt install python3.12 -y && \ 	
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# install dependencies

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install numpy scipy termcolor easydict ipdb tqdm scikit-learn imageio matplotlib tensorboard torchmetrics==0.9.3 && \
    pip install prefetch_generator colored-traceback torch-ema gdown==4.6.0 clean-fid==0.1.35 rich lmdb wandb blobfile

# install git and clone i2sb repository via https

RUN apt install -y git && \ 
    git clone https://github.com/stankevich-mipt/seismic_inversion_via_I2SB.git 

 