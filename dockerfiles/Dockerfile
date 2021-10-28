FROM nvidia/cuda:11.1-devel-ubuntu20.04
# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    ca-certificates \
    libopenmpi-dev \
    sudo \
    bzip2 \
    libx11-6 \
    python3-pip g++ gcc git wget libgl1-mesa-glx libsm6 libxrender1 \
    libfontconfig1 build-essential libglib2.0-0 libsm6 libxext6 \
    libxrender-dev tree gnuplot ghostscript texlive-extra-utils sudo \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /WIBAM
WORKDIR /WIBAM

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && \ 
    chown -R user:user /WIBAM
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh

# packages
RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 pytorch-lightning \
    numpy pandas matplotlib scikit-learn tqdm jupyter \
    einops flake8-black flake8 isort loguru -c pytorch -c conda-forge 

RUN pip install yacs scikit-image tqdm opencv-python Cython numba progress matplotlib easydict \
	scipy \
	pyquaternion \
	nuscenes-devkit \
	pyyaml \
	motmetrics \
	scikit-learn \
	gs-wrap \
	gsutil \
	Cython \
	gcsfs 

# RUN git clone https://github.com/MatthewHowe/DCNv2.git /home/user/DCNv2 && pip install --user --no-cache-dir -e /home/user/DCNv2