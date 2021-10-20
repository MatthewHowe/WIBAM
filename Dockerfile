# FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

# RUN apt-get update && apt-get clean && \
# 		apt-get install g++ gcc git wget libgl1-mesa-glx libsm6 libxrender1 \
# 		libfontconfig1 build-essential libglib2.0-0 libsm6 libxext6 \
# 		libxrender-dev tree gnuplot ghostscript texlive-extra-utils -y \
# 	&& exec $SHELL

# RUN conda install python=3.7 pytorch=1.4 torchvision cudatoolkit=10.0 -c pytorch && \
# 	pip install yacs scikit-image tqdm

# FROM nvidia/cuda:11.1-devel-ubuntu18.04
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel 

RUN apt-get update && apt-get clean && \
		apt-get install python3-pip g++ gcc git wget libgl1-mesa-glx libsm6 libxrender1 \
		libfontconfig1 build-essential libglib2.0-0 libsm6 libxext6 \
		libxrender-dev tree gnuplot ghostscript texlive-extra-utils -y \
	&& exec $SHELL

RUN pip install yacs scikit-image tqdm