FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel 

RUN apt-get update && apt-get clean && \
		apt-get install python3-pip g++ gcc git wget libgl1-mesa-glx libsm6 libxrender1 \
		libfontconfig1 build-essential libglib2.0-0 libsm6 libxext6 \
		libxrender-dev tree gnuplot ghostscript texlive-extra-utils -y \
	&& exec $SHELL

RUN pip install --user pytorch-lightning==1.1.8 \
	tensorboard \
	tensorflow

RUN pip install --user yacs scikit-image tqdm opencv-python Cython numba progress matplotlib easydict \
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

RUN pip install flake8-black flake8 isort black

ENV DOCKER_RUNNING=true
WORKDIR /app