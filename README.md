## Weakly Supervised Training of Monocular 3D Object Detectors Using Wide Baseline Multi-view Traffic Camera Data


## Installation and setup

Use the dockerfile to set up the environment for training and experimentation.
```docker build Dockerfile```

Clone the repository
```git clone https://github.com/MatthewHowe/WIBAM.git```
```cd WIBAM```
Create a data directory
```mkdir data```

Download the [WIBAM dataset](www.google.com) and organise the directory as follows.
```
WIBAM
│   README.md
│   requirements.txt    
│   ...
|
└───data
│   └───wibam
|       └───calib
│       └───annotations
│       └───frames
│       └───image_sets
|       └───models
│   
└───src
    └───lib
    └───tools
    |   ...
```

Install the requirements
```pip install requirements.txt```

Build DCNv2
```cd src/lib/model/networks/DCNv2/ && ./make.sh && cd ../../../../../```

Run training code
```python src/main_lit.py ddd --trainset_percentage=1.0 --output_path=gs://aiml-reid-casr-data/lightning_experiments --load_model=models/nuScenes_3Ddetection_e140.pth --dataset=wibam --batch_size=128 --lr=7.8125e-6 --num_workers=10 --gpus=0,1,2,3```

## License
This repo is a modified clone of CenterTrack https://github.com/xingyizhou/CenterTrack.
CenterTrack is developed upon [CenterNet](https://github.com/xingyizhou/CenterNet). Both codebases are released under MIT License themselves. Some code of CenterNet are from third-parties with different licenses, please check the CenterNet repo for details. In addition, this repo uses [py-motmetrics](https://github.com/cheind/py-motmetrics) for MOT evaluation and [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) for nuScenes evaluation and preprocessing. See [NOTICE](NOTICE) for detail. Please note the licenses of each dataset. Most of the datasets we used in this project are under non-commercial licenses.