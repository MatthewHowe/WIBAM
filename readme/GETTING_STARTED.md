
Use the dockerfile to set up the environment for training and experimentation.
```
docker build Dockerfile
```

Clone the repository
```
git clone https://github.com/MatthewHowe/WIBAM.git
cd WIBAM
```

Create a data directory
```
mkdir data
```

Download the [WIBAM dataset]() and organise the directory as follows.
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
```
pip install requirements.txt
```

Build DCNv2
```
cd src/lib/model/networks/DCNv2/
./make.sh
```

From the main WIBAM directory run training code
```
python src/main_lit.py ddd --trainset_percentage=1.0 --output_path= --load_model=models/nuScenes_3Ddetection_e140.pth --dataset=wibam --batch_size=128 --lr=7.8125e-6 --num_workers=10 --gpus=0,1,2,3
```
