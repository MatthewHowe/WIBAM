
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

Download the [WIBAM dataset](https://universityofadelaide.box.com/s/73gccpx603i43iod7260lth00m4i3v4h) and organise the directory as follows.
The images in the dataset are in individual folders, extract them into the frames directory and rename them 0, 1, 2 and 3. Rename 'wibam_no_frames' to wibam.

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
|       |   └───0
|       |   └───1
|       |   └───2
|       |   └───3
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
