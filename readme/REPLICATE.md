# Quickly replicate the results in this paper

Clone the repo
```
git clone https://github.com/MatthewHowe/WIBAM.git
cd WIBAM
```

Download the demo dataset from [Zenodo](https://zenodo.org/record/5609988#.YXsWkjpBWxs).
Put it in the WIBAM directory under data - as follows, rename it to wibam.
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

Pull the docker image
```
docker pull matthewhowe/wibam
```

Run the docker image
```
make run
```

From within the docker image run
```
python src/main_lit.py ddd --dataset=wibam --load_model=data/wibam/models/wibam.ckpt --batch_size=1 --save_video --gpus=0 --num_workers=1 --test_only
```