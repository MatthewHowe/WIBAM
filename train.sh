export GOOGLE_APPLICATION_CREDENTIALS="/home/matt/WIBAM/WIBAM/data/data-manager.json"
python src/main_lit.py ddd --load_model=data/models/nuScenes_3Ddetection_e140.pth \
                --dataset=nuscenes --batch_size=80 \
                --mixed_dataset=wibam --mixed_batchsize=12 \
                --gpus=0,1,2,3 --num_workers=12 \
                --lr=4.48e-6 \
                --output_path=gs://aiml-reid-casr-data/lightning_experiments
exit 0