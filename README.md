# Unsupervised Domain Adaptation for Object Detection
I use the UDA to object detection task with Cityscapes and Foggy Cityscapes data. The code is based on [ConfMix](https://github.com/giuliomattolin/ConfMix).
## Environment
- Python 3.10
- Ubuntu 20.04 with RTX 3080 GPU

## Installation
```bash
pip install -r requirements.txt
```

## Dataset Preparation (if train)
the dir should be same as yolo v5.
```
dataset
|
├── Cityscapes
│   ├── train
│   │   ├── images
│   │   │   └── 000000.jpg
│   │   └── labels
│   │       └── 000000.txt
│   └── val
│       ├── images
│       │   └── 000000.jpg
│       └── labels
│           └── 000000.txt
└── FoggyCityscapes
    ├── train
    │   ├── images
    │   │   └── 000000.jpg
    │   └── labels
    └── val
        ├── images
        │   └── 000000.jpg
        └── labels
            └── 000000.txt
```
- you should check the yaml file to make sure the path is correct (e.g. `data: ../dataset/Cityscapes2Foggy.yaml`)

## Training
### Phase 1: Model pre-train
The first phase of training consists in the pre-training of the model on the source domain. 
```bash
cd ConfMix
python train.py \
    --name cityspaces \
    --batch 4 \
    --img 1280 \
    --epochs 20 \
    --data data/Cityscapes2Foggy.yaml \
    --weights yolov5s.pt
```
- the source model will be saved in `runs/train/cityspaces/weights/best.pt`

## Phase 2: product foggycityscapes training pseudo-labels
predict on source model for foggy cityscapes training data labels
```bash
python detect.py \
    --weights runs/train/cityscapes/weights/best.pt \
    --source dataset/FoggyCityscapes/train/images/ \
    --imgsz 1280 \
    --conf-thres 0.25 \
    --iou-thres 0.5 \
    --save-txt \
    --nosave \
    --project dataset/FoggyCityscapes1/train/ \
    --name '' \
    --exist-ok 
```
- weights: the path of the source model(phase 1) weights
- source: the path of the foggy cityscapes training images
- the pseudo-labels will be saved in `dataset/FoggyCityscapes1/train/labels/`

## Phase 3: Model adaptation
The third phase of training consists in performing the adaptive learning.
```bash
python uda_train.py \
    --name cityscapes2foggy \
    --batch 4 \
    --img 1280 \
    --epochs 50 \
    --data data/Cityscapes2Foggy.yaml \
    --weights runs/train/cityscapes/weights/last.pt \
    --cache ram \
    --save-period 17 \
```
- the target model will be saved in `runs/train/cityscapes2foggy/weights/best.pt`

## Testing
use the phase 3 model to predict the test dataset
```bash
python ConfMix/detect.py \
        --weights [Phase 3 model best weight] \
        --source [test dataset path] \
        --imgsz 1280 \
        --nosave \
        --save_json \
        --save_json_path [output path] \
        --data ConfMix/data/Cityscapes2Foggy.yaml
```
- the `--weights` should be the best weight of phase 3 model, ex: `runs/train/cityscapes2foggy/weights/best.pt`
- the `--source` should be the path of the test dataset
- the predict output will be saved in `save_json_path`

