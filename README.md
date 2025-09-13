# SerbianBanknoteRecognition

A project for **detecting Serbian banknotes** using [YOLOv8](https://docs.ultralytics.com/).  
This repository contains scripts, configurations, and a dataset for training and testing a custom YOLOv8 object detection model.

## Project Structure

```bash
SerbianBanknoteRecognition/
├── configs/
│ └── data.yaml
├── data/
│ ├── raw/
│ │ ├── images/
│ │ └── labels/
│ └── processed/
│ ├── images/
│ ├── test/
│ │ ├── images/
│ │ └── labels/
│ ├── train/
│ │ ├── images/
│ │ └── labels/
│ └── val/
│ ├── images/
│ └── labels/
├── src/
│ ├── preprocess.py
│ ├── rename_labels.py
│ ├── split_dataset.py
│ ├── train_yolo.py
│ ├── test_yolo.py
│ └── test_yolo_bb.py
├── README.md
├── requirements.txt
└── yolo8n.pt
```

## How to Run

1. Clone the repo:

```bash
   git clone https://github.com/SStjepic/SerbianBanknoteRecognition
   cd SerbianBanknoteRecognition
```

2. Create a virtual environment

```bash
    python -m venv venv
    # Linux/Mac:
    source venv/bin/activate
    # Windows:
    venv\Scripts\activate
```

3. Install dependencies

```bash
    pip install -r requirements.txt
```

## Dataset

**Kaggle URL:** [Serbian Banknotes](https://www.kaggle.com/datasets/stefanstjepic/serbian-banknotes)

Download the Serbian Banknote dataset and extract files and place them into root folder.

### Dataset contents

- **270 .jpg images**: 30 images for each currently circulating Serbian banknote
- **270 .txt files**: bounding box coordinates for each banknote image - YOLO format _class x_center y_center width height_

## Preprocessing & Dataset Preparation

1. Preprocess the dataset

   Run script to resize images to 512x512

```bash
    python src/preprocess.py
```

2. Rename labels (optional)

   Labels should be in format '10RSD_front_01.txt'

```bash
    python src/rename_labels.py
```

3. Split the dataset into train/val/test

   Run this script to split dataset into train/val/test 70/15/15

```bash
    python src/split_dataset.py
```

## YOLOv8

### Config

_configs/data.yaml_

```bash
    nc: 1

    names: ["banknote"]

    train: ../data/processed/train/images
    val: ../data/processed/val/images
    test: ../data/processed/test/images
```

### Training

To start training run this command

```bash
    python src/train_yolo.py
```

Results will be saved in runs/ folder.

### Testing

Run this script to get YOLOv8 metrics:

```bash
    python src/test_yolo.py
```

Run this script to test YOLOv8 model with bounding box visualization

```bash
    python src/test_yolo_bb.py
```
