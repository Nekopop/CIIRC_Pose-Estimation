# How to Use YOLO11/app

This guide summarizes how to use the main scripts in the `YOLO11/app` folder. It assumes the process of detecting objects and estimating poses from camera footage, saving the results as images.

---

## 1. Setup

### 1.1 Preparing the Python Environment
- Ensure that Python 3.x is installed.
- It is recommended to use a virtual environment such as `venv` or `conda`.

### 1.2 Installing Dependencies
- Install the required libraries. Below is an example (add more as needed):

```bash
pip install opencv-python-headless numpy ultralytics
```

- The main libraries used in the scripts are:

```python
import cv2
import numpy as np
from ultralytics import YOLO
import random
import glob
import os
import argparse
```

---

## 2. Execution Steps

### 2.1 Running Distance Estimation
Use the following command to perform distance estimation:

```bash
python YOLO11/app/test.py
  --model YOLO11/weight/2024-12-10-pose-yolo11s-500epoch_2024-11-27_vertical_aug.pt
  --input dataset
```

- `--model`: Specify the path to the model to be used.
- `--input`: Specify the dataset directory to be processed.

### 2.2 Evaluating Distance Estimation Accuracy
Use the following command to evaluate the accuracy of distance estimation:

```bash
python YOLO11/app/pose_eval.py
```

- This evaluation requires `rgb_frame` and `depth_frame` included in a `db3` file. Ensure these are prepared.

### 2.3 Converting the Dataset
Convert the YOLOv8pose training dataset to COCO format, making it uploadable to Roboflow:

```bash
python YOLO11/app/YOLOv8pose2COCO.py
```

- Set `input_base_dir` and `output_base_dir` in the script.

---

## Notes
- Check the necessary data and paths before running each script.
- Using a virtual environment can help prevent dependency conflicts.

With this, you are ready to correctly use the scripts in the `YOLO11/app` folder.

