# YOLOv11 Pose Estimation Project

## Project Purpose
- Training and evaluation of YOLOv11 model for pose estimation of Bricks
- Preprocessing and augmentation of the dataset
- Implementation of distance estimation

## Code Structure and Functions

### 

pose.py


Distance estimation for Bricks
- **Function**: Keypoint detection and 3D pose estimation
- **Usage**:
```bash
python pose.py
```
- **Input**: Image resized to 640x640
- **Output**: Visualization of keypoint detection and distance estimation results

### 

train.py


Training of YOLOv11 model
- **Function**: Training of pose estimation model
- **Usage**:
```bash
python train.py
```
- **Output**: 
  - Training log (date-time.txt)
  - Trained model

### 

save_command_output.py


Save command line input and training results
- **Function**: Save command line input and training results in the same file
- **Usage**:
```bash
python save_command_output.py
```
- **Output**: 
  - File containing command line input and training results (`YYYY-MM-DD_HH-MM-SS.txt`)

## Dataset Structure
```
dataset/
├── YOLOv11/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   └── test/
└── YOLOv11_augmented/
    └── train/
        ├── images/
        └── labels/
```

## Required Packages
```bash
pip install ultralytics opencv-python numpy matplotlib
```

## Notes
- The order of keypoints is clockwise starting from the bottom left
- The dataset annotation format conforms to YOLOv8 format