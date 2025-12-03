# Tools Detection with YOLOv8

This project trains a YOLOv8 model to detect tools (Hammer, Screwdriver, and Wrench) and provides real-time detection using your camera.

## Setup Complete! ✓

All dependencies have been installed and scripts are ready to use.

## Quick Start

### 1. Train the Model

Run the training script to train YOLOv8 on your annotated dataset:

```powershell
.venv\Scripts\python.exe train.py
```

**Training details:**
- Model: YOLOv8 nano (fastest, good for real-time detection)
- Epochs: 100 (with early stopping)
- Classes: Hammer, Screwdriver, Wrench
- Output: Trained model saved to `runs/train/tools_detection/weights/best.pt`

Training will take some time depending on your hardware:
- With GPU: ~15-30 minutes
- With CPU: ~1-2 hours

### 2. Use Your Camera for Detection

After training completes, run the camera detection script:

```powershell
.venv\Scripts\python.exe detect_camera.py
```

**Camera controls:**
- Press **'q'** to quit
- Press **'s'** to save current frame
- Press **'+'** to increase confidence threshold
- Press **'-'** to decrease confidence threshold

## Files Created

- **train.py** - Training script for YOLOv8 model
- **detect_camera.py** - Real-time camera detection script
- **data.yaml** - Dataset configuration (updated with correct paths)

## What Happens During Training

1. Downloads pretrained YOLOv8 weights
2. Trains on your annotated images
3. Validates on validation set
4. Saves best model weights
5. Generates training plots and metrics

## Tips

- **GPU Recommended**: Training is much faster with a GPU
- **Adjust Batch Size**: If you get memory errors, reduce batch size in train.py
- **Model Variants**: You can change `yolov8n.pt` to `yolov8s.pt`, `yolov8m.pt`, etc. for more accuracy (but slower)
- **Confidence Threshold**: Adjust in detect_camera.py if you get too many/few detections

## Next Steps

1. Run `train.py` to train your model
2. Wait for training to complete
3. Run `detect_camera.py` to test with your camera
4. Point your camera at hammers, screwdrivers, or wrenches!

Enjoy detecting your tools! 🔨🔧🪛
