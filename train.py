"""
YOLOv8 Training Script for Tools Detection
Trains a model to detect Hammer, Screwdriver, and Wrench
"""

from ultralytics import YOLO
import os

def train_model():
    # Check if GPU is available
    import torch
    device = '0' if torch.cuda.is_available() else 'cpu'
    batch_size = 16 if torch.cuda.is_available() else 8  # Smaller batch for CPU
    
    # Initialize YOLOv8 model (using nano version for faster training)
    # You can use yolov8n, yolov8s, yolov8m, yolov8l, or yolov8x
    # n=nano (fastest), s=small, m=medium, l=large, x=extra large (most accurate)
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data='data.yaml',           # Path to data configuration
        epochs=50,                  # Number of training epochs (reduced for faster training)
        imgsz=416,                  # Image size (smaller = faster)
        batch=batch_size,           # Batch size (adjust based on your GPU memory)
        name='tools_detection',     # Name for this training run
        patience=10,                # Early stopping patience (reduced)
        save=True,                  # Save checkpoints
        device=device,              # Use GPU 0 or CPU
        project='runs/train',       # Project directory
        exist_ok=True,              # Allow overwriting existing project
        pretrained=True,            # Use pretrained weights
        optimizer='auto',           # Optimizer (auto, SGD, Adam, AdamW)
        verbose=True,               # Verbose output
        seed=42,                    # Random seed for reproducibility
        deterministic=True,         # Deterministic mode
        workers=8,                  # Number of worker threads
        lr0=0.01,                   # Initial learning rate
        lrf=0.01,                   # Final learning rate factor
        momentum=0.937,             # SGD momentum
        weight_decay=0.0005,        # Weight decay
        warmup_epochs=3.0,          # Warmup epochs
        warmup_momentum=0.8,        # Warmup momentum
        box=7.5,                    # Box loss gain
        cls=0.5,                    # Classification loss gain
        dfl=1.5,                    # Distribution focal loss gain
        plots=True,                 # Save training plots
        val=True,                   # Validate during training
    )
    
    # Print training results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best weights saved to: {results.save_dir}/weights/best.pt")
    print(f"Last weights saved to: {results.save_dir}/weights/last.pt")
    print("="*60)
    
    # Validate the model
    print("\nValidating the model...")
    metrics = model.val()
    
    print(f"\nmAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return model, results

if __name__ == "__main__":
    print("Starting YOLOv8 Training for Tools Detection")
    print("Classes: Hammer, Screwdriver, Wrench")
    print("-" * 60)
    
    # Check if GPU is available
    import torch
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU detected. Training will use CPU (slower).")
        print("To use CPU, the script will automatically adjust.")
    
    print("-" * 60 + "\n")
    
    try:
        model, results = train_model()
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        raise
