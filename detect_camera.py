"""
Real-time Tools Detection using Camera
Detects Hammer, Screwdriver, and Wrench in real-time using webcam
"""

from ultralytics import YOLO
import cv2
import os

def detect_with_camera(model_path='runs/train/tools_detection/weights/best.pt', 
                       camera_id=0,
                       confidence_threshold=0.5):
    """
    Run real-time detection using webcam
    
    Args:
        model_path: Path to trained model weights
        camera_id: Camera device ID (0 for default webcam)
        confidence_threshold: Minimum confidence for detection (0-1)
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running: python train.py")
        return
    
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Open camera
    print(f"Opening camera (device {camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Camera opened: {width}x{height} @ {fps}fps")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press '+' to increase confidence threshold")
    print("  - Press '-' to decrease confidence threshold")
    print(f"\nStarting detection (confidence threshold: {confidence_threshold})...\n")
    
    frame_count = 0
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Add info overlay
            info_text = f"Frame: {frame_count} | Threshold: {confidence_threshold:.2f} | Press 'q' to quit"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Count detections
            detections = results[0].boxes
            if len(detections) > 0:
                detection_text = f"Detected: {len(detections)} object(s)"
                cv2.putText(annotated_frame, detection_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Tools Detection - YOLOv8', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f'detection_frame_{frame_count}.jpg'
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved frame to: {filename}")
            elif key == ord('+') or key == ord('='):
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                print(f"Confidence threshold increased to: {confidence_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                confidence_threshold = max(0.05, confidence_threshold - 0.05)
                print(f"Confidence threshold decreased to: {confidence_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")
        print("Detection stopped.")

def list_available_cameras():
    """List available camera devices"""
    print("Checking for available cameras...")
    available_cameras = []
    
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    if available_cameras:
        print(f"Available cameras: {available_cameras}")
    else:
        print("No cameras found")
    
    return available_cameras

if __name__ == "__main__":
    print("="*60)
    print("Real-time Tools Detection using YOLOv8")
    print("Classes: Hammer, Screwdriver, Wrench")
    print("="*60 + "\n")
    
    # List available cameras
    cameras = list_available_cameras()
    
    print()
    
    # Default model path
    model_path = 'runs/train/tools_detection/weights/best.pt'
    
    # Check if trained model exists
    if not os.path.exists(model_path):
        print(f"Warning: Trained model not found at: {model_path}")
        print("\nPlease train the model first:")
        print("  python train.py")
        print("\nOr specify a different model path in the script.")
    else:
        # Start detection
        try:
            detect_with_camera(
                model_path=model_path,
                camera_id=0,  # Use default camera (change if needed)
                confidence_threshold=0.5
            )
        except Exception as e:
            print(f"\nError during detection: {e}")
            raise
