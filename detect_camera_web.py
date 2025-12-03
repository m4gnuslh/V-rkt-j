"""
Real-time Tools Detection with Web Interface
Detects Hammer (0), Screwdriver (1), and Wrench (2) and displays on webpage
"""

from ultralytics import YOLO
import cv2
import os
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import threading
import time

app = Flask(__name__, 
            template_folder='web',
            static_folder='web')
CORS(app)

# Global variables for sharing data between threads
latest_detection = {"tool": None, "name": None, "confidence": 0, "timestamp": 0}
detection_lock = threading.Lock()
frame_data = {"frame": None}
frame_lock = threading.Lock()
NO_DETECTION_TIMEOUT = 2.0  # seconds before showing position 3

# Tool mapping
TOOL_MAP = {
    0: "Hammer",
    1: "Screwdriver", 
    2: "Wrench"
}

def detect_with_camera(model_path='runs/train/tools_detection/weights/best.pt', 
                       camera_id=0,
                       confidence_threshold=0.5):
    """
    Run real-time detection using webcam and update global detection data
    """
    global latest_detection, frame_data
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
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
    
    print(f"\nCamera detection started!")
    print(f"Web interface: http://localhost:5000")
    print(f"Confidence threshold: {confidence_threshold}")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Run detection
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Update frame data for video stream
            with frame_lock:
                frame_data["frame"] = annotated_frame.copy()
            
            # Process detections
            detections = results[0].boxes
            if len(detections) > 0:
                # Get the detection with highest confidence
                best_detection = None
                best_conf = 0
                
                for detection in detections:
                    conf = float(detection.conf[0])
                    cls = int(detection.cls[0])
                    
                    if conf > best_conf:
                        best_conf = conf
                        best_detection = cls
                
                if best_detection is not None:
                    tool_name = TOOL_MAP.get(best_detection, "Unknown")
                    
                    with detection_lock:
                        latest_detection = {
                            "tool": best_detection,
                            "name": tool_name,
                            "confidence": best_conf,
                            "timestamp": time.time()
                        }
                    
                    print(f"Detected: {tool_name} (class {best_detection}) - Confidence: {best_conf:.2f}")
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.03)  # ~30 FPS
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        print("Detection stopped.")

def generate_frames():
    """Generate frames for video streaming"""
    while True:
        with frame_lock:
            if frame_data["frame"] is not None:
                frame = frame_data["frame"].copy()
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection')
def get_detection():
    """API endpoint to get latest detection"""
    with detection_lock:
        return jsonify(latest_detection)

def start_detection_thread(model_path, camera_id, confidence_threshold):
    """Start detection in a separate thread"""
    detection_thread = threading.Thread(
        target=detect_with_camera,
        args=(model_path, camera_id, confidence_threshold),
        daemon=True
    )
    detection_thread.start()

if __name__ == "__main__":
    print("="*60)
    print("Real-time Tools Detection Web Interface")
    print("Classes: 0=Hammer, 1=Screwdriver, 2=Wrench")
    print("="*60 + "\n")
    
    # Default model path
    model_path = 'runs/train/tools_detection/weights/best.pt'
    
    # Check if trained model exists
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at: {model_path}")
        print("\nPlease train the model first:")
        print("  python train.py")
        exit(1)
    
    # Start detection thread
    start_detection_thread(
        model_path=model_path,
        camera_id=0,
        confidence_threshold=0.5
    )
    
    # Start Flask web server
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
