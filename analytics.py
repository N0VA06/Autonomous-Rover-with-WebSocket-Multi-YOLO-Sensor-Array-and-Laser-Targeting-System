import cv2
import threading
import time
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
import torch
from ultralytics import YOLO
import numpy as np

# Configure video source to receive stream from external source on port 8000
stream_source = "http://localhost:8000/video_feed"  # Change to the appropriate source IP
# Use a different port for our server to avoid conflicts
server_port = 8080

frame_lock = threading.Lock()
global_frame = None
processed_frame = None
stop_threads = False
capture_thread = None
processing_thread = None

# Load YOLO models
def load_models():
    models = []
    model_paths = [
        "path/to/model1.pt",  # Replace with actual paths to your models
        "path/to/model2.pt", 
        "path/to/model3.pt",
        "path/to/model4.pt"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = YOLO(model_path)
                models.append(model)
                print(f"Loaded model: {model_path}")
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
        else:
            print(f"Model not found: {model_path}")
    
    return models

# Initialize models
yolo_models = []

def capture_frames():
    """Continuously capture frames from the external stream source"""
    global global_frame, stop_threads
    
    # Initialize stream capture
    print(f"Connecting to stream at {stream_source}")
    cap = cv2.VideoCapture(stream_source)
    
    # Check if the stream opened successfully
    if not cap.isOpened():
        print(f"Error: Could not connect to stream at {stream_source}")
        print("Retrying connection every 5 seconds...")
        
        # Keep trying to connect until successful or stopped
        while not stop_threads and not cap.isOpened():
            time.sleep(5)
            print("Attempting to connect to stream...")
            cap = cv2.VideoCapture(stream_source)
            
        if stop_threads:
            return
            
        if not cap.isOpened():
            print("Failed to connect after multiple attempts. Exiting.")
            return
    
    print(f"Stream connected successfully. Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    connection_errors = 0
    max_errors = 10  # Maximum consecutive errors before reconnecting
    
    while not stop_threads:
        ret, frame = cap.read()
        
        if not ret:
            connection_errors += 1
            print(f"Error: Failed to capture frame ({connection_errors}/{max_errors})")
            
            # If we have too many consecutive errors, try to reconnect
            if connection_errors >= max_errors:
                print("Too many errors, reconnecting to stream...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(stream_source)
                connection_errors = 0
                if not cap.isOpened():
                    print(f"Failed to reconnect to {stream_source}")
                    time.sleep(5)
                    continue
            
            time.sleep(0.5)
            continue
        
        # Reset error counter on successful frame capture
        connection_errors = 0
        
        # Store the frame with thread safety
        with frame_lock:
            global_frame = frame.copy()
        
        # Control capture rate
        time.sleep(0.01)
    
    cap.release()
    print("Stream capture stopped.")

def process_frames():
    """Process frames with YOLO models"""
    global global_frame, processed_frame, stop_threads, yolo_models
    
    # Wait for models to be loaded
    if not yolo_models:
        print("No YOLO models loaded, waiting...")
        time.sleep(5)
        if not yolo_models:
            print("Still no models loaded, will process without models")
    
    while not stop_threads:
        # Wait until we have a frame
        if global_frame is None:
            time.sleep(0.1)
            continue
        
        # Get the current frame with thread safety
        with frame_lock:
            if global_frame is not None:
                frame = global_frame.copy()
            else:
                continue
        
        # Process with each model if available
        if yolo_models:
            # Create a copy for visualization
            result_frame = frame.copy()
            
            # Process with each model
            for i, model in enumerate(yolo_models):
                # Different color for each model
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                color = colors[i % len(colors)]
                
                # Run inference
                try:
                    results = model(frame, conf=0.25)
                    
                    # Draw bounding boxes
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            
                            # Get confidence and class
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            
                            # Get class name
                            class_name = model.names[cls] if cls in model.names else f"Class {cls}"
                            
                            # Draw bounding box
                            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Add label
                            label = f"{class_name} {conf:.2f}"
                            cv2.putText(result_frame, label, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                except Exception as e:
                    print(f"Error processing with model {i}: {e}")
            
            # Store the processed frame
            with frame_lock:
                processed_frame = result_frame
        else:
            # If no models are available, just pass the original frame
            with frame_lock:
                processed_frame = frame
        
        # Control processing rate
        time.sleep(0.05)  # ~20 FPS for processing

def generate_frames():
    """Generate MJPEG frames for streaming"""
    global processed_frame
    
    while not stop_threads:
        # Wait until we have a processed frame
        if processed_frame is None:
            time.sleep(0.1)
            continue
        
        # Get the current processed frame with thread safety
        with frame_lock:
            if processed_frame is not None:
                frame = processed_frame.copy()
            else:
                continue
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if not ret:
            continue
            
        # Convert to bytes and yield for the HTTP response
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Control streaming rate
        time.sleep(0.033)  # ~30 FPS

def start_threads():
    """Start the frame capture and processing threads"""
    global stop_threads, capture_thread, processing_thread, yolo_models
    
    stop_threads = False
    
    # Load the YOLO models
    yolo_models = load_models()
    
    # Start capture thread
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()
    
    # Start processing thread
    processing_thread = threading.Thread(target=process_frames)
    processing_thread.daemon = True
    processing_thread.start()
    
    print("Capture and processing threads started")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup: start the threads
    start_threads()
    print("Application started")
    
    yield  # This is where FastAPI runs
    
    # Shutdown: stop the threads
    global stop_threads
    stop_threads = True
    
    # Wait for threads to finish
    if capture_thread:
        capture_thread.join(timeout=1.0)
    if processing_thread:
        processing_thread.join(timeout=1.0)
    
    print("Application stopping")

# Create FastAPI app with lifespan handler
app = FastAPI(title="YOLO Webcam Stream", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page with the video stream"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raspberry Pi YOLO Stream Processor</title>
        <style>
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Raspberry Pi YOLO Stream Processor</h1>
            <div class="stream-info">
                Input Stream: <code>{stream_source}</code><br>
                Output Port: <code>{server_port}</code>
            </div>
            <div class="status">Stream processing is active</div>
            <div class="model-info">
                <p>Using 4 custom fine-tuned YOLO models</p>
                <ul>
                    <li>Model 1: <span class="color-box" style="display:inline-block;width:12px;height:12px;background-color:rgb(0,255,0);"></span> Green boxes</li>
                    <li>Model 2: <span class="color-box" style="display:inline-block;width:12px;height:12px;background-color:rgb(255,0,0);"></span> Red boxes</li>
                    <li>Model 3: <span class="color-box" style="display:inline-block;width:12px;height:12px;background-color:rgb(0,0,255);"></span> Blue boxes</li>
                    <li>Model 4: <span class="color-box" style="display:inline-block;width:12px;height:12px;background-color:rgb(255,255,0);"></span> Yellow boxes</li>
                </ul>
            </div>
            <div class="video-container">
                <img src="/video_feed" alt="YOLO Processed Stream">
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video_feed")
async def video_feed():
    """Route for the video feed"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=server_port, reload=False, lifespan="on")