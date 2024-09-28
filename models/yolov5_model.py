import cv2
import torch
import numpy as np
import time
import os 
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F
from models.calculate_latency import calculate_latency



# This code for YoLoV5 model
def run_yolov5(image_path):
    print(f"Running model on image: {image_path}")

    # Load YOLOv5 model via torch.hub (simplified loading)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Response Time start
    start_time = time.time()
    # Perform inference on the image
    results = model(image_path)  # Automatically handles resizing, normalization, etc.
    # Response Time end
    end_time = time.time()

    latency = calculate_latency(start_time, end_time)
    print(f"Inference Latency: {latency:.4f} seconds")
    # Process detection results
    detected_objects = []
    total_confidence = 0  # Total confidence for all detected objects
    for *xyxy, conf, cls in results.xyxy[0]:  # results.xyxy[0] gives bbox, conf, class
        x1, y1, x2, y2 = map(int, xyxy)  # Coordinates are already in image format
        detected_objects.append({
            'label': model.names[int(cls)],  # Convert class index to label
            'confidence': float(conf),
            'x': x1,
            'y': y1,
            'width': x2 - x1,
            'height': y2 - y1
        })
        total_confidence += float(conf)

    # Calculate accuracy as the average confidence of detected objects
    accuracy = total_confidence / len(detected_objects) if detected_objects else 0
    print(f"Detection Accuracy: {accuracy:.2f}")

    return detected_objects, latency, accuracy