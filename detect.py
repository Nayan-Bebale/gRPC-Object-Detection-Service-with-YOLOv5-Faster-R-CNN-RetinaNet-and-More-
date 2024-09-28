import cv2
import numpy as np
import time
import os 
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    KeypointRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
    SSDLite320_MobileNet_V3_Large_Weights
)
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "TV",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Function to load the specified model
def load_model(model_name="fasterrcnn"):
    if model_name == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "fasterrcnn_mobilenet":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
    elif model_name == "fasterrcnn_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    elif model_name == "maskrcnn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "maskrcnn_v2":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    elif model_name == "keypointrcnn":
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "ssd":
        model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    elif model_name == "ssdlite":
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.eval()  # Set the model to evaluation mode
    return model

# Function to calculate latency and response time
def calculate_latency(start_time, end_time):
    return end_time - start_time

# Function to run the model and perform object detection
def run_model(image_path, model_name="fasterrcnn"):
    # Load the specified model
    model = load_model(model_name)

    # Load and preprocess the image
    if image_path.startswith('http'):  # If the input is a URL
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    print(image_tensor.shape)
    start_time = time.time()
    
    # Perform object detection
    with torch.no_grad():
        predictions = model(image_tensor)

    # Stop the inference time
    end_time = time.time()
    latency = calculate_latency(start_time, end_time)

    # Process the predictions
    detected_objects = []
    accuracy = 0.0  # Placeholder for accuracy calculation

    if 'scores' in predictions[0] and 'labels' in predictions[0] and 'boxes' in predictions[0]:
        for i, score in enumerate(predictions[0]['scores'].tolist()):  # Convert to list for compatibility
            if score > 0.5:  # Confidence threshold
                box = predictions[0]['boxes'][i].tolist()  # Convert tensor to list
                label_id = predictions[0]['labels'][i].item()  # Convert to Python int
                
                # Convert label_id to string using the COCO class names
                label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]  # Convert to string class name
                # print(f"Detected object label: {label_name} (confidence: {score})")

                detected_objects.append({
                    'label': label_name,  # Store the string label name
                    'confidence': score,
                    'x': box[0],  # x_min
                    'y': box[1],  # y_min
                    'width': box[2] - box[0],  # width
                    'height': box[3] - box[1]  # height
                })
                accuracy += score  # Sum the confidence for accuracy calculation

    # Calculate average accuracy
    accuracy /= len(detected_objects) if detected_objects else 1  # Avoid division by zero
    return detected_objects, latency, accuracy
