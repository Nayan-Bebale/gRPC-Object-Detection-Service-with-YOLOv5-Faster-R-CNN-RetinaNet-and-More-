import cv2
import torch
import time
from matplotlib import pyplot as plt
from models.calculate_latency import calculate_latency
# For R-CNN model
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# This code for Faster R-CNN model
def run_faster_rcnn(image_path):
    print(f"Running model on image: {image_path}")

    # Load Faster R-CNN model directly from torchvision
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()  # Convert HWC to CHW
    image_tensor /= 255.0  # Normalize the image
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Response Time start
    start_time = time.time()
    with torch.no_grad():  # Disable gradient calculation for inference
        results = model(image_tensor)[0]
    # Response Time end
    end_time = time.time()

    latency = calculate_latency(start_time, end_time)
    print(f"Inference Latency: {latency:.4f} seconds")

    # Process detection results
    detected_objects = []
    total_confidence = 0  # Total confidence for all detected objects
    for i in range(len(results['boxes'])):
        box = results['boxes'][i]
        score = results['scores'][i]
        label = results['labels'][i]

        if score > 0.5:  # Consider only detections with confidence > 0.5
            x1, y1, x2, y2 = box.numpy()
            detected_objects.append({
                'label': label.item(),  # Convert label tensor to scalar
                'confidence': score.item(),
                'x': int(x1),
                'y': int(y1),
                'width': int(x2 - x1),
                'height': int(y2 - y1)
            })
            total_confidence += score.item()

    # Calculate accuracy as the average confidence of detected objects
    accuracy = total_confidence / len(detected_objects) if detected_objects else 0
    print(f"Detection Accuracy: {accuracy:.2f}")

    return detected_objects, latency, accuracy
