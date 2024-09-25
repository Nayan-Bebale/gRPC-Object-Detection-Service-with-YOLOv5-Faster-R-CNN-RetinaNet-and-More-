import cv2
import torch
import numpy as np
import time
import os 
from matplotlib import pyplot as plt

# For R-CNN model
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# For RetinaNet model
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image


# Function to calculate latency and response time
def calculate_latency(start_time, end_time):
    return end_time - start_time

# For genral use
# def plot_results(image, detected_objects):
#     for obj in detected_objects:
#         label = obj['label']
#         confidence = obj['confidence']
#         x, y, w, h = obj['x'], obj['y'], obj['width'], obj['height']

#         # Draw bounding box
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Draw label and confidence
#         text = f'{label} {confidence:.2f}'
#         cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     return image


# For RetinaNet model
def plot_results(image, detected_objects):
    for obj in detected_objects:
        # Ensure that coordinates are integers
        x = int(obj['x'])
        y = int(obj['y'])
        w = int(obj['width'])
        h = int(obj['height'])
        
        # Draw rectangle around detected objects
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Optionally put label and confidence text on the image
        label = f"Label: {obj['label']}, Conf: {obj['confidence']:.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

# This code for YoLoV5 model
# def run_model(image_path):
#     print(f"Running model on image: {image_path}")

#     # Load YOLOv5 model via torch.hub (simplified loading)
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
#     # Response Time start
#     start_time = time.time()
#     # Perform inference on the image
#     results = model(image_path)  # Automatically handles resizing, normalization, etc.
#     # Response Time end
#     end_time = time.time()

#     latency = calculate_latency(start_time, end_time)
#     print(f"Inference Latency: {latency:.4f} seconds")
#     # Process detection results
#     detected_objects = []
#     total_confidence = 0  # Total confidence for all detected objects
#     for *xyxy, conf, cls in results.xyxy[0]:  # results.xyxy[0] gives bbox, conf, class
#         x1, y1, x2, y2 = map(int, xyxy)  # Coordinates are already in image format
#         detected_objects.append({
#             'label': model.names[int(cls)],  # Convert class index to label
#             'confidence': float(conf),
#             'x': x1,
#             'y': y1,
#             'width': x2 - x1,
#             'height': y2 - y1
#         })
#         total_confidence += float(conf)

#     # Calculate accuracy as the average confidence of detected objects
#     accuracy = total_confidence / len(detected_objects) if detected_objects else 0
#     print(f"Detection Accuracy: {accuracy:.2f}")

#     return detected_objects, latency, accuracy

# This code for Faster R-CNN model
# def run_model(image_path):
#     print(f"Running model on image: {image_path}")

#     # Load Faster R-CNN model directly from torchvision
#     model = fasterrcnn_resnet50_fpn(pretrained=True)
#     model.eval()  # Set the model to evaluation mode
    
#     # Load and preprocess the image
#     image = cv2.imread(image_path)
#     image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()  # Convert HWC to CHW
#     image_tensor /= 255.0  # Normalize the image
#     image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

#     # Response Time start
#     start_time = time.time()
#     with torch.no_grad():  # Disable gradient calculation for inference
#         results = model(image_tensor)[0]
#     # Response Time end
#     end_time = time.time()

#     latency = calculate_latency(start_time, end_time)
#     print(f"Inference Latency: {latency:.4f} seconds")

#     # Process detection results
#     detected_objects = []
#     total_confidence = 0  # Total confidence for all detected objects
#     for i in range(len(results['boxes'])):
#         box = results['boxes'][i]
#         score = results['scores'][i]
#         label = results['labels'][i]

#         if score > 0.5:  # Consider only detections with confidence > 0.5
#             x1, y1, x2, y2 = box.numpy()
#             detected_objects.append({
#                 'label': label.item(),  # Convert label tensor to scalar
#                 'confidence': score.item(),
#                 'x': int(x1),
#                 'y': int(y1),
#                 'width': int(x2 - x1),
#                 'height': int(y2 - y1)
#             })
#             total_confidence += score.item()

#     # Calculate accuracy as the average confidence of detected objects
#     accuracy = total_confidence / len(detected_objects) if detected_objects else 0
#     print(f"Detection Accuracy: {accuracy:.2f}")

#     return detected_objects, latency, accuracy

# This code for RetinaNet model
def run_model(image_path):
    # Load the RetinaNet model (you can use pre-trained weights)
    model = retinanet_resnet50_fpn(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert to tensor and add batch dimension

    # Start the inference time
    start_time = time.time()
    with torch.no_grad():
        predictions = model(image_tensor)

    # Stop the inference time
    end_time = time.time()
    latency = end_time - start_time

    # Process the predictions
    detected_objects = []
    accuracy = 0.0  # Placeholder for accuracy calculation

    # Iterate through the predictions
    for i, score in enumerate(predictions[0]['scores']):
        if score > 0.5:  # Threshold for detection confidence
            detected_objects.append({
                'label': predictions[0]['labels'][i].item(),
                'confidence': score.item(),
                'x': predictions[0]['boxes'][i][0].item(),  # x_min
                'y': predictions[0]['boxes'][i][1].item(),  # y_min
                'width': (predictions[0]['boxes'][i][2] - predictions[0]['boxes'][i][0]).item(),  # width
                'height': (predictions[0]['boxes'][i][3] - predictions[0]['boxes'][i][1]).item()  # height
            })
            accuracy += score.item()  # Sum the confidence for accuracy calculation

    accuracy /= len(detected_objects) if detected_objects else 1  # Calculate average accuracy

    return detected_objects, latency, accuracy

# Path to the input image
image_path = 'testImage.jpeg'
image = cv2.imread(image_path)

# Run object detection model
detected_objects, latency, accuracy = run_model(image_path)

# Print detected objects
print("Detected objects:", detected_objects)

# Plot the detection results on the image
output_image = plot_results(image, detected_objects)

# Display the output using matplotlib
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the output image (optional)
cv2.imwrite('output_image1.jpg', output_image)


