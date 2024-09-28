import torch
import time
# For RetinaNet model
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image



# This code for RetinaNet model
def run_retinanet(image_path):
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
