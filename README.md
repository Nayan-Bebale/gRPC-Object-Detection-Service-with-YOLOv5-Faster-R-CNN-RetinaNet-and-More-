# Object Detection Service using gRPC

This repository implements an object detection service powered by gRPC. The service currently supports state-of-the-art object detection models including YOLOv5, Faster R-CNN, and RetinaNet. The service is designed to process image data and return the bounding boxes, labels, and confidence scores for detected objects. More models will be added in the future.

## Output
<img src="https://raw.githubusercontent.com/Nayan-Bebale/gRPC-Object-Detection-Service-with-YOLOv5-Faster-R-CNN-RetinaNet-and-More-/refs/heads/main/output.jpg" width="600" height="600" />

with parameters like Latency Time, Accuracy, Energy Efficiency, CPU Usage, Memory Usage<br>
```
Latency Time: 9.6957 seconds
Accuracy: 0.89
Energy Efficiency: CPU Usage: 12.8%, Memory Usage: 76.4%
```
<b><i>Detected objects</i></b>
```
[{"label": "person", "confidence": 0.9996458292007446, "x": 491, "y": 594, "width": 212, "height": 536}, {"label": "person", "confidence": 0.9987173080444336, "x": 1063, "y": 557, "width": 135, "height": 460}, {"label": "bus", "confidence": 0.9984103441238403, "x": 533, "y": 111, "width": 467, "height": 568}, {"label": "bicycle", "confidence": 0.9979090690612793, "x": 1078, "y": 781, "width": 117, "height": 312}, {"label": "bicycle", "confidence": 0.9976182579994202, "x": 447, "y": 825, "width": 175, "height": 307}, {"label": "car", "confidence": 0.9952402114868164, "x": 269, "y": 530, "width": 211, "height": 358}, {"label": "bicycle", "confidence": 0.9887760877609253, "x": 2, "y": 749, "width": 228, "height": 402}, {"label": "person", "confidence": 0.987899899482727, "x": 1016, "y": 502, "width": 33, "height": 108}, {"label": "car", "confidence": 0.9850532412528992, "x": 547, "y": 499, "width": 462, "height": 514}, {"label": "car", "confidence": 0.9749124646186829, "x": 373, "y": 497, "width": 87, "height": 79}, {"label": "person", "confidence": 0.9688332676887512, "x": 20, "y": 532, "width": 194, "height": 498}, {"label": "car", "confidence": 0.9649185538291931, "x": 453, "y": 494, "width": 86, "height": 89}, {"label": "bus", "confidence": 0.9269306063652039, "x": 1103, "y": 186, "width": 93, "height": 528}, {"label": "skis", "confidence": 0.9267846941947937, "x": 535, "y": 840, "width": 90, "height": 147}, {"label": "person", "confidence": 0.9122961759567261, "x": 648, "y": 612, "width": 74, "height": 88}, {"label": "truck", "confidence": 0.8834784030914307, "x": 1, "y": 306, "width": 328, "height": 705}, {"label": "bus", "confidence": 0.8016037940979004, "x": 249, "y": 454, "width": 110, "height": 77}, {"label": "truck", "confidence": 0.6653822660446167, "x": 521, "y": 399, "width": 358, "height": 267}, {"label": "person", "confidence": 0.6590375900268555, "x": 746, "y": 230, "width": 61, "height": 75}, {"label": "car", "confidence": 0.5994460582733154, "x": 3, "y": 482, "width": 312, "height": 519}, {"label": "handbag", "confidence": 0.520714521408081, "x": 1127, "y": 649, "width": 69, "height": 83}]
```

## Features

- **gRPC Service**: Provides high-performance object detection requests via gRPC.
- **Multiple Object Detection Models Supported**:
  - YOLOv5
  - Faster R-CNN
  - RetinaNet
- **Detailed Output**: The service returns the following for each detected object:
  - Label (class name)
  - Confidence score
  - Bounding box coordinates
- **Performance Metrics**: Latency, energy efficiency, and accuracy can be tracked.
- **Extendable**: New object detection models can be added easily.

## Models Supported

- **YOLOv5**: Known for fast and accurate object detection, ideal for real-time scenarios.
- **Faster R-CNN**: Focused on high-quality, accurate object detection.
- **RetinaNet**: Balances speed and accuracy, popular for detecting objects in dense scenes.

### Sample Outputs

Here are sample images generated using the service, showcasing object detection results.

**YOLOv5 Output:**

![YOLOv5 Detection](https://github.com/Nayan-Bebale/gRPC-Object-Detection-Service-with-YOLOv5-Faster-R-CNN-RetinaNet-and-More-/blob/main/YoloV5.jpg)

In this image, YOLOv5 detects objects such as bicycles, cars, and people, along with confidence scores.

**RetinaNet Output:**

![RetinaNet Detection](https://github.com/Nayan-Bebale/gRPC-Object-Detection-Service-with-YOLOv5-Faster-R-CNN-RetinaNet-and-More-/blob/main/Retina.png)

RetinaNet detects multiple objects including trucks, cars, and people, with bounding boxes drawn around them and associated labels.

### Sample Detection Data

The following is a sample detection output generated by the service. The results include object labels, confidence levels, and bounding box coordinates.

```
(file-cWgiBkOhKZMVa4WOrQmJdtIR - Results.txt)

Detected objects:
- Object: Person, Confidence: 0.88, Box: [x1, y1, x2, y2]
- Object: Bicycle, Confidence: 0.46, Box: [x1, y1, x2, y2]
- Object: Truck, Confidence: 0.72, Box: [x1, y1, x2, y2]
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/object-detection-service
   cd object-detection-service
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the gRPC server:
   ```bash
   python server.py
   ```

## Usage

1. Send an image to the gRPC server for detection. The server will return the objects detected along with bounding boxes and confidence scores.
2. You can use any gRPC client to interact with the service. A sample Python client is included in this repository.

## Future Work

- Add support for additional models like EfficientDet, SSD and more.
- Implement further model optimizations for faster inference and reduced energy consumption.

---
