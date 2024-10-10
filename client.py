from flask import Flask, render_template, request, redirect, url_for
import grpc
import object_detection_pb2_grpc, object_detection_pb2
import json
import time
import cv2
import requests
import numpy as np
import os

app = Flask(__name__)

MODELS = {
        '1': 'fasterrcnn',
        '2': 'fasterrcnn_mobilenet',
        '3': 'fasterrcnn_v2',
        '4': 'maskrcnn',
        '5': 'maskrcnn_v2',
        '6': 'keypointrcnn',
        '7': 'retinanet',
        '8': 'ssd',
        '9': 'ssdlite',
        '10': 'yolov3n',
        '11': 'yolov5s',  # YOLOv5 small model
        '12': 'yolov5l', # YOLOv5 large model
        '13': 'yolov5x', # YOLOv5 extra large model
        '14': 'yolov6-n', # YOLOv6 neno model
        '15': 'yolov6-s', # YOLOv6 small model
        '16': 'yolov6-m', # YOLOv6 medium model
        '17': 'yolov6-l', # YOLOv6 large model
        '18': 'yolov6-l6', # YOLOv6 small model
        '19': 'yolov8n', # YOLOv8 nano model
        '20': 'yolov8s', # YOLOv8 small model
        '21': 'yolov8m', # YOLOv8 medium model
        '22': 'yolov8l', # YOLOv8 large model
        '23': 'yolov8x',  # YOLOv8 extra large model
        '24': 'tf-ssd_mobilenet_v2',     
        '25': 'tf-ssd_mobilenet_v1', 
        '26': 'tf-ssd_resnet50', 
        '27': 'tf-faster_rcnn_resnet50',  
        '28': 'tf-faster_rcnn_inception',   
        '29': 'tf-efficientdet_d0',    
        '30': 'tf-efficientdet_d1',
        '31': 'tf-efficientdet_d2',
        '32': 'tf-efficientdet_d3',
        '33': 'tf-retinanet',
        '34': 'tf-centernet_hourglass',
        '35': 'tf-centernet_resnet50',
        '36': 'tf-mask_rcnn_resnet50',
        '37': 'tf-mask_rcnn_inception',
        '38': 'tf-yolo_v4',
    }

# Ensure the static directory for output images exists
if not os.path.exists('static/output_images'):
    os.makedirs('static/output_images')

# Function to download the image from URL
def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        print(f"Failed to download image from URL. Status code: {response.status_code}")
        return None

# Function to plot results and save the image
def plot_results(image, detected_objects, output_image_name):
    for obj in detected_objects:
        x = int(obj['x'])
        y = int(obj['y'])
        w = int(obj['width'])
        h = int(obj['height'])

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"L: {obj['label']}, C: {obj['confidence']:.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    output_image_path = os.path.join('static', 'output_images', output_image_name)
    cv2.imwrite(output_image_path, image)
    print(f"Saved the output image with detected objects at: {output_image_path}")
    return output_image_path

# gRPC request function
def run(image_path, model_type):
    output = []
    try:
        with grpc.insecure_channel('localhost:50051') as channel:
            start_time = time.time()
            stub = object_detection_pb2_grpc.ObjectDetectionServiceStub(channel)

            request = object_detection_pb2.DetectionRequest(image_path=image_path, model_type=model_type)
            response = stub.DetectObjects(request)
            end_time = time.time()

            latency_time = end_time - start_time
            latency_time = round(latency_time, 4)
            print(f"Latency Time: {latency_time:.4f} seconds")
            print(f"Accuracy: {response.accuracy:.2f}")
            print(f"Energy Efficiency: {response.energy_efficiency}")

            accuracy = response.accuracy
            energy_efficiency = response.energy_efficiency

            for obj in response.objects:
                output.append({
                    'label': obj.label,
                    'confidence': obj.confidence,
                    'x': obj.x,
                    'y': obj.y,
                    'width': obj.width,
                    'height': obj.height
                })

            with open('output.json', 'w') as f:
                json.dump(output, f)

    except grpc.RpcError as e:
        print(f"gRPC call failed: {e.details()}")
        print(f"Status code: {e.code()}")

    if image_path.startswith('http://') or image_path.startswith('https://'):
        image = download_image(image_path)
    else:
        image = cv2.imread(image_path)

    if image is None:
        print("Image loading failed. Exiting.")
        return None, None

    output_image_path = 'output.jpg'
    result_image_path = plot_results(image, output, output_image_path)
    return output, result_image_path, accuracy, energy_efficiency, latency_time

# Route to serve the HTML form
@app.route('/')
def index():
    
    return render_template('index.html', models=MODELS)

# Route to handle form submission
@app.route('/detect', methods=['POST'])
def detect():
    image_url = request.form['image_url']
    model_num = request.form['model']
    model_type = MODELS[model_num]


    objects_detected, result_image, accuracy, energy_efficiency, latency_time = run(image_url, model_type)


    
    if objects_detected:
        result_image_url = url_for('static', filename=f'output_images/{os.path.basename(result_image)}')
        return render_template('results.html', objects=objects_detected, result_image=result_image_url, accuracy=accuracy, energy_efficiency=energy_efficiency, latency_time=latency_time)
    else:
        return "Object detection failed."

if __name__ == '__main__':
    app.run(debug=True)
