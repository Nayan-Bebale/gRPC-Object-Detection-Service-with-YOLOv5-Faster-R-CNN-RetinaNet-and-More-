import grpc
import object_detection_pb2_grpc, object_detection_pb2
import json
import time
import cv2


def plot_results(image, detected_objects, output_image_path):
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
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Save the image with plotted results to client side
    cv2.imwrite(output_image_path, image)
    print(f"Saved the output image with detected objects at: {output_image_path}")

    return image


def run(image_path, model_type):
    output = []
    try:
        with grpc.insecure_channel('localhost:50051') as channel:
            start_time = time.time()
            stub = object_detection_pb2_grpc.ObjectDetectionServiceStub(channel)

            request = object_detection_pb2.DetectionRequest(image_path=image_path, model_type=model_type)
            print(request)
            response = stub.DetectObjects(request)
            end_time = time.time()

            latency_time = end_time - start_time
            print(f"Latency Time: {latency_time:.4f} seconds")
            print(f"Accuracy: {response.accuracy:.2f}")
            print(f"Energy Efficiency: {response.energy_efficiency}")

            
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

            print("Detected objects:")
            for obj in response.objects:
                print(f"Label: {obj.label}, Confidence: {obj.confidence}, "
                      f"Bounding Box: (x: {obj.x}, y: {obj.y}, width: {obj.width}, height: {obj.height})")

    except grpc.RpcError as e:
        print(f"gRPC call failed: {e.details()}")
        print(f"Status code: {e.code()}")

    # Load the image and plot the results
    image = cv2.imread(image_path)
    output_image_path = 'output.jpg'
    plot_results(image, output, output_image_path)


if __name__ == '__main__':
    models = {
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
        '24': 'tf-ssd',     # TensorFlow SSD
        '25': 'tf-fasterrcnn', # TensorFlow fasterrcnn
        '26': 'tf-efficientdet', # TensorFlow efficientdet
        '27': 'tf-retinanet',   # TensorFlow retinanet
        '28': 'tf-centernet',   # TensorFlow centernet
        '29': 'tf-maskrcnn',    # TensorFlow maskrcnn
    }

        
    image_path = input("Enter the image path: ")


    print("Select a model for object detection:")
    for key, model in models.items():
        print(f"{key}: {model}")
    model_type = input("Enter the number corresponding to the model you want to use: ")
    model_name = models[model_type]
    run(image_path, model_name)
