import grpc
import object_detection_pb2_grpc, object_detection_pb2
import json
import time

def run(image_path):
    # Connect to the gRPC server
    with grpc.insecure_channel('localhost:50051') as channel:
        # Latency Time start
        start_time = time.time()
        stub = object_detection_pb2_grpc.ObjectDetectionServiceStub(channel)
        # Create a request
        request = object_detection_pb2.DetectionRequest(image_path=image_path)
        # Call the RPC method
        response = stub.DetectObjects(request)
        # Latency Time end
        end_time = time.time()
        # Latency and other metrics
        latency_time = end_time - start_time
        accuracy = response.accuracy
        energy_efficiency = response.energy_efficiency

        # Print the metrics
        print(f"Latency Time: {latency_time:.4f} seconds")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Energy Efficiency: {energy_efficiency} watts")

        output = []
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
        if response.objects:
            for obj in response.objects:
                print(f"Label: {obj.label}, Confidence: {obj.confidence}")
                print(f"Bounding Box: (x: {obj.x}, y: {obj.y}, width: {obj.width}, height: {obj.height})")
        else:
            print("No objects detected.")



if __name__ == '__main__':
    image_path = 'testImage.jpeg'
    print(f"Running object detection on image: {image_path}")

    
    
    run(image_path)
