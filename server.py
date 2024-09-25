import grpc
from concurrent import futures
import object_detection_pb2_grpc, object_detection_pb2
from detect import run_model  # Assuming you have an existing function to run the model
import warnings
import os
import psutil
import time

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')


def get_energy_efficiency():
    # Get CPU usage and memory consumption
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    # Return CPU and memory usage as an indicator of energy efficiency
    return f"CPU Usage: {cpu_percent}%, Memory Usage: {memory_info.percent}%"

class ObjectDetectionServiceServicer(object_detection_pb2_grpc.ObjectDetectionServiceServicer):
    def DetectObjects(self, request, context):
        print(f"Received request for image path: {request.image_path}")  # Debug print
        detected_objects, latency, accuracy = run_model(request.image_path)

        # Debug print the detected objects
        print(f"Detected objects: {detected_objects}")
        # Measure and print energy efficiency
        energy_efficiency = get_energy_efficiency()
        
        print(f"Latency: {latency:.4f} seconds, Accuracy: {accuracy:.2f}, Energy Efficiency: {energy_efficiency} watts")

        # Prepare the response with accuracy and energy efficiency
        response = object_detection_pb2.DetectionResponse()
        response.accuracy = accuracy
        response.energy_efficiency = energy_efficiency

        # Prepare the response objects
        for obj in detected_objects:
            detected_object = object_detection_pb2.DetectedObject(
                label=obj['label'],
                confidence=obj['confidence'],
                x=obj['x'],
                y=obj['y'],
                width=obj['width'],
                height=obj['height']
            )
            response.objects.append(detected_object)
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    object_detection_pb2_grpc.add_ObjectDetectionServiceServicer_to_server(
        ObjectDetectionServiceServicer(), server)
    server.add_insecure_port('[::]:50051')  # Change to 50051
    server.start()
    print("Server running on port 50051...")  # Confirming server startup
    server.wait_for_termination()
    
    try:
        while True:
            time.sleep(86400)  # Keep the server alive
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
