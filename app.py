import ultralytics
from ultralytics import YOLO
import os
import time
import json
import requests
import cv2
from requests.auth import HTTPBasicAuth
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, ContentSettings
from geopy.geocoders import Nominatim
from io import BytesIO
import geocoder  # Make sure to install the geocoder library using: pip install geocoder

weights_path = 'best.pt'  # Adjust this to the path of your YOLOv5 weights
model = YOLO(weights_path)

# Replace with your Azure Blob Storage credentials
blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=alexandsonsstorage;AccountKey=bvUyX6uT35VX1WWSIHmrUxEzJtNSEVv5sukMyPorzSrEswkfopO1cW/qob230VdzRAHLCrAHVMUB+AStHrBtAw==;EndpointSuffix=core.windows.net")
container_client = blob_service_client.get_container_client("towtruckpics")

# Replace with your API endpoint and headers for the user and telemetry API
telemetry_api_url = "https://reposessionsapp.azurewebsites.net/ingest_json"
headers = {"Content-Type": "application/json"}

# Function to get location data based on latitude and longitude
def get_location_data(latitude, longitude):
    geolocator = Nominatim(user_agent="location_service")
    location = geolocator.reverse((latitude, longitude), language="en")
    return location.address if location else None

# Function to get the current location data of the system
def get_system_location():
    location = geocoder.ip('me')
    return location.latlng if location else None

# Function to capture and upload images
def capture_and_upload_images():
    user_id = "user1"  # Replace with the unique user ID

    cap = cv2.VideoCapture(1)  # 0 corresponds to the default camera, you can change it based on your camera setup

    while True:
        ret, frame = cap.read()

        # Get the system's current location
        system_location = get_system_location()
        if system_location:
            latitude, longitude = system_location
        else:
            latitude, longitude = 0.0, 0.0  # Default values if location retrieval fails

        # Perform YOLOv5 inference on the captured frame
        results = model(frame)
        box_found = False
        for result in results:
            data = result.boxes.data
            target_boxes = data[(data[:, -1] == 0) & (data[:, -2] >= 0.5)]
            if len(target_boxes) > 0:
                box_found = True

        # Display the frame
        cv2.imshow('OpenCV Camera', frame)
        if box_found:
            # Upload the image to Azure Blob Storage
            image_data = cv2.imencode('.png', frame)[1].tobytes()
            image_stream = BytesIO(image_data)

            # Generate a unique blob name (you can customize this based on your needs)
            blob_name = f"image_{time.time()}.png"

            # Upload the image to Azure Blob Storage
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(image_stream.read(), content_settings=ContentSettings(content_type="image/png"))

            location_data = get_location_data(latitude, longitude)

            # Prepare telemetry data
            telemetry_data = {
                "user_id": user_id,
                "blob_id": blob_client.get_blob_properties()['name'],
                "location": location_data,
                "timestamp": time.time(),
                "fleet_car": "fleet1"
                # Add other telemetry data as needed
            }
            print(telemetry_data)

            response = requests.post(telemetry_api_url, json=telemetry_data, headers=headers,
                                     auth=HTTPBasicAuth('user', 'pass'))
            print(response)
            if response.status_code == 200:
                print("POST request successful!")
                print("Response:", response.text)
            else:
                print(f"POST request failed with status code {response.status_code}")
                print("Response:", response.text)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(3)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_upload_images()
