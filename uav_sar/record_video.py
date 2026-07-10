import airsim
import cv2
import numpy as np
import time
import os

print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, "SimpleFlight")
client.armDisarm(True, "SimpleFlight")

print("Taking off...")
client.takeoffAsync(vehicle_name="SimpleFlight").join()
client.moveByVelocityZAsync(0, 0, -3, 1, vehicle_name="SimpleFlight").join()

print("Recording video...")
video_path = os.path.expanduser("~/Desktop/drone_flight_proof.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

# Fly forward slowly and record for 10 seconds
client.moveByVelocityZAsync(2.0, 0, -3, 10, vehicle_name="SimpleFlight")
start_time = time.time()

while time.time() - start_time < 10.0:
    try:
        response = client.simGetImage("0", airsim.ImageType.Scene, vehicle_name="SimpleFlight")
        if response:
            # Decode the png binary
            img1d = np.frombuffer(response, dtype=np.uint8)
            img_rgb = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
            
            if out is None:
                height, width, _ = img_rgb.shape
                out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
                
            out.write(img_rgb)
    except Exception as e:
        print(f"Error capturing image: {e}")
    time.sleep(0.1)

if out is not None:
    out.release()

print("Landing...")
client.landAsync(vehicle_name="SimpleFlight").join()
client.armDisarm(False, "SimpleFlight")
client.enableApiControl(False, "SimpleFlight")
print(f"Video saved to {video_path}")
