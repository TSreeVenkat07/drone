import cv2
import numpy as np
import time
import os
from PIL import ImageGrab

print("Screen Recording video...")
video_path = os.path.expanduser("~/Desktop/drone_flight_proof.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Get screen size from first frame
img = ImageGrab.grab()
width, height = img.size
out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

start_time = time.time()

# Record for 90 seconds while the simulation runs in the background
while time.time() - start_time < 90.0:
    img = ImageGrab.grab()
    img_np = np.array(img)
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    out.write(img_bgr)
    time.sleep(0.1)

out.release()
print(f"Video successfully saved to {video_path}")
