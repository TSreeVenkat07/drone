import airsim
import time
import subprocess

print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected! Waiting 2 seconds for Unreal Engine physics to settle...")
time.sleep(2)

drones = [
    ('drone_1', 2, 0),
    ('drone_2', 0, 2),
    ('drone_3', 2, 2)
]

print("Safely spawning missing drones...")
for name, x, y in drones:
    try:
        client.simAddVehicle(name, 'SimpleFlight', airsim.Pose(airsim.Vector3r(x, y, -3), airsim.to_quaternion(0, 0, 0)))
        print(f"Spawned {name}")
        time.sleep(0.5)
    except Exception as e:
        print(f"{name} already exists or error: {e}")

time.sleep(1)

print("Starting AirSim Multi-Agent Wrapper...")
subprocess.run(["python", "airsim_wrapper.py"])
