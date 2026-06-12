import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()

print("Spawning missing drones...")

# Spawn drone_1
try:
    client.simAddVehicle('drone_1', 'SimpleFlight', airsim.Pose(airsim.Vector3r(2, 0, -3), airsim.to_quaternion(0, 0, 0)))
    print("Spawned drone_1")
except Exception as e:
    print(f"drone_1 already exists or error: {e}")

# Spawn drone_2
try:
    client.simAddVehicle('drone_2', 'SimpleFlight', airsim.Pose(airsim.Vector3r(0, 2, -3), airsim.to_quaternion(0, 0, 0)))
    print("Spawned drone_2")
except Exception as e:
    print(f"drone_2 already exists or error: {e}")

# Spawn drone_3
try:
    client.simAddVehicle('drone_3', 'SimpleFlight', airsim.Pose(airsim.Vector3r(2, 2, -3), airsim.to_quaternion(0, 0, 0)))
    print("Spawned drone_3")
except Exception as e:
    print(f"drone_3 already exists or error: {e}")

print("All 4 drones should now be in the environment!")
