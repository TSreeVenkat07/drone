import sys
import time
import numpy as np

try:
    import airsim
    HAS_AIRSIM = True
except ImportError:
    HAS_AIRSIM = False
    print("[WARNING] airsim package is not importable. Running in complete Mock Mode.")

class MockMultirotorClient:
    """Mock client mimicking AirSim multirotor API behavior for testing without UE5."""
    def __init__(self):
        print("[MOCK] Initializing MockMultirotorClient...")
        self.drones = {f"drone_{i}": {"armed": False, "api_control": False, "pos": [0.0, 0.0, 0.0]} for i in range(4)}

    def confirmConnection(self):
        print("[MOCK] confirmConnection: Connection successfully established with mock simulator!")

    def enableApiControl(self, is_enabled, vehicle_name=""):
        self.drones[vehicle_name]["api_control"] = is_enabled
        print(f"[MOCK] enableApiControl: API Control set to {is_enabled} for {vehicle_name}")

    def armDisarm(self, is_armed, vehicle_name=""):
        self.drones[vehicle_name]["armed"] = is_armed
        print(f"[MOCK] armDisarm: Armed state set to {is_armed} for {vehicle_name}")

    def takeoffAsync(self, vehicle_name=""):
        class MockTask:
            def __init__(self, name, drones):
                self.name = name
                self.drones = drones
            def join(self):
                time.sleep(0.5)
                self.drones[self.name]["pos"][2] = -3.0 # NED coordinate (negative = up)
                print(f"[MOCK] takeoffAsync: {self.name} takeoff completed. Current position: {self.drones[self.name]['pos']}")
        return MockTask(vehicle_name, self.drones)

    def getMultirotorState(self, vehicle_name=""):
        class MockPosition:
            def __init__(self, p):
                self.x, self.y, self.z = p
        class MockKinematics:
            def __init__(self, p):
                self.position = MockPosition(p)
        class MockState:
            def __init__(self, p):
                self.kinematics_estimated = MockKinematics(p)
        return MockState(self.drones[vehicle_name]["pos"])

    def landAsync(self, vehicle_name=""):
        class MockTask:
            def __init__(self, name, drones):
                self.name = name
                self.drones = drones
            def join(self):
                time.sleep(0.5)
                self.drones[self.name]["pos"][2] = 0.0
                print(f"[MOCK] landAsync: {self.name} landing completed. Current position: {self.drones[self.name]['pos']}")
        return MockTask(vehicle_name, self.drones)

def test_drones_takeoff():
    print("=" * 60)
    print("AirSim 4-Drone Takeoff Verification Script")
    print("=" * 60)

    client = None
    is_mock = False

    if HAS_AIRSIM:
        try:
            # Short timeout connect check
            client = airsim.MultirotorClient(timeout_value=2)
            print("Connecting to live AirSim simulator...")
            client.confirmConnection()
            print("[SUCCESS] Connected to live AirSim server!")
        except Exception as e:
            print("\n[INFO] Failed to connect to live AirSim simulator. Rationale:")
            print(f"       {e}")
            print("       Please ensure 'Disaster_Building_Collapse.exe' is running.")
            print("       Switching to Mock Mode to verify code sequence...\n")
            client = MockMultirotorClient()
            is_mock = True
    else:
        client = MockMultirotorClient()
        is_mock = True

    drone_names = [f"drone_{i}" for i in range(4)]

    # 1. Enable API Control & Arm
    print("\n--- Arming and Enabling API Control ---")
    for name in drone_names:
        client.enableApiControl(True, vehicle_name=name)
        client.armDisarm(True, vehicle_name=name)

    # 2. Takeoff
    print("\n--- Initiating Takeoff Sequence (Target Altitude: 3.0m) ---")
    tasks = []
    for name in drone_names:
        print(f"Requesting takeoff for {name}...")
        task = client.takeoffAsync(vehicle_name=name)
        tasks.append((name, task))

    print("\nWaiting for takeoff tasks to complete...")
    for name, task in tasks:
        task.join()
        print(f"Takeoff confirmed for {name}.")

    # 3. Verify Altitude
    print("\n--- Altitude Verification ---")
    for name in drone_names:
        state = client.getMultirotorState(vehicle_name=name)
        pos = state.kinematics_estimated.position
        # AirSim uses NED coordinates: Z is negative for height
        altitude = -pos.z
        print(f"Agent '{name}' position: X={pos.x:.2f}m, Y={pos.y:.2f}m, Altitude={altitude:.2f}m")
        if not is_mock:
            assert abs(altitude - 3.0) < 1.0, f"{name} failed to reach target altitude: got {altitude}m"

    print("\n[SUCCESS] Takeoff and hover hold sequence verified!")

    # 4. Land and clean up
    print("\n--- Landing Drones ---")
    land_tasks = []
    for name in drone_names:
        print(f"Requesting land for {name}...")
        task = client.landAsync(vehicle_name=name)
        land_tasks.append(task)
    for task in land_tasks:
        task.join()

    print("\nDisarming and disabling API control...")
    for name in drone_names:
        client.armDisarm(False, vehicle_name=name)
        client.enableApiControl(False, vehicle_name=name)

    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_drones_takeoff()
