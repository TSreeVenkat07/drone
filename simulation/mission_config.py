import os
import json
import yaml

class MissionConfig:
    def __init__(self):
        # Default environment parameters
        self.grid_size = 32
        self.cell_size_meters = 5.0
        self.altitude_meters = -3.0 # Z is negative up in NED
        
        # GPS reference origin (AirSim default coordinates or near it)
        self.lat_ref = 47.641468
        self.lon_ref = -122.140165
        
        # Load grid size dynamically from uav_sar configuration if available
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env_config_path = os.path.join(self.project_root, "uav_sar", "config", "env_config.yaml")
        if os.path.exists(env_config_path):
            try:
                with open(env_config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                    self.grid_size = cfg.get("grid_size", self.grid_size)
            except Exception as e:
                print(f"[MissionConfig] Warning: Failed to load env_config.yaml: {e}")
        
        # Mission state configuration defaults
        self.map_environment = "SAR_Collapse" # Options: SAR_Collapse, SAR_Wildfire, SAR_Flood
        self.victim_positions_3d = [] # List of [x, y, z] in meters
        
        self.config_filepath = os.path.join(os.path.dirname(__file__), "mission_config.json")
        self.load()

    def load(self):
        """Loads configuration from JSON if it exists."""
        if os.path.exists(self.config_filepath):
            try:
                with open(self.config_filepath, "r") as f:
                    data = json.load(f)
                    self.map_environment = data.get("map_environment", self.map_environment)
                    self.victim_positions_3d = data.get("victims", [])
                    print(f"[MissionConfig] Loaded configuration from {self.config_filepath} successfully.")
            except Exception as e:
                print(f"[MissionConfig] Error loading mission_config.json: {e}")

    def save(self, map_env=None, victims=None):
        """Saves configuration to JSON."""
        if map_env is not None:
            self.map_environment = map_env
        if victims is not None:
            self.victim_positions_3d = victims
            
        data = {
            "map_environment": self.map_environment,
            "victims": self.victim_positions_3d
        }
        try:
            with open(self.config_filepath, "w") as f:
                json.dump(data, f, indent=4)
            print(f"[MissionConfig] Saved configuration to {self.config_filepath} successfully.")
        except Exception as e:
            print(f"[MissionConfig] Error saving mission_config.json: {e}")
            
    def get_map_scenario(self):
        """Maps operator friendly map name to RL environment scenarios."""
        mapping = {
            "SAR_Collapse": "building_collapse",
            "SAR_Wildfire": "wildfire",
            "SAR_Flood": "flood",
            "Building Collapse": "building_collapse",
            "Wildfire Zone": "wildfire",
            "Urban Flood": "flood"
        }
        return mapping.get(self.map_environment, "building_collapse")
