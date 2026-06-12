class VictimTracker:
    def __init__(self):
        self.found_victims = {}
        self.origin_lat = 17.3850
        self.origin_lon = 78.4867

    def physical_to_gps(self, grid_x: int, grid_y: int) -> tuple:
        """Converts grid coordinates to GPS using fixed origin (Hyderabad)."""
        lat = self.origin_lat + (grid_y * 0.0001)
        lon = self.origin_lon + (grid_x * 0.0001)
        return lat, lon

    def record_victim_found(self, victim_idx: int, grid_x: int, grid_y: int, step: int) -> dict:
        if victim_idx in self.found_victims:
            return self.found_victims[victim_idx]
            
        lat, lon = self.physical_to_gps(grid_x, grid_y)
        
        victim_details = {
            "index": victim_idx,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "latitude": lat,
            "longitude": lon,
            "step_found": step
        }
        self.found_victims[victim_idx] = victim_details
        return victim_details

    def get_found_victims_list(self) -> list:
        return [self.found_victims[idx] for idx in sorted(self.found_victims.keys())]
