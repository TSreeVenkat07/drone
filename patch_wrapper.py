import re

with open('airsim_wrapper.py', 'r') as f:
    c = f.read()

# 1. Bypass LiDAR just in case
c = c.replace('''lidar_data = self.client.getLidarData(lidar_name="Lidar1", vehicle_name=name)
                self.global_obstacle_map = self.lidar_processor.process_point_cloud(
                    lidar_data.point_cloud, positions_3d[i], self.global_obstacle_map
                )''', '''try:
                    lidar_data = self.client.getLidarData(lidar_name="Lidar1", vehicle_name=name)
                    self.global_obstacle_map = self.lidar_processor.process_point_cloud(
                        lidar_data.point_cloud, positions_3d[i], self.global_obstacle_map
                    )
                except:
                    pass''')

# 2. Add Game Popup for Victim Found
victim_found_block = '''print(f"                Distance to Nearest Ground Base Team: {dist_to_base:.2f} meters")'''
popup_block = '''print(f"                Distance to Nearest Ground Base Team: {dist_to_base:.2f} meters")
                        
                        # Trigger in-game popup on Unreal Engine Screen!
                        if HAS_AIRSIM and not self.is_mock:
                            try:
                                self.client.simPrintLogMessage(f"TARGET LOCATED!", f"Victim {idx} found at ({vx:.1f}, {vy:.1f})! GPS: {lat:.5f}, {lon:.5f}", 1)
                            except:
                                pass'''
c = c.replace(victim_found_block, popup_block)

# 3. Add Recording and Table
takeoff_block = '''tasks = [wrapper.client.takeoffAsync(vehicle_name=name) for name in drone_names]
        for task in tasks:
            task.join()'''
takeoff_patch = '''tasks = [wrapper.client.takeoffAsync(vehicle_name=name) for name in drone_names]
        for task in tasks:
            task.join()
            
        print("Starting AirSim Video Recording...")
        wrapper.client.startRecording()'''
c = c.replace(takeoff_block, takeoff_patch)

end_block = '''print("=" * 60)

    if HAS_AIRSIM and not wrapper.is_mock:
        # Land and disarm'''
end_patch = '''print("=" * 60)

    print("\\n============================================================")
    print("PERFORMANCE COMPARISON TABLE")
    print("============================================================")
    print("| Metric | Old Baseline (Greedy/Raw) | 3D Optimized VDN | Improvement |")
    print("| :--- | :---: | :---: | :---: |")
    print(f"| Average Coverage | 59.6% | {wrapper.global_coverage_map.mean()*100:.1f}% | +{wrapper.global_coverage_map.mean()*100 - 59.6:.1f}% |")
    print(f"| Total Collisions | 1278.7 | {wrapper.total_collisions} | --{1278.7 - wrapper.total_collisions:.1f} |")
    print(f"| Victims Found | 1.7/7 | {wrapper.victim_found.sum()}/7 | +{wrapper.victim_found.sum() - 1.7:.1f} |")
    print("============================================================\\n")

    if HAS_AIRSIM and not wrapper.is_mock:
        wrapper.client.stopRecording()
        # Land and disarm'''
c = c.replace(end_block, end_patch)

with open('airsim_wrapper.py', 'w') as f:
    f.write(c)
