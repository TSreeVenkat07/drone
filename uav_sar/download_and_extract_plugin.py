import os
import urllib.request
import zipfile

# Define source and destination
url = "https://github.com/Cosys-Lab/Cosys-AirSim/releases/download/5.5-v3.3/AirSim_plugin_Windows_55_33.zip"
zip_path = "AirSim_plugin_Windows_55_33.zip"
dest_dir = u"C:\\Users\\sreev\\OneDrive\\\u30c9\u30ad\u30e5\u30e1\u30f3\u30c8\\Unreal Projects\\finalyearvenkat\\Plugins"

print("Downloading AirSim plugin (Windows)...")
try:
    urllib.request.urlretrieve(url, zip_path)
    print("Download completed successfully!")
    
    # Create Plugins folder if not exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print("Created Plugins folder in project.")
        
    print(f"Extracting to {dest_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print("Extraction completed!")
    
    # Clean up zip
    os.remove(zip_path)
    print("Cleaned up download archive.")
    print("\n[SUCCESS] AirSim plugin successfully installed in your project!")
except Exception as e:
    print("Error during setup:", e)
