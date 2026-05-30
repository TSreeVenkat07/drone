import os
import zipfile

zip_path = "AirSim_plugin_Windows_55_33.zip"
dest_dir = u"C:\\Users\\sreev\\OneDrive\\\u30c9\u30ad\u30e5\u30e1\u30f3\u30c8\\Unreal Projects\\finalyearvenkat\\Plugins"

try:
    if os.path.exists(zip_path):
        print("Zip archive found locally. Extracting...")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Extraction completed successfully!")
        os.remove(zip_path)
        print("Cleaned up archive.")
    else:
        print("Zip archive not found. Checking if extraction was already done.")
except Exception as e:
    print("Error during extraction:", str(e).encode('utf-8'))
