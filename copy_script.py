import shutil
import os

src = r"C:\Users\sreev\Documents\AirSim\settings.json"
dest = r"C:\Users\sreev\OneDrive\ドキュメント\AirSim\settings.json"

try:
    shutil.copy(src, dest)
    print("Successfully copied settings.json to OneDrive/ドキュメント")
except Exception as e:
    print(f"Error copying: {e}")
