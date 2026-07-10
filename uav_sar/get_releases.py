import urllib.request
import json

url = "https://api.github.com/repos/Cosys-Lab/Cosys-AirSim/releases"
req = urllib.request.Request(
    url, 
    headers={'User-Agent': 'Mozilla/5.0'}
)

try:
    with urllib.request.urlopen(req) as response:
        releases = json.loads(response.read().decode('utf-8'))
        print("Available Cosys-AirSim Releases:")
        for r in releases:
            print(f"\nTag: {r['tag_name']} - {r['name']}")
            for asset in r.get('assets', []):
                print(f"  Asset: {asset['name']}")
                print(f"  URL: {asset['browser_download_url']}")
except Exception as e:
    print("Error fetching releases:", e)
