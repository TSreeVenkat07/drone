"""
xBD Dataset Download & Selection Script
Downloads the xView2 challenge dataset from Kaggle, then selects
1-2 post-disaster images per disaster type (flood, wildfire, building collapse).
"""

import os
import sys
import json
import glob
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

EVENT_CLASSIFICATION = {
    "midwest-flooding": "flood", "nepal-flooding": "flood", "florence": "flood",
    "harvey": "flood", "matthew": "flood", "flooding": "flood",
    "socal-fire": "wildfire", "santa-rosa-wildfire": "wildfire", "woolsey-fire": "wildfire",
    "carr-fire": "wildfire", "camp-fire": "wildfire", "paradise": "wildfire",
    "portugal-wildfire": "wildfire", "wildfire": "wildfire", "pinery-bushfire": "wildfire",
    "mexico-earthquake": "building_collapse", "guatemala-volcano": "building_collapse",
    "palu-tsunami": "building_collapse", "moore-tornado": "building_collapse",
    "joplin-tornado": "building_collapse", "tuscaloosa-tornado": "building_collapse",
    "tornado": "building_collapse", "earthquake": "building_collapse",
    "tsunami": "building_collapse", "volcano": "building_collapse",
    "hurricane-michael": "building_collapse", "hurricane-matthew": "building_collapse",
    "hurricane": "building_collapse", "lower-puna-volcano": "building_collapse",
    "sunda-tsunami": "building_collapse",
}


def classify_event(event_name: str) -> str:
    name_lower = event_name.lower().replace("_", "-")
    if name_lower in EVENT_CLASSIFICATION:
        return EVENT_CLASSIFICATION[name_lower]
    for key, category in EVENT_CLASSIFICATION.items():
        if key in name_lower or name_lower in key:
            return category
    if any(w in name_lower for w in ["flood", "harvey", "florence", "matthew", "cyclone"]):
        return "flood"
    if any(w in name_lower for w in ["fire", "wildfire", "burn", "blaze"]):
        return "wildfire"
    if any(w in name_lower for w in ["earthquake", "tornado", "tsunami", "volcano", "collapse", "hurricane"]):
        return "building_collapse"
    return "unknown"


def count_damage_levels(json_path: str) -> Dict[str, int]:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}
    counts = {"no-damage": 0, "minor-damage": 0, "major-damage": 0, "destroyed": 0, "un-classified": 0}
    features = data.get("features", {})
    xy_features = features.get("xy", [])
    if not xy_features and isinstance(features, list):
        xy_features = features
    for feat in xy_features:
        props = feat.get("properties", {})
        damage = props.get("subtype", props.get("damage", "un-classified"))
        if damage in counts:
            counts[damage] += 1
        else:
            counts["un-classified"] += 1
    return counts


def find_post_disaster_images(data_dir: str) -> List[Dict]:
    results = []
    search_paths = [
        os.path.join(data_dir, "**", "*_post_disaster.json"),
        os.path.join(data_dir, "**", "*post*.json"),
    ]
    json_files = set()
    for pattern in search_paths:
        json_files.update(glob.glob(pattern, recursive=True))

    for json_path in sorted(json_files):
        basename = os.path.basename(json_path)
        parts = basename.replace("_post_disaster.json", "").rsplit("_", 1)
        if len(parts) >= 2:
            event_name, image_id = parts[0], parts[1]
        else:
            event_name, image_id = basename.replace(".json", ""), "0"

        disaster_type = classify_event(event_name)
        image_path = json_path.replace(".json", ".png")
        if not os.path.exists(image_path):
            image_path = json_path.replace(".json", ".tif")
        if not os.path.exists(image_path):
            image_path = None

        damage_counts = count_damage_levels(json_path)
        total_damaged = damage_counts.get("major-damage", 0) + damage_counts.get("destroyed", 0)

        results.append({
            "event_name": event_name, "image_id": image_id,
            "disaster_type": disaster_type, "json_path": json_path,
            "image_path": image_path, "damage_counts": damage_counts,
            "total_damaged": total_damaged,
            "total_buildings": sum(damage_counts.values()),
        })
    return results


def select_best_images(all_images: List[Dict], per_type: int = 2) -> Dict[str, List[Dict]]:
    by_type = {"flood": [], "wildfire": [], "building_collapse": []}
    for img in all_images:
        dtype = img["disaster_type"]
        if dtype in by_type:
            by_type[dtype].append(img)
    selected = {}
    for dtype, images in by_type.items():
        images.sort(key=lambda x: x["total_damaged"], reverse=True)
        selected[dtype] = images[:per_type]
    return selected


def download_dataset(output_dir: str):
    try:
        import kaggle
        print(f"Downloading xView2 dataset to {output_dir}...")
        print("This may take a while (dataset is ~23 GB)...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "tunguz/xview2-challenge-dataset-train-and-test",
            path=output_dir, unzip=True,
        )
        print("Download complete!")
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        sys.exit(1)


def create_synthetic_xbd_data(selected_dir: str):
    """Create synthetic xBD-format test data with real-world GPS coordinates."""
    os.makedirs(selected_dir, exist_ok=True)

    scenarios = {
        "flood": [
            {"event": "midwest-flooding", "id": "00001",
             "bounds": [-90.1237, 29.9511, -90.1137, 29.9611],
             "damage_dist": {"no-damage": 10, "minor-damage": 8, "major-damage": 12, "destroyed": 15}},
            {"event": "nepal-flooding", "id": "00002",
             "bounds": [85.3000, 27.7000, 85.3100, 27.7100],
             "damage_dist": {"no-damage": 5, "minor-damage": 10, "major-damage": 10, "destroyed": 13}},
        ],
        "wildfire": [
            {"event": "socal-fire", "id": "00001",
             "bounds": [-118.7500, 34.2700, -118.7400, 34.2800],
             "damage_dist": {"no-damage": 8, "minor-damage": 5, "major-damage": 10, "destroyed": 12}},
            {"event": "santa-rosa-wildfire", "id": "00002",
             "bounds": [-122.7200, 38.4400, -122.7100, 38.4500],
             "damage_dist": {"no-damage": 6, "minor-damage": 7, "major-damage": 12, "destroyed": 15}},
        ],
        "building_collapse": [
            {"event": "mexico-earthquake", "id": "00001",
             "bounds": [-99.1600, 19.4200, -99.1500, 19.4300],
             "damage_dist": {"no-damage": 12, "minor-damage": 10, "major-damage": 13, "destroyed": 15}},
            {"event": "palu-tsunami", "id": "00002",
             "bounds": [119.8400, -0.9000, 119.8500, -0.8900],
             "damage_dist": {"no-damage": 8, "minor-damage": 6, "major-damage": 14, "destroyed": 14}},
        ],
    }

    rng = np.random.default_rng(42)

    for dtype, events in scenarios.items():
        for event_info in events:
            filename_base = f"{event_info['event']}_{event_info['id']}_post_disaster"
            json_path = os.path.join(selected_dir, f"{filename_base}.json")

            bounds = event_info["bounds"]
            features = []
            damage_labels = []
            for label, count in event_info["damage_dist"].items():
                damage_labels.extend([label] * count)
            rng.shuffle(damage_labels)

            for i, damage_label in enumerate(damage_labels):
                cx, cy = rng.uniform(0, 1024), rng.uniform(0, 1024)
                w, h = rng.uniform(15, 45), rng.uniform(15, 45)
                coords = [
                    (cx - w/2, cy - h/2), (cx + w/2, cy - h/2),
                    (cx + w/2, cy + h/2), (cx - w/2, cy + h/2),
                    (cx - w/2, cy - h/2),
                ]
                wkt_str = "POLYGON ((" + ", ".join(f"{x:.1f} {y:.1f}" for x, y in coords) + "))"
                features.append({
                    "properties": {
                        "feature_type": "building", "subtype": damage_label,
                        "uid": f"bldg_{i:04d}", "wkt": wkt_str,
                    }
                })

            annotation = {
                "metadata": {
                    "img": {"width": 1024, "height": 1024},
                    "geo": {
                        "bounds": bounds,
                        "lng": (bounds[0] + bounds[2]) / 2,
                        "lat": (bounds[1] + bounds[3]) / 2,
                    },
                    "disaster": event_info["event"],
                    "disaster_type": dtype,
                },
                "features": {"xy": features},
            }

            with open(json_path, "w") as f:
                json.dump(annotation, f, indent=2)

            n_buildings = sum(event_info["damage_dist"].values())
            n_destroyed = event_info["damage_dist"]["destroyed"]
            print(f"  Created: {filename_base}.json ({n_buildings} buildings, {n_destroyed} destroyed)")


def main():
    parser = argparse.ArgumentParser(description="Download and select xBD dataset images")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--per-type", type=int, default=2)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data", "xbd_raw")
    selected_dir = os.path.join(base_dir, "data", "selected")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(selected_dir, exist_ok=True)

    if args.synthetic:
        print("=" * 70)
        print("GENERATING SYNTHETIC xBD DATA (for development/testing)")
        print("=" * 70)
        create_synthetic_xbd_data(selected_dir)
        print(f"\nSynthetic data created in: {selected_dir}")
        return

    data_dir = args.data_dir or raw_dir
    if not args.skip_download:
        download_dataset(raw_dir)
        data_dir = raw_dir

    print(f"\nScanning for post-disaster images in: {data_dir}")
    all_images = find_post_disaster_images(data_dir)
    print(f"Found {len(all_images)} post-disaster annotations")

    if not all_images:
        print("\nNo xBD images found. Generating synthetic data instead...")
        create_synthetic_xbd_data(selected_dir)
        return

    selected = select_best_images(all_images, per_type=args.per_type)
    print(f"\nSelected images for evaluation:")
    for dtype, images in selected.items():
        print(f"\n  {dtype.upper()}:")
        for img in images:
            print(f"    {img['event_name']}_{img['image_id']}: "
                  f"{img['total_buildings']} buildings, "
                  f"{img['damage_counts'].get('destroyed', 0)} destroyed")

    for dtype, images in selected.items():
        for img in images:
            src_json = img["json_path"]
            dst_json = os.path.join(selected_dir, os.path.basename(src_json))
            shutil.copy2(src_json, dst_json)
            if img["image_path"] and os.path.exists(img["image_path"]):
                dst_img = os.path.join(selected_dir, os.path.basename(img["image_path"]))
                shutil.copy2(img["image_path"], dst_img)

    print(f"\nSelected files copied to: {selected_dir}")


if __name__ == "__main__":
    main()
