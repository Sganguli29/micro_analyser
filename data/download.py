"""
data/download.py - Parallel download of dataset files from GitHub raw storage.
"""

import os
import requests
import concurrent.futures
from pathlib import Path

REPO_API = "https://api.github.com/repos/sombsuk/dataset_microplastic/contents"
RAW_BASE = "https://raw.githubusercontent.com/sombsuk/dataset_microplastic/main"

def download_file(item, dest_dir: Path):
    name = item["name"]
    dest_path = dest_dir / name
    if dest_path.exists() and dest_path.stat().st_size == item["size"]:
        print(f"Skipping {name} (already downloaded, size match: {item['size']} bytes)")
        return name, True

    url = item.get("download_url") or f"{RAW_BASE}/{name.replace(' ', '%20')}"
    print(f"Downloading {name} ({item['size'] / (1024*1024):.2f} MB)...")
    
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"Finished {name} ({dest_path.stat().st_size} bytes)")
            return name, True
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {name}: {e}")
            if dest_path.exists():
                dest_path.unlink()
    return name, False

def main():
    dest_dir = Path("c:/Users/user/Desktop/microplastics_demo/data/dataset_microplastic")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Fetching repository contents listing...")
    r = requests.get(REPO_API, timeout=30)
    r.raise_for_status()
    items = r.json()
    
    zip_items = [it for it in items if it["name"].endswith(".zip")]
    print(f"Found {len(zip_items)} zip files to download.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_file, it, dest_dir) for it in zip_items]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
    print(f"Download completed! {sum(1 for _, ok in results if ok)}/{len(zip_items)} succeeded.")

if __name__ == "__main__":
    main()
