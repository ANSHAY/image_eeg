import torch
import os
import time
from datasets import load_dataset, Image

def main():
    pth_path = "data/raw/spampinato/eeg_signals_raw_with_mean_std.pth"
    dest_dir = "data/raw/imagenet_stimuli"
    os.makedirs(dest_dir, exist_ok=True)
    
    print("Loading target filenames from Spampinato .pth...")
    try:
        d = torch.load(pth_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading {pth_path}: {e}")
        return
        
    targets = set([name.replace(".JPEG", "").replace(".jpg", "") for name in d['images']])
    total_targets = len(targets)
    
    # Remove already downloaded images
    existing_files = set(os.listdir(dest_dir))
    targets = targets - existing_files
    
    print(f"Total needed: {total_targets}. Already have: {len(existing_files)}. Remaining to fetch: {len(targets)}")
    if len(targets) == 0:
        print("All images downloaded successfully!")
        return

    print("Scanning public ImageNet stream...")
    
    max_retries = 10
    for attempt in range(max_retries):
        try:
            ds = load_dataset('visual-layer/imagenet-1k-vl-enriched', split='train', streaming=True)
            
            # EXTREMELY IMPORTANT: Do not decode images, avoid PIL crashes!
            ds = ds.cast_column('image', Image(decode=False))
            
            # Skip the first 400,000 rows to save massive network bandwidth
            print("Skipping first 400,000 rows to save ~50GB of network bandwidth...")
            ds = ds.skip(400000)
            
            found = 0
            for row in ds:
                img_id = row['image_id']
                if img_id in targets:
                    img_bytes = row['image']['bytes']
                    out_path = os.path.join(dest_dir, f"{img_id}")
                    with open(out_path, 'wb') as f:
                        f.write(img_bytes)
                        
                    targets.remove(img_id)
                    found += 1
                    if found % 50 == 0 or len(targets) == 0:
                        print(f"Progress: {len(existing_files) + found}/{total_targets} images...")
                    
                    if len(targets) == 0:
                        print("All images downloaded successfully from train split!")
                        return
            
            break
            
        except Exception as e:
            print(f"Network error during stream (attempt {attempt+1}/{max_retries}): {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

    if len(targets) > 0:
        print(f"Warning: {len(targets)} images missing. Checking validation set...")
        try:
            ds_val = load_dataset('visual-layer/imagenet-1k-vl-enriched', split='validation', streaming=True)
            ds_val = ds_val.cast_column('image', Image(decode=False))
            
            for row in ds_val:
                img_id = row['image_id']
                if img_id in targets:
                    img_bytes = row['image']['bytes']
                    out_path = os.path.join(dest_dir, f"{img_id}")
                    with open(out_path, 'wb') as f:
                        f.write(img_bytes)
                        
                    targets.remove(img_id)
                    if len(targets) == 0:
                        print("All images downloaded successfully!")
                        return
        except Exception as e:
            print(f"Network error reading validation set: {e}")

    print(f"Finished. Remaining missing images: {len(targets)}")

if __name__ == "__main__":
    main()
