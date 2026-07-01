"""
add_location_to_npz.py
Add geographic metadata (bounds, center coordinates) to existing .npz files
"""

import os
import numpy as np
import rasterio
from tqdm import tqdm

# Paths
DATA_DIR = r'C:\Users\SREELAKSHMI L\Desktop\DisasterM3_Project\datasets\Sen1Floods11_data\v1.1'
NPZ_DIR = r'C:\Users\SREELAKSHMI L\Desktop\DisasterM3_Project\datasets\Sen1Floods11_npy'

print("="*70)
print("📍 ADDING LOCATION DATA TO .NPZ FILES")
print("="*70)

def get_location_data(event, chip_id):
    """Extract location from original .tif file"""
    
    base_path = os.path.join(DATA_DIR, 'data', 'flood_events', 'HandLabeled')
    s1_path = os.path.join(base_path, 'S1Hand', f'{event}_{chip_id}_S1Hand.tif')
    
    if not os.path.exists(s1_path):
        return None
    
    try:
        with rasterio.open(s1_path) as src:
            bounds = src.bounds  # (left, bottom, right, top)
            crs = str(src.crs)
            
            # Calculate center
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            
            return {
                'bounds': bounds,
                'crs': crs,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'left': bounds[0],
                'bottom': bounds[1],
                'right': bounds[2],
                'top': bounds[3]
            }
    except Exception as e:
        return None

# Process all .npz files
total_updated = 0

for split in ['train', 'val', 'test']:
    split_dir = os.path.join(NPZ_DIR, split)
    if not os.path.exists(split_dir):
        continue
    
    files = [f for f in os.listdir(split_dir) if f.endswith('.npz')]
    print(f"\n📂 Processing {split} ({len(files)} files)...")
    
    for filename in tqdm(files, desc=f"Updating {split}"):
        file_path = os.path.join(split_dir, filename)
        
        # Load existing data
        data = np.load(file_path, allow_pickle=True)
        
        # Check if location data already exists
        if 'center_lat' in data:
            continue
        
        # Parse event and chip_id from filename
        parts = filename.replace('.npz', '').split('_')
        if len(parts) >= 2:
            event = parts[0]
            chip_id = parts[1]
        else:
            continue
        
        # Get location data
        loc = get_location_data(event, chip_id)
        if loc is None:
            continue
        
        # Create new data with location
        new_data = {
            'image': data['image'],
            'mask': data['mask'],
            'event': data['event'],
            'chip_id': data['chip_id'],
            'bounds': loc['bounds'],
            'crs': loc['crs'],
            'center_lat': loc['center_lat'],
            'center_lon': loc['center_lon'],
            'left': loc['left'],
            'bottom': loc['bottom'],
            'right': loc['right'],
            'top': loc['top']
        }
        
        # Save updated file
        np.savez_compressed(file_path, **new_data)
        total_updated += 1

print("\n" + "="*70)
print("✅ LOCATION DATA ADDED!")
print("="*70)
print(f"📊 Files updated: {total_updated}")
print("="*70)