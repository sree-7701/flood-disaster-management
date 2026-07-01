"""
convert_tif_to_npy_fixed.py
Convert .tif to .npz with CORRECT water value (1 = water)
"""

import os
import numpy as np
import rasterio
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Paths
DATA_DIR = r'C:\Users\SREELAKSHMI L\Desktop\DisasterM3_Project\datasets\Sen1Floods11_data\v1.1'
OUTPUT_DIR = r'C:\Users\SREELAKSHMI L\Desktop\DisasterM3_Project\datasets\Sen1Floods11_npy'

print("="*70)
print("📥 CONVERTING .TIF → .NPY (FIXED: WATER = 1)")
print("="*70)
print(f"📂 Input:  {DATA_DIR}")
print(f"📂 Output: {OUTPUT_DIR}")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

def convert_sample(event, chip_id, output_split):
    """Convert a single sample"""
    
    base_path = os.path.join(DATA_DIR, 'data', 'flood_events', 'HandLabeled')
    
    s1_path = os.path.join(base_path, 'S1Hand', f'{event}_{chip_id}_S1Hand.tif')
    s2_path = os.path.join(base_path, 'S2Hand', f'{event}_{chip_id}_S2Hand.tif')
    label_path = os.path.join(base_path, 'LabelHand', f'{event}_{chip_id}_LabelHand.tif')
    
    if not os.path.exists(s1_path) or not os.path.exists(s2_path) or not os.path.exists(label_path):
        return None
    
    try:
        with rasterio.open(s1_path) as src:
            s1 = src.read()
        with rasterio.open(s2_path) as src:
            s2 = src.read()
        with rasterio.open(label_path) as src:
            label = src.read(1)
        
        # Normalize
        s1 = np.nan_to_num(s1, nan=0.0, posinf=1.0, neginf=-1.0)
        s1_min = s1.min()
        s1_max = s1.max()
        if s1_max > s1_min:
            s1 = (s1 - s1_min) / (s1_max - s1_min + 1e-8)
        else:
            s1 = np.zeros_like(s1)
        
        s2 = np.nan_to_num(s2, nan=0.0, posinf=1.0, neginf=-1.0)
        s2_min = s2.min()
        s2_max = s2.max()
        if s2_max > s2_min:
            s2 = (s2 - s2_min) / (s2_max - s2_min + 1e-8)
        else:
            s2 = np.zeros_like(s2)
        
        # Combine: SAR (2) + RGB (3) = 5 channels
        rgb = s2[:3]
        combined = np.concatenate([s1, rgb], axis=0)
        
        # FIXED: water = 1 (not 2)
        label_binary = (label == 1).astype(np.float32)
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, output_split, f'{event}_{chip_id}.npz')
        np.savez_compressed(output_path,
            image=combined,
            mask=label_binary,
            event=event,
            chip_id=chip_id
        )
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {event}_{chip_id} - {e}")
        return None

# Process splits
SPLIT_MAP = {'train': 'train', 'valid': 'val', 'test': 'test'}

print("\n📄 Processing split files...")

total_converted = 0
total_samples = 0

for csv_split, output_split in SPLIT_MAP.items():
    csv_path = os.path.join(DATA_DIR, 'splits', 'flood_handlabeled', f'flood_{csv_split}_data.csv')
    
    if not os.path.exists(csv_path):
        print(f"⚠️ No {csv_split} file found")
        continue
    
    df = pd.read_csv(csv_path, header=None)
    total = len(df)
    converted = 0
    
    print(f"\n📂 Converting {csv_split} → {output_split} ({total} samples)...")
    
    for idx, row in tqdm(df.iterrows(), total=total):
        s1_file = str(row[0]).strip()
        parts = s1_file.split('_')
        
        if len(parts) >= 3:
            event = parts[0]
            chip_id = parts[1]
        else:
            continue
        
        result = convert_sample(event, chip_id, output_split)
        if result:
            converted += 1
    
    total_converted += converted
    total_samples += total
    print(f"   ✅ Converted {converted}/{total} ({converted/total*100:.1f}%)")

print("\n" + "="*70)
print("✅ CONVERSION COMPLETE!")
print("="*70)
print(f"📊 Total samples: {total_samples}")
print(f"📊 Converted: {total_converted}")
print(f"📂 Output: {OUTPUT_DIR}")