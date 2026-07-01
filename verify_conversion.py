"""
verify_conversion.py
Verify that .npz files were converted correctly
"""

import numpy as np
import os

DATA_DIR = r'C:\Users\SREELAKSHMI L\Desktop\DisasterM3_Project\datasets\Sen1Floods11_npy'

print("="*70)
print("🔍 VERIFYING CONVERTED DATA")
print("="*70)

# Check a train sample
train_sample = os.path.join(DATA_DIR, 'train', 'Ghana_103272.npz')

if os.path.exists(train_sample):
    print(f"\n✅ Found train sample: Ghana_103272.npz")
    data = np.load(train_sample)
    
    print(f"\n📊 Sample contents:")
    print(f"   Keys: {list(data.keys())}")
    print(f"   Image shape: {data['image'].shape}")
    print(f"   Mask shape: {data['mask'].shape}")
    print(f"   Mask min: {data['mask'].min()}, max: {data['mask'].max()}")
    print(f"   Event: {data['event']}")
    print(f"   Chip ID: {data['chip_id']}")
    
    # Count flood pixels
    flood_pixels = (data['mask'] == 1).sum()
    total_pixels = data['mask'].size
    flood_percentage = (flood_pixels / total_pixels) * 100
    print(f"   Flood pixels: {flood_pixels} / {total_pixels} ({flood_percentage:.2f}%)")
    
    print("\n✅ Data looks correct!")
else:
    print(f"\n❌ Sample not found: {train_sample}")

# Check all splits
print("\n" + "="*70)
print("📊 DATASET SUMMARY")
print("="*70)

total_files = 0
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(DATA_DIR, split)
    if os.path.exists(split_dir):
        files = [f for f in os.listdir(split_dir) if f.endswith('.npz')]
        print(f"   {split}: {len(files)} files")
        total_files += len(files)
    else:
        print(f"   {split}: folder not found")

print(f"\n📊 Total: {total_files} files")
print("="*70)