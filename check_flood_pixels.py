"""
check_flood_pixels.py
Check if the dataset has any flood pixels
"""

import numpy as np
import os

DATA_DIR = r'C:\Users\SREELAKSHMI L\Desktop\DisasterM3_Project\datasets\Sen1Floods11_npy\train'
files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npz')]

print("="*60)
print("🔍 CHECKING FLOOD PIXELS IN DATASET")
print("="*60)
print(f"📂 Checking {len(files)} files...\n")

total_flood = 0
total_pixels = 0
samples_with_flood = 0

for f in files[:20]:
    data = np.load(os.path.join(DATA_DIR, f))
    mask = data['mask']
    flood = (mask == 1).sum()
    total = mask.size
    total_flood += flood
    total_pixels += total
    if flood > 0:
        samples_with_flood += 1
    print(f"  {f}: {flood}/{total} ({flood/total*100:.2f}%)")

print("\n" + "="*60)
print("📊 Summary (first 20 samples):")
print(f"   Samples with flood: {samples_with_flood}/20")
if samples_with_flood > 0:
    print(f"   ✅ Dataset has flood pixels!")
else:
    print(f"   ❌ No flood pixels found in first 20 samples!")
print("="*60)