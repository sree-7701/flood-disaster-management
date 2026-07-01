"""
check_label_values.py
Check actual label values in .tif files
"""

import rasterio
import numpy as np
import os

# Check a sample label file
label_path = r'C:\Users\SREELAKSHMI L\Desktop\DisasterM3_Project\datasets\Sen1Floods11_data\v1.1\data\flood_events\HandLabeled\LabelHand\Ghana_103272_LabelHand.tif'

print("="*60)
print("🔍 CHECKING LABEL VALUES")
print("="*60)

if os.path.exists(label_path):
    with rasterio.open(label_path) as src:
        label = src.read(1)
        print(f"✅ File: Ghana_103272_LabelHand.tif")
        print(f"   Shape: {label.shape}")
        print(f"   Min value: {label.min()}")
        print(f"   Max value: {label.max()}")
        print(f"   Unique values: {np.unique(label)}")
        print(f"\n📊 Value counts:")
        print(f"   Value 2 (water): {(label == 2).sum()}")
        print(f"   Value 1: {(label == 1).sum()}")
        print(f"   Value 0: {(label == 0).sum()}")
        print(f"   Value -1: {(label == -1).sum()}")
else:
    print(f"❌ File not found")