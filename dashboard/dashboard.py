"""
dashboard.py
UNIFIED FLOOD DISASTER MANAGEMENT DASHBOARD
Combines flood segmentation + resource recommendations + damage assessment
"""

import os
import sys
import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'segmentation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fusion'))

from unet_model import UNet
from load_data import get_loaders
from fusion import fuse_and_recommend

print("="*70)
print("🌊 UNIFIED FLOOD DISASTER MANAGEMENT DASHBOARD")
print("="*70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Using device: {device}")

# ============================================
# 1. LOAD SEGMENTATION MODEL
# ============================================

print("\n📥 Loading Segmentation Model...")
model = UNet(in_channels=5, out_channels=1).to(device)

# Try multiple paths
model_paths = [
    'models/unet_real_best.pth',
    'datasets/Sen1Floods11/models/unet_real_best.pth',
    '../models/unet_real_best.pth',
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    print("❌ Model not found!")
    print("   Looking in:", model_paths)
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Model loaded: {model_path}")

# ============================================
# 2. GET DATA
# ============================================

try:
    _, _, test_loader = get_loaders()
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# ============================================
# 3. GENERATE PREDICTIONS & RECOMMENDATIONS
# ============================================

print("\n🔍 Generating flood predictions and recommendations...")
predictions = []
flood_areas = []
all_reports = []

# Simulated DisasterM3 data (replace with actual when available)
damage_classes = ['damaged', 'destroyed', 'damaged', 'intact', 'destroyed']
severities = ['high', 'medium', 'high', 'low', 'medium']

with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
        if i >= 5:
            break
        
        images = images.to(device)
        outputs = model(images)
        
        for j in range(outputs.shape[0]):
            # Use optimal threshold 0.60
            pred = (torch.sigmoid(outputs[j]) > 0.60).cpu().numpy().squeeze()
            mask = masks[j].cpu().numpy().squeeze()
            
            if pred.ndim == 3:
                pred = pred[0]
            if mask.ndim == 3:
                mask = mask[0]
            
            flood_pct = float((pred > 0.60).sum() / pred.size * 100)
            
            # Get recommendation
            damage_class = damage_classes[(i * outputs.shape[0] + j) % len(damage_classes)]
            severity = severities[(i * outputs.shape[0] + j) % len(severities)]
            report = fuse_and_recommend(flood_pct, damage_class, severity)
            
            predictions.append({
                'pred': pred, 
                'gt': mask, 
                'flood_pct': flood_pct,
                'report': report
            })
            flood_areas.append(flood_pct)
            all_reports.append(report)
            
            print(f"   Sample {i*outputs.shape[0] + j + 1}: Flood = {flood_pct:.1f}%, Priority: {report['priority']}")
            
            if len(predictions) >= 5:
                break
        if len(predictions) >= 5:
            break

# ============================================
# 4. VISUALIZATION (4 columns)
# ============================================

print("\n📊 Creating unified visualization...")
num_samples = min(len(predictions), 5)

if num_samples == 0:
    print("❌ No predictions generated!")
    exit()

fig, axes = plt.subplots(num_samples, 4, figsize=(20, 4*num_samples))
if num_samples == 1:
    axes = axes.reshape(1, -1)

for i in range(num_samples):
    p = predictions[i]
    report = p['report']
    
    # Column 1: Predicted Mask
    axes[i, 0].imshow(p['pred'], cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[i, 0].set_title(f'Sample {i+1}: Flood Mask\n{p["flood_pct"]:.1f}%')
    axes[i, 0].axis('off')
    
    # Column 2: Ground Truth
    axes[i, 1].imshow(p['gt'], cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[i, 1].set_title('Ground Truth')
    axes[i, 1].axis('off')
    
    # Column 3: Assessment Info
    axes[i, 2].text(0.5, 0.85, f"Damage: {report['damage_class'].upper()}", 
                    ha='center', va='center', fontsize=12, fontweight='bold')
    axes[i, 2].text(0.5, 0.65, f"Severity: {report['severity'].upper()}", 
                    ha='center', va='center', fontsize=12)
    axes[i, 2].text(0.5, 0.45, f"Priority: {report['priority']}", 
                    ha='center', va='center', fontsize=14, 
                    color='red' if report['priority'] == 'critical' else 'orange')
    axes[i, 2].set_title('Disaster Assessment')
    axes[i, 2].axis('off')
    
    # Column 4: Resources
    resources_text = "\n".join([f"{k.title()}: {v}" for k, v in report['resources'].items()])
    axes[i, 3].text(0.1, 0.95, "RESOURCES:", ha='left', va='top', fontsize=11, fontweight='bold')
    axes[i, 3].text(0.1, 0.85, resources_text, ha='left', va='top', fontsize=10)
    axes[i, 3].set_title('Resource Recommendations')
    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig('unified_dashboard_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("💾 Saved: unified_dashboard_results.png")

# ============================================
# 5. SUMMARY
# ============================================

summary = {
    'samples': num_samples,
    'average_flood': sum(flood_areas[:num_samples]) / num_samples,
    'threshold': 0.60,
    'reports': all_reports[:num_samples],
    'model': 'U-Net (Sen1Floods11)',
    'epochs': 50
}

with open('unified_dashboard_report.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n💾 Saved: unified_dashboard_report.json")

# ============================================
# 6. PRINT SUMMARY
# ============================================

print("\n" + "="*70)
print("✅ UNIFIED DASHBOARD COMPLETE!")
print("="*70)
print(f"\n📊 Results:")
print(f"   Samples Analyzed: {num_samples}")
print(f"   Average Flood Area: {summary['average_flood']:.2f}%")
print(f"\n📁 Generated Files:")
print(f"   1. unified_dashboard_results.png - Complete visualization")
print(f"   2. unified_dashboard_report.json - Full report")
print(f"\n📋 Sample Recommendation:")
if all_reports:
    sample = all_reports[0]
    print(f"   Priority: {sample['priority']}")
    print(f"   Resources:")
    for key, value in sample['resources'].items():
        print(f"      {key.title()}: {value}")
print("\n" + "="*70)
print("🚀 Dashboard Ready!")
print("="*70)