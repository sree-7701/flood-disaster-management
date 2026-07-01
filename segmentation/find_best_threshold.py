"""
find_best_threshold.py
Test different thresholds to find optimal IoU/Dice
"""

import torch
import numpy as np
from sklearn.metrics import jaccard_score, f1_score
from tqdm import tqdm
import json

from unet_model import UNet
from load_data import get_loaders

def evaluate_threshold(model, test_loader, device, threshold):
    """Evaluate model at a specific threshold"""
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > threshold).cpu().numpy().flatten()
            masks_flat = masks.cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_masks.extend(masks_flat)
    
    iou = jaccard_score(all_masks, all_preds, average='binary', zero_division=0)
    dice = f1_score(all_masks, all_preds, average='binary', zero_division=0)
    
    return iou, dice

def main():
    print("="*70)
    print("🔍 FINDING BEST THRESHOLD")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    # Load model
    model = UNet(in_channels=5, out_channels=1).to(device)
    model.load_state_dict(torch.load('models/unet_real_best.pth', map_location=device))
    model.eval()
    print("✅ Model loaded")
    
    # Get test data
    _, _, test_loader = get_loaders()
    
    # Test different thresholds
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    results = []
    print("\n📊 Testing thresholds...")
    print("-"*60)
    print(f"{'Threshold':<12} {'IoU':<10} {'Dice':<10}")
    print("-"*60)
    
    for thresh in thresholds:
        iou, dice = evaluate_threshold(model, test_loader, device, thresh)
        results.append({'threshold': thresh, 'iou': iou, 'dice': dice})
        print(f"{thresh:<12.2f} {iou:<10.4f} {dice:<10.4f}")
    
    # Find best
    best_iou = max(results, key=lambda x: x['iou'])
    best_dice = max(results, key=lambda x: x['dice'])
    
    print("-"*60)
    print(f"\n✅ Best IoU: {best_iou['iou']:.4f} at threshold {best_iou['threshold']:.2f}")
    print(f"✅ Best Dice: {best_dice['dice']:.4f} at threshold {best_dice['threshold']:.2f}")
    
    # Save results
    with open('threshold_results.json', 'w') as f:
        json.dump({
            'results': results,
            'best_iou': best_iou,
            'best_dice': best_dice
        }, f, indent=2)
    
    print("\n💾 Results saved to: threshold_results.json")

if __name__ == "__main__":
    main()