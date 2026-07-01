"""
evaluate_real.py
Evaluate trained model on test set
"""

import torch
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
from tqdm import tqdm
import json

from unet_model import UNet
from load_data import get_loaders

def evaluate():
    print("="*60)
    print("📊 EVALUATING U-NET ON REAL DATA")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    # Load model
    model = UNet(in_channels=5, out_channels=1).to(device)
    model.load_state_dict(torch.load('models/unet_real_best.pth', map_location=device))
    model.eval()
    print("✅ Model loaded")
    
    # Get test data
    _, _, test_loader = get_loaders()
    
    # Evaluate
    all_preds = []
    all_masks = []
    
    print("\n🔍 Running evaluation...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.6).cpu().numpy().flatten()
            masks_flat = masks.cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_masks.extend(masks_flat)
    
    # Metrics
    iou = jaccard_score(all_masks, all_preds, average='binary', zero_division=0)
    dice = f1_score(all_masks, all_preds, average='binary', zero_division=0)
    accuracy = accuracy_score(all_masks, all_preds)
    
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"   IoU: {iou:.4f}")
    print(f"   Dice: {dice:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print("="*60)
    
    # Save results
    results = {
        'iou': float(iou),
        'dice': float(dice),
        'accuracy': float(accuracy),
        'model': 'UNet',
        'dataset': 'Sen1Floods11_real'
    }
    
    with open('evaluation_results_real.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Results saved to: evaluation_results_real.json")

if __name__ == "__main__":
    evaluate()