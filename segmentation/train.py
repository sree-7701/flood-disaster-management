"""
train.py
Train U-Net on real Sen1Floods11 data with memory optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from unet_model import UNet
from load_data import get_loaders, get_class_weights

def train():
    print("="*70)
    print("🏋️ U-NET TRAINING ON REAL SEN1FLOODS11 DATA")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    # Clear GPU cache
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Get data loaders with smaller batch and image size
    train_loader, val_loader, test_loader = get_loaders(batch_size=4, image_size=256)
    
    # Get class weights for imbalance
    class_weights = get_class_weights()
    class_weights = class_weights.to(device)
    
    # For binary classification, use pos_weight
    pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
    print(f"📊 Positive class weight: {pos_weight:.4f}")
    
    # Model
    model = UNet(in_channels=5, out_channels=1).to(device)
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Mixed precision scaler (saves memory)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
    num_epochs = 50
    best_loss = float('inf')
    
    # Training history
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []
    
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, masks in progress:
            images = images.to(device)
            masks = masks.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.float().to(device).unsqueeze(1)
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten()
                masks_flat = masks.cpu().numpy().flatten()
                
                all_preds.extend(preds)
                all_masks.extend(masks_flat)
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        iou = jaccard_score(all_masks, all_preds, average='binary', zero_division=0)
        dice = f1_score(all_masks, all_preds, average='binary', zero_division=0)
        val_ious.append(iou)
        val_dices.append(dice)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  IoU: {iou:.4f}")
        print(f"  Dice: {dice:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/unet_real_best.pth')
            print(f"  ✅ Best model saved!")
    
    # Save final model
    torch.save(model.state_dict(), 'models/unet_real_final.pth')
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(val_ious, label='IoU', color='green')
    axes[1].plot(val_dices, label='Dice', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Segmentation Performance')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()
    
    print("\n✅ Training complete!")
    print(f"📁 Best model: models/unet_real_best.pth")
    print(f"📁 Final model: models/unet_real_final.pth")
    print(f"📁 Training plot: training_results.png")

if __name__ == "__main__":
    train()