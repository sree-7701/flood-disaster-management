"""
load_data.py
Load real Sen1Floods11 .npz files with memory optimization
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F

DATA_DIR = r'C:\Users\SREELAKSHMI L\Desktop\DisasterM3_Project\datasets\Sen1Floods11_npy'

class FloodDataset(Dataset):
    def __init__(self, split='train', image_size=256):
        self.data_dir = os.path.join(DATA_DIR, split)
        self.files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.image_size = image_size
        print(f"✅ Loaded {len(self.files)} {split} samples")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path)
        
        image = torch.FloatTensor(data['image'])
        mask = torch.LongTensor(data['mask'])
        
        # Resize to 256x256 to save memory
        if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            image = F.interpolate(image.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear').squeeze(0)
            mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(self.image_size, self.image_size), mode='nearest').squeeze(0).squeeze(0).long()
        
        return image, mask

def get_loaders(batch_size=4, image_size=256):
    """
    Get data loaders with memory optimization
    
    Args:
        batch_size: Number of samples per batch (default 4, reduce to 2 if OOM)
        image_size: Size to resize images (default 256)
    """
    train_dataset = FloodDataset('train', image_size=image_size)
    val_dataset = FloodDataset('val', image_size=image_size)
    test_dataset = FloodDataset('test', image_size=image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def get_class_weights():
    """Calculate class weights for imbalance"""
    train_dataset = FloodDataset('train')
    all_labels = []
    
    print("\n📊 Computing class weights...")
    for idx in range(len(train_dataset)):
        _, mask = train_dataset[idx]
        all_labels.extend(mask.numpy().flatten())
    
    all_labels = np.array(all_labels)
    classes = np.unique(all_labels)
    weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    
    print(f"   Classes: {classes}")
    for cls, weight in zip(classes, weights):
        print(f"   Class {cls}: {weight:.4f}")
    
    return torch.FloatTensor(weights)

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING DATA LOADER")
    print("="*60)
    
    train_loader, val_loader, test_loader = get_loaders(batch_size=4, image_size=256)
    
    print(f"\n📊 DataLoader Summary:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    print("\n🧪 Testing one batch...")
    for images, masks in train_loader:
        print(f"   Images shape: {images.shape}")
        print(f"   Masks shape: {masks.shape}")
        print(f"   Unique mask values: {torch.unique(masks)}")
        break