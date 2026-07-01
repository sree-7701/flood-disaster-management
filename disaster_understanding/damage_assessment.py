"""
01_train_damage_assessment_fixed.py
STAGE 1: Flood Damage Assessment Model (No Internet Required)
Uses a simple CNN instead of downloading EfficientNet weights
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

print("="*80)
print("🚀 FLOOD DAMAGE ASSESSMENT MODEL - STAGE 1 (NO INTERNET)")
print("="*80)

# ============================================
# CONFIGURATION
# ============================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n📱 Using device: {DEVICE}")

CONFIG = {
    'batch_size': 16,
    'num_epochs': 5,
    'learning_rate': 1e-3,
    'image_size': 128,  # Smaller for faster training
    'num_classes': 3,
    'class_names': ['Intact', 'Damaged', 'Destroyed'],
    'output_dir': './outputs',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================
# SIMPLE CNN MODEL (No Pretrained Weights Needed)
# ============================================

class SimpleDamageModel(nn.Module):
    """Simple CNN for damage assessment - no internet required"""
    
    def __init__(self, num_classes=3):
        super(SimpleDamageModel, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================
# DATASET
# ============================================

class FloodDamageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
        self.task_to_label = {
            'referring expression segmentation': 1,
            'disaster report': 1,
            'building damage counting': 2,
            'road damage counting': 2,
            'disaster restoration advice': 1,
            'disaster scene recognition': 1,
            'disaster type recognition': 0,
            'disaster bearing bodies recognition': 1,
            'relational reasoning': 0
        }
    
    def __len__(self):
        return len(self.data)
    
    def find_image(self, path):
        if not path:
            return None
        filename = os.path.basename(path)
        base_paths = [
            'DisasterM3/DisasterM3_Bench/test_images',
            'DisasterM3/DisasterM3_Bench',
            'DisasterM3',
            '.',
        ]
        for base in base_paths:
            test_path = os.path.join(base, filename)
            if os.path.exists(test_path):
                return test_path
        return None
    
    def get_label(self, item):
        task = item.get('task', '').lower()
        for key, label in self.task_to_label.items():
            if key in task:
                return label
        return 1
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        post_path = item.get('post_image_path', '')
        img_path = self.find_image(post_path)
        
        if img_path is None:
            return torch.zeros(3, CONFIG['image_size'], CONFIG['image_size']), torch.tensor(0)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            return torch.zeros(3, CONFIG['image_size'], CONFIG['image_size']), torch.tensor(0)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.get_label(item)
        return image, label

# ============================================
# LOAD DATA
# ============================================

def load_flood_data():
    print("\n" + "="*80)
    print("📊 LOADING DATA")
    print("="*80)
    
    flood_files = [
        'flood_referring_expression_segmentation.json',
        'flood_disaster_report.json',
        'flood_building_damage_counting.json',
    ]
    
    all_data = []
    
    for file in flood_files:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ Loaded {len(data)} samples from {file}")
                all_data.extend(data)
    
    if len(all_data) == 0:
        print("❌ No flood data files found!")
        return None
    
    print(f"✅ Total flood samples: {len(all_data)}")
    
    # Use only a subset for faster training
    np.random.seed(42)
    indices = np.random.permutation(len(all_data))
    all_data = [all_data[i] for i in indices[:2000]]  # Use 2000 samples
    
    train_idx = indices[:int(0.7 * len(all_data))]
    val_idx = indices[int(0.7 * len(all_data)):int(0.85 * len(all_data))]
    test_idx = indices[int(0.85 * len(all_data)):]
    
    train_data = [all_data[i] for i in train_idx]
    val_data = [all_data[i] for i in val_idx]
    test_data = [all_data[i] for i in test_idx]
    
    print(f"\n📊 Data Split (using 2000 samples):")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    
    return train_data, val_data, test_data

# ============================================
# TRANSFORMS
# ============================================

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# ============================================
# TRAINING
# ============================================

def train_model(model, train_loader, val_loader, criterion, optimizer):
    print("\n" + "="*80)
    print("🏋️ TRAINING")
    print("="*80)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        for images, labels in progress:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            progress.set_postfix({'loss': loss.item()})
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], 'best_model_simple.pt'))
            print(f"✅ Best model saved! Val Acc: {val_acc:.4f}")
        
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        print("-"*40)
    
    return train_losses, val_losses, train_accs, val_accs

# ============================================
# EVALUATION
# ============================================

def evaluate_model(model, test_loader):
    print("\n" + "="*80)
    print("📈 EVALUATION ON TEST SET")
    print("="*80)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if len(all_preds) == 0:
        print("❌ No test predictions!")
        return None
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
    print(f"✅ Test F1 Score: {f1:.4f}")
    
    print("\n📊 Confusion Matrix:")
    print(cm)
    
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CONFIG['class_names'], zero_division=0))
    
    results = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
    }
    
    with open(os.path.join(CONFIG['output_dir'], 'damage_assessment_results_simple.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CONFIG['class_names'],
                yticklabels=CONFIG['class_names'])
    plt.title('Confusion Matrix - Damage Assessment')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix_simple.png'))
    plt.show()
    
    return results

# ============================================
# PLOT
# ============================================

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(train_accs, label='Train Acc')
    axes[1].plot(val_accs, label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'training_history_simple.png'))
    plt.show()

# ============================================
# MAIN
# ============================================

def main():
    data = load_flood_data()
    if data is None:
        print("\n❌ Please run: python extract_flood_tasks.py")
        return
    
    train_data, val_data, test_data = data
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = FloodDamageDataset(train_data, transform=train_transform)
    val_dataset = FloodDamageDataset(val_data, transform=val_transform)
    test_dataset = FloodDamageDataset(test_data, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    print("\n" + "="*80)
    print("🤖 INITIALIZING MODEL (No Internet Required)")
    print("="*80)
    model = SimpleDamageModel(num_classes=CONFIG['num_classes'])
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer
    )
    
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    best_model_path = os.path.join(CONFIG['output_dir'], 'best_model_simple.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    results = evaluate_model(model, test_loader)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {CONFIG['output_dir']}/")

if __name__ == "__main__":
    main()