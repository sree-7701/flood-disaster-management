"""
02_train_severity_estimation.py
STAGE 2: Flood Severity Estimation Model
Predicts Low/Medium/High severity based on damage features
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

print("="*80)
print("🚀 FLOOD SEVERITY ESTIMATION - STAGE 2")
print("="*80)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n📱 Using device: {DEVICE}")

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'num_classes': 3,
    'class_names': ['Low', 'Medium', 'High'],
    'output_dir': './outputs',
    'hidden_dims': [128, 64, 32]
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================
# SEVERITY ESTIMATOR MODEL
# ============================================

class SeverityEstimator(nn.Module):
    """MLP for severity estimation"""
    
    def __init__(self, input_dim=3, hidden_dims=[128, 64, 32], num_classes=3):
        super(SeverityEstimator, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============================================
# GENERATE SYNTHETIC DATA
# ============================================

def generate_severity_data(num_samples=2000):
    """
    Generate synthetic severity data based on damage features
    
    Features:
    1. buildings_affected (0-500)
    2. population_affected (0-50000)
    3. area_affected_km2 (0-100)
    
    Severity levels:
    - Low: small scale damage
    - Medium: moderate damage
    - High: severe damage
    """
    
    print("\n📊 Generating synthetic severity data...")
    
    np.random.seed(42)
    
    data = []
    labels = []
    
    for _ in range(num_samples):
        # Generate random features
        buildings = np.random.randint(0, 500)
        population = np.random.randint(0, 50000)
        area = np.random.randint(0, 100)
        
        # Determine severity based on features
        # Low severity: few buildings, small population, small area
        if buildings < 50 and population < 1000 and area < 10:
            label = 0  # Low
            # Add some noise
            if np.random.random() < 0.1:
                label = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
        
        # High severity: many buildings, large population, large area
        elif buildings > 200 and population > 20000 and area > 50:
            label = 2  # High
            if np.random.random() < 0.1:
                label = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])
        
        # Medium severity: in between
        else:
            label = 1  # Medium
            if np.random.random() < 0.1:
                label = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
        
        data.append([buildings, population, area])
        labels.append(label)
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Print distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"📊 Severity Distribution:")
    for cls, count in zip(unique, counts):
        print(f"   {CONFIG['class_names'][cls]}: {count} ({count/len(labels)*100:.1f}%)")
    
    return data, labels

# ============================================
# TRAINING FUNCTION
# ============================================

def train_model(model, train_loader, val_loader, criterion, optimizer):
    """Train the severity estimation model"""
    
    print("\n" + "="*80)
    print("🏋️ TRAINING")
    print("="*80)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        for X_batch, y_batch in progress:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
            
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
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], 'best_severity_model.pt'))
            print(f"✅ Best model saved! Val Acc: {val_acc:.4f}")
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}:")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
            print("-"*40)
    
    return train_losses, val_losses, train_accs, val_accs

# ============================================
# EVALUATION
# ============================================

def evaluate_model(model, test_loader):
    """Evaluate the severity estimation model"""
    
    print("\n" + "="*80)
    print("📈 EVALUATION ON TEST SET")
    print("="*80)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
    print(f"✅ Test F1 Score: {f1:.4f}")
    
    print("\n📊 Confusion Matrix:")
    print(cm)
    
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CONFIG['class_names'], zero_division=0))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CONFIG['class_names'],
                yticklabels=CONFIG['class_names'])
    plt.title('Confusion Matrix - Severity Estimation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(CONFIG['output_dir'], 'severity_confusion_matrix.png'))
    plt.show()
    
    return accuracy, f1, cm

# ============================================
# PLOT TRAINING HISTORY
# ============================================

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    
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
    plt.savefig(os.path.join(CONFIG['output_dir'], 'severity_training_history.png'))
    plt.show()

# ============================================
# MAIN
# ============================================

def main():
    # Generate data
    X, y = generate_severity_data(num_samples=2000)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Initialize model
    print("\n" + "="*80)
    print("🤖 INITIALIZING MODEL")
    print("="*80)
    model = SeverityEstimator(
        input_dim=3,
        hidden_dims=CONFIG['hidden_dims'],
        num_classes=CONFIG['num_classes']
    )
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Train
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Load best model
    model_path = os.path.join(CONFIG['output_dir'], 'best_severity_model.pt')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    # Evaluate
    accuracy, f1, cm = evaluate_model(model, test_loader)
    
    # Save results
    results = {
        'test_accuracy': float(accuracy),
        'test_f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_names': CONFIG['class_names']
    }
    
    with open(os.path.join(CONFIG['output_dir'], 'severity_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {CONFIG['output_dir']}/")
    print("   - best_severity_model.pt")
    print("   - severity_results.json")
    print("   - severity_confusion_matrix.png")
    print("   - severity_training_history.png")

if __name__ == "__main__":
    main()