"""
fusion/fusion.py
Fusion module combining Sen1Floods11 and DisasterM3
"""

import torch
import torch.nn as nn
import json
import numpy as np

class FeatureFusion(nn.Module):
    """Fuse flood segmentation features with disaster understanding features"""
    
    def __init__(self, flood_dim=128, disaster_dim=256, fused_dim=128):
        super(FeatureFusion, self).__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(flood_dim + disaster_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU()
        )
    
    def forward(self, flood_features, disaster_features):
        combined = torch.cat([flood_features, disaster_features], dim=-1)
        return self.fusion(combined)

class DecisionEngine:
    """Combine fusion results into resource recommendations"""
    
    def __init__(self):
        self.resource_matrix = {
            'critical': {'teams': 10, 'boats': 15, 'medical': 6, 'shelters': 15, 'food': 500, 'water': 10000},
            'high': {'teams': 6, 'boats': 8, 'medical': 3, 'shelters': 8, 'food': 300, 'water': 4000},
            'medium': {'teams': 3, 'boats': 4, 'medical': 1, 'shelters': 4, 'food': 150, 'water': 2000},
            'low': {'teams': 1, 'boats': 2, 'medical': 1, 'shelters': 2, 'food': 50, 'water': 1000}
        }
    
    def recommend(self, flood_extent, damage_class, severity):
        """Generate resource recommendations based on fused features"""
        
        # Calculate priority score
        score = 0
        if flood_extent > 50:
            score += 2
        elif flood_extent > 20:
            score += 1
        
        damage_scores = {'intact': 0, 'damaged': 1, 'destroyed': 2}
        severity_scores = {'low': 0, 'medium': 1, 'high': 2}
        
        score += damage_scores.get(damage_class, 0)
        score += severity_scores.get(severity, 0)
        
        if score >= 4:
            level = 'critical'
        elif score >= 2:
            level = 'high'
        elif score >= 1:
            level = 'medium'
        else:
            level = 'low'
        
        resources = self.resource_matrix[level].copy()
        return resources, level

def fuse_and_recommend(flood_extent, damage_class, severity):
    """Complete fusion pipeline"""
    
    engine = DecisionEngine()
    resources, level = engine.recommend(flood_extent, damage_class, severity)
    
    report = {
        'flood_extent': flood_extent,
        'damage_class': damage_class,
        'severity': severity,
        'priority': level,
        'resources': resources,
        'timestamp': str(__import__('datetime').datetime.now())
    }
    
    return report

if __name__ == "__main__":
    # Example usage
    report = fuse_and_recommend(75.0, 'destroyed', 'high')
    print("="*70)
    print("📋 FUSION REPORT")
    print("="*70)
    print(f"Flood Extent: {report['flood_extent']:.1f}%")
    print(f"Damage Class: {report['damage_class']}")
    print(f"Severity: {report['severity']}")
    print(f"Priority: {report['priority']}")
    print(f"\n📦 Resources:")
    for key, value in report['resources'].items():
        print(f"   {key.title()}: {value}")