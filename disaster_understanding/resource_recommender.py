"""
03_resource_recommendation_engine.py
STAGE 3: Resource Recommendation Engine
Recommends resources based on damage assessment + severity estimation
"""

import json
import os
import numpy as np
from typing import Dict, List

print("="*80)
print("🚀 RESOURCE RECOMMENDATION ENGINE - STAGE 3")
print("="*80)

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'output_dir': './outputs',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================
# RESOURCE ALLOCATION MATRIX
# ============================================

RESOURCE_MATRIX = {
    # damage_class: {severity: {resources}}
    'intact': {
        'low': {'rescue_teams': 0, 'boats': 0, 'medical_units': 0, 'shelters': 0, 'food_kits': 0, 'water_supply_liters': 0},
        'medium': {'rescue_teams': 0, 'boats': 0, 'medical_units': 0, 'shelters': 0, 'food_kits': 0, 'water_supply_liters': 0},
        'high': {'rescue_teams': 0, 'boats': 0, 'medical_units': 0, 'shelters': 0, 'food_kits': 0, 'water_supply_liters': 0}
    },
    'damaged': {
        'low': {'rescue_teams': 2, 'boats': 3, 'medical_units': 1, 'shelters': 2, 'food_kits': 50, 'water_supply_liters': 1000},
        'medium': {'rescue_teams': 4, 'boats': 6, 'medical_units': 2, 'shelters': 4, 'food_kits': 100, 'water_supply_liters': 2000},
        'high': {'rescue_teams': 6, 'boats': 8, 'medical_units': 3, 'shelters': 6, 'food_kits': 200, 'water_supply_liters': 4000}
    },
    'destroyed': {
        'low': {'rescue_teams': 3, 'boats': 5, 'medical_units': 2, 'shelters': 4, 'food_kits': 100, 'water_supply_liters': 2000},
        'medium': {'rescue_teams': 6, 'boats': 10, 'medical_units': 4, 'shelters': 8, 'food_kits': 250, 'water_supply_liters': 5000},
        'high': {'rescue_teams': 10, 'boats': 15, 'medical_units': 6, 'shelters': 12, 'food_kits': 500, 'water_supply_liters': 10000}
    }
}

# ============================================
# RECOMMENDATION ENGINE
# ============================================

class ResourceRecommendationEngine:
    """Intelligent resource allocation based on disaster assessment"""
    
    def __init__(self):
        """Initialize the resource recommendation engine"""
        self.resource_matrix = RESOURCE_MATRIX
    
    def recommend_resources(self, damage_class: str, severity: str, 
                           population_affected: int = 0,
                           buildings_affected: int = 0,
                           area_affected_km2: float = 0.0) -> Dict:
        """
        Recommend resources based on damage assessment and severity
        
        Args:
            damage_class: 'intact', 'damaged', or 'destroyed'
            severity: 'low', 'medium', or 'high'
            population_affected: Number of people affected
            buildings_affected: Number of buildings affected
            area_affected_km2: Area affected in square kilometers
        
        Returns:
            Dictionary with resource recommendations
        """
        
        # Normalize inputs
        damage_class = damage_class.lower()
        severity = severity.lower()
        
        # Get base allocation
        if damage_class in self.resource_matrix:
            if severity in self.resource_matrix[damage_class]:
                base_resources = self.resource_matrix[damage_class][severity].copy()
            else:
                base_resources = self.resource_matrix[damage_class]['medium'].copy()
        else:
            base_resources = self.resource_matrix['damaged']['medium'].copy()
        
        # Adjust based on population
        if population_affected > 0:
            population_factor = min(population_affected / 1000, 5.0)
            base_resources['food_kits'] = int(base_resources['food_kits'] * (1 + population_factor * 0.5))
            base_resources['water_supply_liters'] = int(base_resources['water_supply_liters'] * (1 + population_factor * 0.5))
            base_resources['shelters'] = int(base_resources['shelters'] * (1 + population_factor * 0.3))
            base_resources['medical_units'] = max(1, int(base_resources['medical_units'] * (1 + population_factor * 0.2)))
        
        # Adjust based on buildings affected
        if buildings_affected > 0:
            building_factor = min(buildings_affected / 100, 3.0)
            base_resources['rescue_teams'] = max(1, int(base_resources['rescue_teams'] * (1 + building_factor * 0.3)))
            base_resources['boats'] = max(1, int(base_resources['boats'] * (1 + building_factor * 0.2)))
        
        # Adjust based on area
        if area_affected_km2 > 0:
            area_factor = min(area_affected_km2 / 10, 3.0)
            base_resources['rescue_teams'] = max(1, int(base_resources['rescue_teams'] * (1 + area_factor * 0.2)))
            base_resources['boats'] = max(1, int(base_resources['boats'] * (1 + area_factor * 0.2)))
        
        # Priority level
        priority = self._calculate_priority(damage_class, severity)
        
        # Generate rationale
        rationale = self._generate_rationale(damage_class, severity, base_resources)
        
        return {
            'resources': base_resources,
            'priority': priority,
            'rationale': rationale
        }
    
    def _calculate_priority(self, damage_class: str, severity: str) -> str:
        """Calculate emergency priority level"""
        
        damage_scores = {'intact': 0, 'damaged': 1, 'destroyed': 2}
        severity_scores = {'low': 0, 'medium': 1, 'high': 2}
        
        total_score = damage_scores.get(damage_class, 0) + severity_scores.get(severity, 0)
        
        if total_score >= 3:
            return 'CRITICAL'
        elif total_score >= 2:
            return 'HIGH'
        elif total_score >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_rationale(self, damage_class: str, severity: str, resources: Dict) -> str:
        """Generate human-readable explanation"""
        
        rationale = f"Damage: {damage_class.upper()}, Severity: {severity.upper()}. "
        rationale += f"Allocating {resources['rescue_teams']} rescue teams, "
        rationale += f"{resources['boats']} boats, "
        rationale += f"{resources['medical_units']} medical units, "
        rationale += f"{resources['shelters']} shelters, "
        rationale += f"{resources['food_kits']} food kits, and "
        rationale += f"{resources['water_supply_liters']} liters of water."
        
        return rationale

# ============================================
# GENERATE SCENARIOS
# ============================================

def generate_test_scenarios():
    """Generate test scenarios for resource recommendation"""
    
    scenarios = [
        {
            'name': 'Scenario 1: Minor Flood',
            'damage_class': 'damaged',
            'severity': 'low',
            'population_affected': 100,
            'buildings_affected': 10,
            'area_affected_km2': 1.0
        },
        {
            'name': 'Scenario 2: Moderate Flood',
            'damage_class': 'damaged',
            'severity': 'medium',
            'population_affected': 1000,
            'buildings_affected': 50,
            'area_affected_km2': 5.0
        },
        {
            'name': 'Scenario 3: Severe Flood',
            'damage_class': 'destroyed',
            'severity': 'high',
            'population_affected': 5000,
            'buildings_affected': 200,
            'area_affected_km2': 25.0
        },
        {
            'name': 'Scenario 4: Urban Flood',
            'damage_class': 'damaged',
            'severity': 'high',
            'population_affected': 2000,
            'buildings_affected': 150,
            'area_affected_km2': 8.0
        },
        {
            'name': 'Scenario 5: Rural Flood',
            'damage_class': 'destroyed',
            'severity': 'medium',
            'population_affected': 500,
            'buildings_affected': 30,
            'area_affected_km2': 50.0
        }
    ]
    
    return scenarios

# ============================================
# DISPLAY RECOMMENDATIONS
# ============================================

def display_recommendations(engine, scenarios):
    """Display resource recommendations for all scenarios"""
    
    print("\n" + "="*80)
    print("📋 RESOURCE RECOMMENDATIONS")
    print("="*80)
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n🔴 {scenario['name']}")
        print("-"*60)
        print(f"   Damage: {scenario['damage_class'].upper()}")
        print(f"   Severity: {scenario['severity'].upper()}")
        print(f"   Population Affected: {scenario['population_affected']}")
        print(f"   Buildings Affected: {scenario['buildings_affected']}")
        print(f"   Area Affected: {scenario['area_affected_km2']} km²")
        
        # Get recommendation
        result = engine.recommend_resources(
            damage_class=scenario['damage_class'],
            severity=scenario['severity'],
            population_affected=scenario['population_affected'],
            buildings_affected=scenario['buildings_affected'],
            area_affected_km2=scenario['area_affected_km2']
        )
        
        print(f"\n   📊 Priority: {result['priority']}")
        print(f"   📦 Resources:")
        for resource, amount in result['resources'].items():
            if amount > 0:
                print(f"      • {resource.replace('_', ' ').title()}: {amount}")
        print(f"\n   💡 Rationale: {result['rationale']}")
        
        # Store result
        all_results.append({
            'scenario': scenario['name'],
            'damage_class': scenario['damage_class'],
            'severity': scenario['severity'],
            'priority': result['priority'],
            'resources': result['resources'],
            'rationale': result['rationale']
        })
    
    return all_results

# ============================================
# SAVE RESULTS
# ============================================

def save_results(results):
    """Save recommendations to JSON"""
    
    output = {
        'engine': 'Resource Recommendation Engine (Stage 3)',
        'total_scenarios': len(results),
        'recommendations': results
    }
    
    output_file = os.path.join(CONFIG['output_dir'], 'resource_recommendations.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    return output_file

# ============================================
# COMPARISON TABLE
# ============================================

def display_comparison_table(results):
    """Display comparison table of all scenarios"""
    
    print("\n" + "="*80)
    print("📊 COMPARISON TABLE")
    print("="*80)
    
    print("\n{:<20} {:<12} {:<10} {:<8} {:<8} {:<10}".format(
        "Scenario", "Damage", "Severity", "Priority", "Teams", "Boats"
    ))
    print("-"*80)
    
    for result in results:
        resources = result['resources']
        print("{:<20} {:<12} {:<10} {:<8} {:<8} {:<10}".format(
            result['scenario'][:18],
            result['damage_class'].upper(),
            result['severity'].upper(),
            result['priority'],
            resources['rescue_teams'],
            resources['boats']
        ))

# ============================================
# MAIN
# ============================================

def main():
    # Initialize engine
    engine = ResourceRecommendationEngine()
    
    # Generate scenarios
    scenarios = generate_test_scenarios()
    
    # Display recommendations
    results = display_recommendations(engine, scenarios)
    
    # Display comparison table
    display_comparison_table(results)
    
    # Save results
    output_file = save_results(results)
    
    print("\n" + "="*80)
    print("✅ STAGE 3 COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {output_file}")
    print("\n📋 Summary:")
    print(f"   Total Scenarios: {len(results)}")
    print(f"   Priority Distribution:")
    
    priority_counts = {}
    for r in results:
        priority_counts[r['priority']] = priority_counts.get(r['priority'], 0) + 1
    
    for priority, count in priority_counts.items():
        print(f"      {priority}: {count}")
    
    print("\n" + "="*80)
    print("🚀 ALL STAGES COMPLETE!")
    print("="*80)
    print("\n📁 Project Outputs:")
    print("   - outputs/best_model_simple.pt (Stage 1)")
    print("   - outputs/damage_assessment_results_simple.json")
    print("   - outputs/best_severity_model.pt (Stage 2)")
    print("   - outputs/severity_results.json")
    print("   - outputs/resource_recommendations.json (Stage 3)")

if __name__ == "__main__":
    main()