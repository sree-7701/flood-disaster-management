"""
flood_analysis.py
Analyze and visualize flood-specific data from DisasterM3
"""

import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Try to import cv2
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not installed. Run: pip install opencv-python")

# ============================================
# LOAD FLOOD DATA
# ============================================

def load_flood_data():
    """Load the flood samples we saved earlier"""
    with open('flood_samples.json', 'r') as f:
        flood_data = json.load(f)
    
    print(f"✅ Loaded {len(flood_data)} flood-related samples")
    return flood_data

# ============================================
# ANALYZE FLOOD DATA
# ============================================

def analyze_flood_data(flood_data):
    """Comprehensive analysis of flood dataset"""
    
    print("\n" + "="*60)
    print("📊 FLOOD DATASET ANALYSIS")
    print("="*60)
    
    # 1. Task distribution
    tasks = Counter([item.get('task', 'Unknown') for item in flood_data])
    print("\n📝 Task Distribution:")
    for task, count in tasks.most_common():
        percentage = (count / len(flood_data)) * 100
        print(f"   {task}: {count} ({percentage:.1f}%)")
    
    # 2. Image types
    img_types = Counter([item.get('post_image_type', 'Unknown') for item in flood_data])
    print("\n📷 Image Types:")
    for img_type, count in img_types.most_common():
        percentage = (count / len(flood_data)) * 100
        print(f"   {img_type}: {count} ({percentage:.1f}%)")
    
    # 3. Extract event names from paths
    events = Counter()
    for item in flood_data:
        pre_path = item.get('pre_image_path', '')
        # Extract event name from path
        if 'hurricane' in pre_path.lower():
            parts = pre_path.lower().split('hurricane_')
            if len(parts) > 1:
                event = parts[1].split('_')[0]
                if event and event not in ['pre', 'post']:
                    events[event] += 1
        elif 'midwest' in pre_path.lower():
            events['midwest_flood'] += 1
        elif 'flood' in pre_path.lower() and 'hurricane' not in pre_path.lower():
            events['general_flood'] += 1
    
    print("\n🌪️ Disaster Events Found:")
    for event, count in events.most_common(10):
        print(f"   {event}: {count}")
    
    # 4. Ground truth analysis - FIXED: handle different types
    all_objects = Counter()
    for item in flood_data:
        gt = item.get('ground_truth', '')
        if gt and isinstance(gt, str):  # Only process if it's a string
            # Split by comma and clean
            objects = [obj.strip() for obj in gt.split(',') if obj.strip()]
            for obj in objects:
                all_objects[obj] += 1
        elif gt and isinstance(gt, list):  # If it's a list
            for obj in gt:
                if isinstance(obj, str):
                    all_objects[obj.strip()] += 1
    
    print("\n🏗️ Most Common Objects in Ground Truth:")
    for obj, count in all_objects.most_common(10):
        print(f"   {obj}: {count}")
    
    # 5. Check for sample images with masks
    mask_count = 0
    for item in flood_data:
        if 'mask' in str(item).lower():
            mask_count += 1
    
    print(f"\n🎯 Samples with masks: {mask_count}")
    
    return tasks, img_types, events, all_objects
# ============================================
# VISUALIZE TASK DISTRIBUTION
# ============================================

def plot_task_distribution(tasks):
    """Plot the distribution of tasks"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    task_names = list(tasks.keys())
    task_counts = list(tasks.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(task_names)))
    
    bars = ax.barh(task_names, task_counts, color=colors)
    ax.set_xlabel('Number of Samples')
    ax.set_title('Flood Dataset: Task Distribution', fontsize=14)
    
    # Add count labels
    for bar, count in zip(bars, task_counts):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                str(count), va='center')
    
    plt.tight_layout()
    plt.savefig('flood_task_distribution.png', dpi=150)
    plt.show()
    print("💾 Saved: flood_task_distribution.png")

# ============================================
# VISUALIZE SAMPLE IMAGES
# ============================================

def visualize_flood_samples(flood_data, num_samples=4):
    """Display sample flood image pairs"""
    
    if not CV2_AVAILABLE:
        print("❌ OpenCV not available. Install: pip install opencv-python")
        return
    
    # Filter samples with both pre and post images
    valid_samples = []
    for item in flood_data:
        pre = item.get('pre_image_path', '')
        post = item.get('post_image_path', '')
        if pre and post:
            # Check if files exist
            pre_path = f"DisasterM3/{pre}"
            if not os.path.exists(pre_path):
                pre_path = pre
            post_path = f"DisasterM3/{post}"
            if not os.path.exists(post_path):
                post_path = post
            
            if os.path.exists(pre_path) and os.path.exists(post_path):
                valid_samples.append({
                    'item': item,
                    'pre_path': pre_path,
                    'post_path': post_path
                })
    
    print(f"Found {len(valid_samples)} samples with valid images")
    
    if len(valid_samples) == 0:
        print("❌ No valid image pairs found!")
        return
    
    # Select samples to display
    samples_to_show = valid_samples[:min(num_samples, len(valid_samples))]
    
    fig, axes = plt.subplots(len(samples_to_show), 2, figsize=(12, 4*len(samples_to_show)))
    if len(samples_to_show) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample in enumerate(samples_to_show):
        item = sample['item']
        
        # Read images
        pre_img = cv2.imread(sample['pre_path'])
        post_img = cv2.imread(sample['post_path'])
        
        if pre_img is None or post_img is None:
            print(f"⚠️ Could not read images for sample {idx}")
            continue
        
        # Convert BGR to RGB
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
        
        # Display
        axes[idx, 0].imshow(pre_img)
        axes[idx, 0].set_title(f"Pre-Disaster\n{item.get('task', '')[:30]}", fontsize=10)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(post_img)
        axes[idx, 1].set_title(f"Post-Disaster\nType: {item.get('post_image_type', '')}", fontsize=10)
        axes[idx, 1].axis('off')
    
    plt.suptitle('Flood Disaster Sample Images', fontsize=16)
    plt.tight_layout()
    plt.savefig('flood_samples.png', dpi=150)
    plt.show()
    print("💾 Saved: flood_samples.png")

# ============================================
# ANALYZE DIFFERENT TASKS
# ============================================

def analyze_tasks_by_event(flood_data):
    """Analyze task distribution by event type"""
    
    # Extract event from path
    events = {}
    for item in flood_data:
        pre_path = item.get('pre_image_path', '')
        task = item.get('task', 'Unknown')
        
        # Extract event
        if 'hurricane_florence' in pre_path:
            event = 'Hurricane Florence'
        elif 'hurricane_harvey' in pre_path:
            event = 'Hurricane Harvey'
        elif 'hurricane_matthew' in pre_path:
            event = 'Hurricane Matthew'
        elif 'midwest' in pre_path:
            event = 'Midwest Flood'
        else:
            event = 'Other'
        
        key = f"{event} - {task}"
        events[key] = events.get(key, 0) + 1
    
    print("\n" + "="*60)
    print("📊 TASK DISTRIBUTION BY EVENT:")
    print("="*60)
    
    for key, count in sorted(events.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"   {key}: {count}")

# ============================================
# SAVE FLOOD DATA BY TASK
# ============================================

def save_flood_by_task(flood_data):
    """Save flood data organized by task"""
    
    tasks = {}
    for item in flood_data:
        task = item.get('task', 'Unknown')
        if task not in tasks:
            tasks[task] = []
        tasks[task].append(item)
    
    print("\n" + "="*60)
    print("💾 Saving flood data by task:")
    print("="*60)
    
    for task, samples in tasks.items():
        filename = f"flood_{task.replace(' ', '_').lower()}.json"
        with open(filename, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"   Saved {len(samples)} samples to: {filename}")
    
    return tasks

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    print("="*60)
    print("🌊 FLOOD DATA ANALYZER")
    print("="*60)
    
    # Load data
    flood_data = load_flood_data()
    
    # Analyze
    tasks, img_types, events, objects = analyze_flood_data(flood_data)
    
    # Plot task distribution
    plot_task_distribution(tasks)
    
    # Analyze tasks by event
    analyze_tasks_by_event(flood_data)
    
    # Visualize samples
    visualize_flood_samples(flood_data, num_samples=3)
    
    # Save organized data
    save_flood_by_task(flood_data)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\n📊 Dataset Summary:")
    print(f"   Total flood samples: {len(flood_data)}")
    print(f"   Number of tasks: {len(tasks)}")
    print(f"   Image types: {', '.join(img_types.keys())}")
    print(f"   Main events: {', '.join(list(events.keys())[:4])}")

if __name__ == "__main__":
    main()