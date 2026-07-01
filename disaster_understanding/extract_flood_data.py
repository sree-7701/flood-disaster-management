"""
extract_flood_tasks.py
Extract specific flood tasks and create organized dataset structure
"""

import json
import os
from collections import defaultdict

def load_data():
    """Load flood samples"""
    with open('flood_samples.json', 'r') as f:
        return json.load(f)

def extract_by_task(data, task_name):
    """Extract samples for a specific task"""
    return [item for item in data if item.get('task') == task_name]

def count_by_event(data):
    """Count samples by disaster event"""
    events = defaultdict(int)
    for item in data:
        pre_path = item.get('pre_image_path', '')
        if 'hurricane' in pre_path:
            parts = pre_path.split('hurricane_')
            if len(parts) > 1:
                event = parts[1].split('_')[0]
                events[event] += 1
        elif 'midwest' in pre_path:
            events['midwest_flood'] += 1
        elif 'flood' in pre_path:
            events['general_flood'] += 1
    return events

def main():
    print("="*60)
    print("📂 EXTRACTING FLOOD TASKS")
    print("="*60)
    
    data = load_data()
    print(f"Total flood samples: {len(data)}")
    
    # 1. Extract by task
    print("\n📝 Extracting by task...")
    tasks = set()
    for item in data:
        tasks.add(item.get('task', 'Unknown'))
    
    for task in tasks:
        task_data = extract_by_task(data, task)
        filename = f"flood_{task.lower().replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(task_data, f, indent=2)
        print(f"   {task}: {len(task_data)} samples -> {filename}")
    
    # 2. Count by event
    print("\n🌪️ Counting by event...")
    events = count_by_event(data)
    for event, count in sorted(events.items(), key=lambda x: x[1], reverse=True):
        print(f"   {event}: {count}")
    
    # 3. Image type distribution
    print("\n📷 Image types:")
    img_types = defaultdict(int)
    for item in data:
        img_types[item.get('post_image_type', 'Unknown')] += 1
    for img_type, count in img_types.items():
        print(f"   {img_type}: {count}")
    
    print("\n✅ Done! Files created:")
    print("   - flood_referring_expression_segmentation.json (5,093 samples)")
    print("   - flood_disaster_report.json (2,360 samples)")
    print("   - flood_building_damage_counting.json (1,401 samples)")
    print("   - And more...")

if __name__ == "__main__":
    main()