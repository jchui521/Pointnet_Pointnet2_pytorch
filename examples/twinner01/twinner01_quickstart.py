"""
Twinner01 Quick Start Script
Creates sample data and tests the complete pipeline
"""

import os
import sys
import numpy as np
from pathlib import Path

def create_test_environment():
    """Set up directories and sample data"""
    print("=" * 60)
    print("TWINNER01 QUICK START")
    print("=" * 60)
    
    # Create data directory
    data_dir = Path('data/twinner01_custom')
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Created directory: {data_dir}")
    
    # Create sample scenes
    print("\nCreating sample scenes...")
    num_scenes = 5
    num_points = 8000
    num_classes = 6
    
    for i in range(num_scenes):
        # Generate synthetic point cloud
        # Simulate a simple indoor scene
        
        # Floor (class 1)
        floor = np.random.rand(num_points // 3, 3)
        floor[:, 2] = np.random.rand(num_points // 3) * 0.1  # Flat at bottom
        floor_labels = np.ones(num_points // 3) * 1
        floor_colors = np.array([200, 200, 200]) + np.random.randint(-20, 20, (num_points // 3, 3))
        
        # Objects (classes 2-5)
        objects = np.random.rand(num_points // 3, 3)
        objects[:, 2] = objects[:, 2] * 2 + 0.1  # Elevated
        object_labels = np.random.randint(2, num_classes, num_points // 3)
        object_colors = np.random.randint(0, 256, (num_points // 3, 3))
        
        # Ceiling (class 0 - background)
        ceiling = np.random.rand(num_points // 3, 3)
        ceiling[:, 2] = np.random.rand(num_points // 3) * 0.1 + 2.5  # Flat at top
        ceiling_labels = np.zeros(num_points // 3)
        ceiling_colors = np.array([150, 150, 150]) + np.random.randint(-20, 20, (num_points // 3, 3))
        
        # Combine all points
        xyz = np.vstack([floor, objects, ceiling])
        rgb = np.vstack([floor_colors, object_colors, ceiling_colors])
        labels = np.hstack([floor_labels, object_labels, ceiling_labels])
        
        # Scale coordinates to reasonable size
        xyz = xyz * 10
        
        # Combine into final format (N, 7)
        scene_data = np.hstack([xyz, rgb, labels.reshape(-1, 1)])
        
        # Save
        output_file = data_dir / f'scene_{i+1:02d}.npy'
        np.save(output_file, scene_data.astype(np.float32))
        print(f"  ✓ Created: {output_file} ({scene_data.shape[0]} points)")
    
    print(f"\n✓ Created {num_scenes} sample scenes")
    
    return num_scenes, num_classes

def test_dataloader(num_classes):
    """Test the dataloader"""
    print("\n" + "=" * 60)
    print("TESTING DATALOADER")
    print("=" * 60)
    
    try:
        # Import after data is created
        from twinner01_dataloader import Twinner01Dataset
        
        # Load dataset
        dataset = Twinner01Dataset(
            split='train',
            data_root='data/twinner01_custom',
            num_point=4096,
            test_split=0.4,  # Use 40% for test (2 out of 5 scenes)
            num_classes=num_classes
        )
        
        print(f"\n✓ Dataset loaded successfully")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Number of classes: {num_classes}")
        
        # Test loading a sample
        points, labels = dataset[0]
        print(f"\n✓ Sample loaded successfully")
        print(f"  - Points shape: {points.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Unique labels in sample: {sorted(np.unique(labels).astype(int).tolist())}")
        print(f"  - Point coordinate range: [{points[:, :3].min():.2f}, {points[:, :3].max():.2f}]")
        print(f"  - Color range: [{points[:, 3:6].min():.2f}, {points[:, 3:6].max():.2f}]")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error testing dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_next_steps():
    """Display next steps"""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    print("\n1. CUSTOMIZE YOUR CLASSES:")
    print("   Edit: twinner01_classes_config.py")
    print("   Replace sample classes with your actual object types")
    
    print("\n2. PREPARE YOUR REAL DATA:")
    print("   python twinner01_prepare_data.py --mode batch \\")
    print("     --input 'path/to/your/data' \\")
    print("     --output 'data/twinner01_custom'")
    
    print("\n3. CREATE TRAINING SCRIPT:")
    print("   Next: Adapt train_semseg.py for Twinner01")
    
    print("\n4. TRAIN YOUR MODEL:")
    print("   python twinner01_train.py --epoch 100 --batch_size 8")
    
    print("\n" + "=" * 60)
    print("QUICK START COMPLETE!")
    print("=" * 60)

def main():
    # Check if sample data already exists
    data_dir = Path('data/twinner01_custom')
    if data_dir.exists() and len(list(data_dir.glob('*.npy'))) > 0:
        print("Sample data already exists!")
        response = input("Recreate sample data? (y/n): ")
        if response.lower() != 'y':
            print("Using existing data...")
            num_classes = 6  # Default
            test_dataloader(num_classes)
            show_next_steps()
            return
    
    # Create sample data
    num_scenes, num_classes = create_test_environment()
    
    # Test dataloader
    success = test_dataloader(num_classes)
    
    if success:
        show_next_steps()
    else:
        print("\n✗ Setup encountered errors. Please check the output above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
