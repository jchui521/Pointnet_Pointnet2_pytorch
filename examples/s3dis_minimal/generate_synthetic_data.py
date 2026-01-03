#!/usr/bin/env python3
"""
Generate synthetic S3DIS minimal room data for testing the workflow.
Creates a sample room file with random point cloud data.
"""

import numpy as np
from pathlib import Path
import argparse


def generate_synthetic_room(num_points=50000, num_classes=13, seed=42):
    """
    Generate synthetic point cloud data in S3DIS format.
    
    Returns:
        ndarray of shape (num_points, 7) with [X, Y, Z, R, G, B, Label]
    """
    np.random.seed(seed)
    
    # XYZ: random points in a 10x10x5 box (typical room dimensions)
    xyz = np.random.uniform(0, 10, (num_points, 3))
    xyz[:, 2] = np.random.uniform(0, 5, num_points)  # height 0-5
    
    # RGB: random colors (0-255)
    rgb = np.random.randint(0, 256, (num_points, 3), dtype=np.uint8).astype(np.float32)
    
    # Labels: randomly assign to classes (0-12, 13 total S3DIS classes)
    # Bias towards floor (class 1) and wall (class 2) to be more realistic
    labels = np.random.randint(0, num_classes, num_points)
    # Make floor more common (bottom 20% of points)
    floor_mask = xyz[:, 2] < 1.0
    labels[floor_mask] = 1  # floor
    
    # Make walls more common (edges)
    wall_mask = (xyz[:, 0] < 0.5) | (xyz[:, 0] > 9.5) | (xyz[:, 1] < 0.5) | (xyz[:, 1] > 9.5)
    labels[wall_mask & ~floor_mask] = 2  # wall
    
    # Combine: [X, Y, Z, R, G, B, Label]
    data = np.concatenate([xyz, rgb, labels.reshape(-1, 1)], axis=1)
    
    return data


def main():
    parser = argparse.ArgumentParser("Generate synthetic S3DIS minimal data")
    parser.add_argument("--output_dir", type=str, default="data/stanford_indoor3d_minimal",
                       help="Output directory [default: data/stanford_indoor3d_minimal]")
    parser.add_argument("--room_name", type=str, default="Area_5_conferenceRoom_1",
                       help="Room file name (without .npy) [default: Area_5_conferenceRoom_1]")
    parser.add_argument("--num_points", type=int, default=50000,
                       help="Number of points in the room [default: 50000]")
    parser.add_argument("--num_classes", type=int, default=13,
                       help="Number of semantic classes [default: 13 (S3DIS)]")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility [default: 42]")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("  GENERATING SYNTHETIC S3DIS MINIMAL DATA")
    print(f"{'='*70}\n")
    
    print(f"Output directory: {output_dir}")
    print(f"Room name:        {args.room_name}")
    print(f"Num points:       {args.num_points:,}")
    print(f"Num classes:      {args.num_classes}")
    print(f"Seed:             {args.seed}")
    
    # Generate synthetic room data
    print(f"\nGenerating synthetic point cloud...")
    room_data = generate_synthetic_room(
        num_points=args.num_points,
        num_classes=args.num_classes,
        seed=args.seed
    )
    
    # Save as .npy file
    output_file = output_dir / f"{args.room_name}.npy"
    np.save(str(output_file), room_data)
    
    print(f"✓ Saved to: {output_file}")
    print(f"  Shape: {room_data.shape}")
    print(f"  Format: [X, Y, Z, R, G, B, Label]")
    print(f"  XYZ range: {room_data[:, :3].min(axis=0)} to {room_data[:, :3].max(axis=0)}")
    print(f"  Label range: {int(room_data[:, 6].min())} to {int(room_data[:, 6].max())}")
    
    # Print label distribution
    unique_labels, counts = np.unique(room_data[:, 6].astype(int), return_counts=True)
    print(f"\n  Label distribution:")
    for label, count in zip(unique_labels, counts):
        pct = 100.0 * count / len(room_data)
        print(f"    Class {label:2d}: {count:6d} points ({pct:5.1f}%)")
    
    print(f"\n{'='*70}")
    print("  ✓ DATA GENERATION COMPLETE")
    print(f"{'='*70}\n")
    print(f"You can now run the workflow:")
    print(f"  python examples/s3dis_minimal/run_workflow.py --epoch 5 --batch_size 2\n")


if __name__ == '__main__':
    main()
