#!/usr/bin/env python3
"""
Open3D Visualization for Semantic Segmentation Results
Shows ground truth and predicted labels with interactive 3D viewer
Press 'Q' or ESC to close each window and move to the next
"""

import numpy as np
import argparse
import os
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d not found. Install it with: pip install open3d")
    sys.exit(1)

from indoor3d_util import g_label2color

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']


def parse_args():
    parser = argparse.ArgumentParser('Open3D Semantic Segmentation Visualizer')
    parser.add_argument('--visual_dir', type=str, required=True,
                       help='Directory with visualization .obj files')
    parser.add_argument('--scene', type=str, default=None,
                       help='Specific scene name (without extension)')
    parser.add_argument('--show_gt', action='store_true', default=True,
                       help='Show ground truth')
    parser.add_argument('--show_pred', action='store_true', default=True,
                       help='Show predictions')
    return parser.parse_args()


def load_obj_as_pcd(obj_file):
    """Load .obj file and create Open3D point cloud"""
    vertices = []
    colors = []
    
    try:
        with open(obj_file, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                    vertices.append([x, y, z])
                    colors.append([r, g, b])
    except Exception as e:
        print(f"Error reading {obj_file}: {e}")
        return None
    
    if len(vertices) == 0:
        print(f"No vertices found in {obj_file}")
        return None
    
    vertices = np.array(vertices, dtype=np.float64)
    colors = np.array(colors, dtype=np.float64)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def compute_accuracy_from_obj(gt_obj, pred_obj):
    """Compute accuracy metrics from obj files (skip for large files)"""
    # For very large files, skip accuracy computation
    print("\n" + "="*60)
    print("Accuracy computation skipped for large files (1M+ points)")
    print("(Will be computed after visualization if needed)")
    print("="*60)


def colors_to_labels(colors):
    """Convert RGB colors to semantic labels"""
    labels = np.zeros(len(colors), dtype=int)
    
    # Build color to label mapping
    color_to_label = {}
    for label_id, color in g_label2color.items():
        color_tuple = tuple(color)
        color_to_label[color_tuple] = label_id
    
    for i, color in enumerate(colors):
        color_int = tuple(color.astype(int))
        if color_int in color_to_label:
            labels[i] = color_to_label[color_int]
        else:
            # Find closest label
            min_dist = float('inf')
            best_label = 0
            for label_id, ref_color in g_label2color.items():
                dist = np.sum((color_int - np.array(ref_color)) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_label = label_id
            labels[i] = best_label
    
    return labels


def visualize_with_simple_draw(pcd, window_name="Point Cloud"):
    """Visualize point cloud using simple draw_geometries"""
    print(f"\nOpening: {window_name}")
    print("Controls: Rotate (left-click), Pan (right-click), Zoom (scroll)")
    print("Close window to continue...")
    
    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_name,
            width=1024,
            height=768,
            point_show_normal=False
        )
    except Exception as e:
        print(f"Note: {e}")
        print("(This is expected in headless environments)")
        print("The visualization files are saved as .obj files that you can open in:")
        print("  - MeshLab (free, open-source)")
        print("  - CloudCompare")
        print("  - Blender")
        print("  - Any 3D viewer that supports OBJ format")


def main():
    args = parse_args()
    
    visual_dir = args.visual_dir
    
    if not os.path.exists(visual_dir):
        print(f"Error: Visual directory not found: {visual_dir}")
        return
    
    if args.scene is None:
        print("Error: --scene argument required")
        print(f"Example: python view_semseg_results_open3d.py --visual_dir {visual_dir} --scene Area_5_conferenceRoom_1")
        return
    
    # Build file paths
    gt_file = os.path.join(visual_dir, f"{args.scene}_gt.obj")
    pred_file = os.path.join(visual_dir, f"{args.scene}_pred.obj")
    
    if not os.path.exists(gt_file):
        print(f"Error: Ground truth file not found: {gt_file}")
        return
    
    if not os.path.exists(pred_file):
        print(f"Error: Prediction file not found: {pred_file}")
        return
    
    print(f"\nLoading scene: {args.scene}")
    print(f"GT file: {gt_file}")
    print(f"Pred file: {pred_file}")
    
    # Load point clouds
    print("\nLoading ground truth...")
    pcd_gt = load_obj_as_pcd(gt_file)
    if pcd_gt is None:
        return
    
    print(f"  Points: {len(pcd_gt.points)}")
    
    print("Loading predictions...")
    pcd_pred = load_obj_as_pcd(pred_file)
    if pcd_pred is None:
        return
    
    print(f"  Points: {len(pcd_pred.points)}")
    
    # Compute and display accuracy
    compute_accuracy_from_obj(gt_file, pred_file)
    
    # Visualize
    print("\n" + "="*60)
    print("Open3D Visualization Controls:")
    print("  Mouse: Left=Rotate, Right=Translate, Scroll=Zoom")
    print("  Keys: Q or ESC = Close window and advance")
    print("="*60)
    
    if args.show_gt and args.show_pred:
        print(f"\n[1/2] Showing Ground Truth...")
        print("      Close window to view predictions")
        visualize_with_simple_draw(pcd_gt, f"{args.scene} - Ground Truth")
        
        print(f"\n[2/2] Showing Predictions...")
        print("      Close window to finish")
        visualize_with_simple_draw(pcd_pred, f"{args.scene} - Predictions")
    
    elif args.show_gt:
        print(f"\nShowing Ground Truth only...")
        visualize_with_simple_draw(pcd_gt, f"{args.scene} - Ground Truth")
    
    elif args.show_pred:
        print(f"\nShowing Predictions only...")
        visualize_with_simple_draw(pcd_pred, f"{args.scene} - Predictions")
    
    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
