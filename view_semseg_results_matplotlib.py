#!/usr/bin/env python3
"""
3D Visualization using Matplotlib for Semantic Segmentation Results
Works in headless/remote environments
Saves interactive 3D visualization as PNG images
"""

import numpy as np
import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))

from indoor3d_util import g_label2color

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']


def parse_args():
    parser = argparse.ArgumentParser('3D Matplotlib Visualizer for Semantic Segmentation')
    parser.add_argument('--visual_dir', type=str, required=True,
                       help='Directory with visualization .obj files')
    parser.add_argument('--scene', type=str, default=None,
                       help='Specific scene name (without extension)')
    parser.add_argument('--sample_rate', type=float, default=0.05,
                       help='Sample rate for visualization (0.05 = 5%% of points) [default: 0.05]')
    parser.add_argument('--dpi', type=int, default=100,
                       help='DPI for output image [default: 100]')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for images. Default: visual_dir')
    return parser.parse_args()


def load_obj_file(obj_file, sample_rate=1.0):
    """Load .obj file"""
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
        return None, None
    
    if len(vertices) == 0:
        print(f"No vertices found in {obj_file}")
        return None, None
    
    vertices = np.array(vertices, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    
    # Sample if needed
    if sample_rate < 1.0:
        n_points = len(vertices)
        n_sample = int(n_points * sample_rate)
        print(f"  Sampling {n_sample} from {n_points} points ({sample_rate*100:.1f}%)")
        indices = np.random.choice(n_points, n_sample, replace=False)
        vertices = vertices[indices]
        colors = colors[indices]
    
    return vertices, colors


def colors_to_labels(colors):
    """Convert RGB colors to semantic labels"""
    labels = np.zeros(len(colors), dtype=int)
    
    color_to_label = {}
    for label_id, color in g_label2color.items():
        color_tuple = tuple(color)
        color_to_label[color_tuple] = label_id
    
    for i, color in enumerate(colors):
        color_int = tuple(color.astype(int))
        if color_int in color_to_label:
            labels[i] = color_to_label[color_int]
        else:
            # Find closest
            min_dist = float('inf')
            best_label = 0
            for label_id, ref_color in g_label2color.items():
                dist = np.sum((color_int - np.array(ref_color)) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_label = label_id
            labels[i] = best_label
    
    return labels


def plot_3d_scatter(xyz, colors, title, output_file=None, dpi=100):
    """Create 3D scatter plot"""
    fig = plt.figure(figsize=(12, 9), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors to 0-1 range for matplotlib
    colors_normalized = colors / 255.0
    
    # Plot points
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors_normalized, s=1, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = np.array([xyz[:, 0].max() - xyz[:, 0].min(),
                          xyz[:, 1].max() - xyz[:, 1].min(),
                          xyz[:, 2].max() - xyz[:, 2].min()]).max() / 2.0
    mid_x = (xyz[:, 0].max() + xyz[:, 0].min()) * 0.5
    mid_y = (xyz[:, 1].max() + xyz[:, 1].min()) * 0.5
    mid_z = (xyz[:, 2].max() + xyz[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig


def compute_accuracy(gt_colors, pred_colors):
    """Compute accuracy metrics"""
    gt_labels = colors_to_labels(gt_colors)
    pred_labels = colors_to_labels(pred_colors)
    
    accuracy = np.mean(gt_labels == pred_labels)
    
    print("\n" + "="*60)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print("\nPer-class accuracy:")
    print("-"*60)
    
    for i, cls_name in enumerate(classes):
        mask = gt_labels == i
        if mask.sum() > 0:
            cls_acc = (pred_labels[mask] == gt_labels[mask]).mean()
            correct = (pred_labels[mask] == gt_labels[mask]).sum()
            total = mask.sum()
            print(f"  {cls_name:12s}: {cls_acc*100:5.2f}% ({correct:6d}/{total:6d})")
    print("="*60)
    
    return accuracy


def main():
    args = parse_args()
    
    visual_dir = args.visual_dir
    
    if not os.path.exists(visual_dir):
        print(f"Error: Visual directory not found: {visual_dir}")
        return
    
    if args.scene is None:
        print("Error: --scene argument required")
        print(f"Example: python view_semseg_results_matplotlib.py --visual_dir {visual_dir} --scene Area_5_conferenceRoom_1")
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
    
    # Set output directory
    if args.output_dir is None:
        output_dir = visual_dir
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading scene: {args.scene}")
    print(f"Sample rate: {args.sample_rate*100:.1f}%")
    
    # Load files
    print("\nLoading ground truth...")
    xyz_gt, colors_gt = load_obj_file(gt_file, args.sample_rate)
    if xyz_gt is None:
        return
    
    print("Loading predictions...")
    xyz_pred, colors_pred = load_obj_file(pred_file, args.sample_rate)
    if xyz_pred is None:
        return
    
    # Use GT coordinates
    xyz = xyz_gt
    
    # Compute accuracy
    accuracy = compute_accuracy(colors_gt, colors_pred)
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    
    # Ground truth
    gt_output = os.path.join(output_dir, f"{args.scene}_gt_visualization.png")
    print("\nGround Truth:")
    fig_gt = plot_3d_scatter(xyz, colors_gt, f"{args.scene} - Ground Truth Labels", gt_output, args.dpi)
    plt.close(fig_gt)
    
    # Predictions
    pred_output = os.path.join(output_dir, f"{args.scene}_pred_visualization.png")
    print("Predictions:")
    fig_pred = plot_3d_scatter(xyz, colors_pred, f"{args.scene} - Predicted Labels", pred_output, args.dpi)
    plt.close(fig_pred)
    
    # Comparison (side by side)
    print("\nComparison (side-by-side):")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': '3d'}, dpi=args.dpi)
    
    # Ground truth
    colors_gt_norm = colors_gt / 255.0
    axes[0].scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors_gt_norm, s=1, alpha=0.8)
    axes[0].set_title(f"{args.scene} - Ground Truth", fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_zlabel('Z')
    
    # Predictions
    colors_pred_norm = colors_pred / 255.0
    axes[1].scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors_pred_norm, s=1, alpha=0.8)
    axes[1].set_title(f"{args.scene} - Predictions (Accuracy: {accuracy*100:.2f}%)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_zlabel('Z')
    
    # Set equal aspect ratios
    max_range = np.array([xyz[:, 0].max() - xyz[:, 0].min(),
                          xyz[:, 1].max() - xyz[:, 1].min(),
                          xyz[:, 2].max() - xyz[:, 2].min()]).max() / 2.0
    mid_x = (xyz[:, 0].max() + xyz[:, 0].min()) * 0.5
    mid_y = (xyz[:, 1].max() + xyz[:, 1].min()) * 0.5
    mid_z = (xyz[:, 2].max() + xyz[:, 2].min()) * 0.5
    
    for ax in axes:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    comparison_output = os.path.join(output_dir, f"{args.scene}_comparison.png")
    plt.savefig(comparison_output, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {comparison_output}")
    plt.close(fig)
    
    print("\n" + "="*60)
    print("✓ Visualizations complete!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
