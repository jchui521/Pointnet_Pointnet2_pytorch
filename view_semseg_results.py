"""
Visualizer for Semantic Segmentation Results
View ground truth vs predictions from test_semseg.py output
Uses Open3D for visualization
"""

import numpy as np
import argparse
import os
import sys
from pathlib import Path
import open3d as o3d

# Add data_utils to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))

from indoor3d_util import g_label2color

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']

def parse_args():
    parser = argparse.ArgumentParser('Semantic Segmentation Visualizer')
    parser.add_argument('--visual_dir', type=str, required=True,
                       help='Directory with visualization files (e.g., log/sem_seg/pointnet2_sem_seg/visual)')
    parser.add_argument('--scene', type=str, default=None,
                       help='Specific scene to visualize (e.g., Area_5_office_1). If None, shows list.')
    parser.add_argument('--show_gt', action='store_true', default=True,
                       help='Show ground truth')
    parser.add_argument('--show_pred', action='store_true', default=True,
                       help='Show predictions')
    return parser.parse_args()


def load_obj_file(obj_file):
    """Load point cloud from .obj file"""
    vertices = []
    colors = []
    
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # v x y z r g b
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                vertices.append([x, y, z])
                colors.append([r, g, b])
    
    return np.array(vertices), np.array(colors)


def load_scene_data(visual_dir, scene_name):
    """Load scene point cloud from ground truth and prediction .obj files"""
    gt_file = os.path.join(visual_dir, scene_name + '_gt.obj')
    pred_file = os.path.join(visual_dir, scene_name + '_pred.obj')
    
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    
    # Load ground truth
    xyz_gt, colors_gt = load_obj_file(gt_file)
    
    # Load predictions (should have same points, different colors)
    xyz_pred, colors_pred = load_obj_file(pred_file)
    
    if len(xyz_gt) != len(xyz_pred):
        print(f"Warning: GT and Pred have different number of points ({len(xyz_gt)} vs {len(xyz_pred)})")
    
    # Use GT coordinates (they should be the same)
    xyz = xyz_gt
    
    # Colors already represent the semantic labels
    return xyz, colors_gt, colors_pred


def labels_to_colors(labels):
    """Convert semantic labels to RGB colors"""
    colors = np.zeros((len(labels), 3))
    for i, label in enumerate(labels):
        if label < len(g_label2color):
            colors[i] = g_label2color[label]
        else:
            colors[i] = [128, 128, 128]  # Gray for unknown
    return colors


def colors_to_labels(colors):
    """Convert RGB colors back to semantic labels"""
    labels = np.zeros(len(colors), dtype=int)
    
    # Build reverse mapping
    color_to_label = {}
    for label_id, color in g_label2color.items():
        color_tuple = tuple(color)
        color_to_label[color_tuple] = label_id
    
    for i, color in enumerate(colors):
        color_tuple = tuple(color.astype(int))
        if color_tuple in color_to_label:
            labels[i] = color_to_label[color_tuple]
        else:
            # Find closest color
            min_dist = float('inf')
            best_label = 0
            for label_id, ref_color in g_label2color.items():
                dist = np.sum((color - ref_color) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_label = label_id
            labels[i] = best_label
    
    return labels


def print_scene_stats(scene_name, gt_colors, pred_colors):
    """Print statistics about the scene"""
    # Convert colors back to labels
    gt_labels = colors_to_labels(gt_colors)
    pred_labels = colors_to_labels(pred_colors)
    
    print("\n" + "="*60)
    print(f"Scene: {scene_name}")
    print("="*60)
    print(f"Total points: {len(gt_labels)}")
    
    # Accuracy
    correct = np.sum(gt_labels == pred_labels)
    accuracy = correct / len(gt_labels)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(gt_labels)} points correct)")
    
    # Per-class statistics
    print("\nPer-class breakdown:")
    print("-"*60)
    print(f"{'Class':<15} {'GT Points':<12} {'Pred Points':<12} {'Correct':<10} {'IoU':<10}")
    print("-"*60)
    
    for class_id, class_name in enumerate(classes):
        gt_mask = (gt_labels == class_id)
        pred_mask = (pred_labels == class_id)
        gt_count = np.sum(gt_mask)
        pred_count = np.sum(pred_mask)
        
        if gt_count > 0:
            correct_class = np.sum((gt_labels == class_id) & (pred_labels == class_id))
            union = np.sum((gt_labels == class_id) | (pred_labels == class_id))
            iou = correct_class / union if union > 0 else 0
            print(f"{class_name:<15} {gt_count:<12} {pred_count:<12} {correct_class:<10} {iou:<10.3f}")
    
    print("="*60 + "\n")


def create_point_cloud(xyz, colors):
    """Create Open3D point cloud from coordinates and colors"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Open3D expects 0-1 range
    return pcd


def visualize_scene(visual_dir, scene_name, show_gt=True, show_pred=True):
    """Visualize a scene with ground truth and predictions using Open3D"""
    print(f"\nLoading scene: {scene_name}")
    
    # Load data
    xyz, gt_colors, pred_colors = load_scene_data(visual_dir, scene_name)
    
    # Print statistics
    print_scene_stats(scene_name, gt_colors, pred_colors)
    
    # Create point clouds
    if show_gt and show_pred:
        print("\nShowing: Ground Truth (first) and Predictions (second)")
        print("Close window to see next view...")
        
        # Show ground truth
        pcd_gt = create_point_cloud(xyz, gt_colors)
        print("\n[1/2] Ground Truth - Press Q to close and view predictions")
        o3d.visualization.draw_geometries([pcd_gt], 
                                         window_name=f"{scene_name} - Ground Truth",
                                         width=1024, height=768,
                                         point_show_normal=False)
        
        # Show predictions
        pcd_pred = create_point_cloud(xyz, pred_colors)
        print("\n[2/2] Predictions - Press Q to close")
        o3d.visualization.draw_geometries([pcd_pred], 
                                         window_name=f"{scene_name} - Predictions",
                                         width=1024, height=768,
                                         point_show_normal=False)
        
    elif show_gt:
        print("\nShowing: Ground Truth only")
        pcd_gt = create_point_cloud(xyz, gt_colors)
        o3d.visualization.draw_geometries([pcd_gt], 
                                         window_name=f"{scene_name} - Ground Truth",
                                         width=1024, height=768,
                                         point_show_normal=False)
    elif show_pred:
        print("\nShowing: Predictions only")
        pcd_pred = create_point_cloud(xyz, pred_colors)
        o3d.visualization.draw_geometries([pcd_pred], 
                                         window_name=f"{scene_name} - Predictions",
                                         width=1024, height=768,
                                         point_show_normal=False)


def list_available_scenes(visual_dir):
    """List all available scenes in the visual directory"""
    txt_files = sorted([f for f in os.listdir(visual_dir) if f.endswith('.txt')])
    scene_names = [f[:-4] for f in txt_files]
    
    print("\n" + "="*60)
    print(f"Available scenes in: {visual_dir}")
    print("="*60)
    
    # Group by room type
    room_types = {}
    for scene in scene_names:
        parts = scene.split('_')
        if len(parts) >= 3:
            room_type = '_'.join(parts[1:-1])  # e.g., "office", "hallway", "conferenceRoom"
        else:
            room_type = "other"
        
        if room_type not in room_types:
            room_types[room_type] = []
        room_types[room_type].append(scene)
    
    for room_type, scenes in sorted(room_types.items()):
        print(f"\n{room_type.upper()} ({len(scenes)} scenes):")
        for i, scene in enumerate(scenes[:5], 1):  # Show first 5
            print(f"  {i}. {scene}")
        if len(scenes) > 5:
            print(f"  ... and {len(scenes) - 5} more")
    
    print("\n" + "="*60)
    print(f"Total: {len(scene_names)} scenes")
    print("\nTo visualize a specific scene, run:")
    print(f"  python view_semseg_results.py --visual_dir {visual_dir} --scene <scene_name>")
    print("\nExample:")
    print(f"  python view_semseg_results.py --visual_dir {visual_dir} --scene {scene_names[0]}")
    print("="*60 + "\n")


def main():
    args = parse_args()
    
    visual_dir = args.visual_dir
    
    if not os.path.exists(visual_dir):
        print(f"Error: Visual directory not found: {visual_dir}")
        return
    
    if args.scene is None:
        # List available scenes
        list_available_scenes(visual_dir)
    else:
        # Visualize specific scene
        try:
            visualize_scene(visual_dir, args.scene, args.show_gt, args.show_pred)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nAvailable scenes:")
            list_available_scenes(visual_dir)
        except Exception as e:
            print(f"Error visualizing scene: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
