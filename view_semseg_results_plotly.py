#!/usr/bin/env python3
"""
3D Semantic Segmentation Visualizer using Plotly
Generates interactive HTML visualizations of predictions vs ground truth
No display/GUI required - opens in browser or VS Code
"""

import numpy as np
import argparse
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add data_utils to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))

from indoor3d_util import g_label2color

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']


def parse_args():
    parser = argparse.ArgumentParser('3D Semantic Segmentation Visualizer (Plotly)')
    parser.add_argument('--visual_dir', type=str, required=True,
                       help='Directory with visualization files')
    parser.add_argument('--scene', type=str, default=None,
                       help='Specific scene to visualize (e.g., Area_5_conferenceRoom_1)')
    parser.add_argument('--output_html', type=str, default=None,
                       help='Output HTML file path. Default: visual_dir/<scene>_comparison.html')
    parser.add_argument('--sample_rate', type=float, default=0.1,
                       help='Sample rate for visualization (1.0 = all points, 0.1 = 10%% of points) [default: 0.1]')
    parser.add_argument('--show_gt', action='store_true', default=True,
                       help='Show ground truth')
    parser.add_argument('--show_pred', action='store_true', default=True,
                       help='Show predictions')
    return parser.parse_args()


def load_obj_file(obj_file, sample_rate=1.0):
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
    
    vertices = np.array(vertices)
    colors = np.array(colors)
    
    # Sample if needed
    if sample_rate < 1.0:
        n_points = len(vertices)
        n_sample = int(n_points * sample_rate)
        indices = np.random.choice(n_points, n_sample, replace=False)
        vertices = vertices[indices]
        colors = colors[indices]
    
    return vertices, colors


def load_scene_data(visual_dir, scene_name, sample_rate=1.0):
    """Load scene point cloud from .obj files"""
    gt_file = os.path.join(visual_dir, scene_name + '_gt.obj')
    pred_file = os.path.join(visual_dir, scene_name + '_pred.obj')
    
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    
    # Load ground truth
    xyz_gt, colors_gt = load_obj_file(gt_file, sample_rate)
    
    # Load predictions
    xyz_pred, colors_pred = load_obj_file(pred_file, sample_rate)
    
    if len(xyz_gt) != len(xyz_pred):
        print(f"Warning: GT and Pred have different number of points ({len(xyz_gt)} vs {len(xyz_pred)})")
    
    # Use GT coordinates
    xyz = xyz_gt
    
    return xyz, colors_gt, colors_pred


def colors_to_labels(colors):
    """Convert RGB colors back to semantic labels"""
    labels = np.zeros(len(colors), dtype=int)
    
    # Build reverse mapping
    color_to_label = {}
    for label_id, color in g_label2color.items():
        # Convert to 0-1 range for comparison
        color_01 = tuple(np.array(color) / 255.0)
        color_to_label[color_01] = label_id
    
    for i, color in enumerate(colors):
        color_tuple = tuple(color)
        if color_tuple in color_to_label:
            labels[i] = color_to_label[color_tuple]
        else:
            # Find closest color
            min_dist = float('inf')
            best_label = 0
            for label_id, ref_color in g_label2color.items():
                ref_color_01 = np.array(ref_color) / 255.0
                dist = np.sum((color - ref_color_01) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_label = label_id
            labels[i] = best_label
    
    return labels


def compute_accuracy(gt_labels, pred_labels):
    """Compute accuracy metrics"""
    correct = np.sum(gt_labels == pred_labels)
    accuracy = correct / len(gt_labels)
    
    # Per-class accuracy
    per_class_acc = {}
    for class_id, class_name in enumerate(classes):
        mask = gt_labels == class_id
        if mask.sum() > 0:
            class_correct = np.sum((gt_labels == class_id) & (pred_labels == class_id))
            class_acc = class_correct / mask.sum()
            per_class_acc[class_name] = {
                'accuracy': class_acc,
                'count': mask.sum(),
                'correct': class_correct
            }
    
    return accuracy, per_class_acc


def create_3d_scatter(xyz, colors, label=None, max_points=100000):
    """Create 3D scatter trace for Plotly"""
    # Limit points for performance
    if len(xyz) > max_points:
        print(f"Sampling from {len(xyz)} to {max_points} points for visualization")
        indices = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[indices]
        colors = colors[indices]
    
    # Convert colors from 0-1 to 0-255 range for Plotly
    colors_255 = (colors * 255).astype(int)
    color_strings = [f'rgb({r},{g},{b})' for r, g, b in colors_255]
    
    trace = go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=color_strings,
            opacity=0.8,
        ),
        name=label if label else 'Points',
        text=[f'Point {i}' for i in range(len(xyz))],
        hoverinfo='x+y+z+text'
    )
    
    return trace


def create_comparison_visualization(xyz, colors_gt, colors_pred, scene_name, output_html):
    """Create interactive side-by-side comparison visualization"""
    print(f"Creating visualization for {scene_name}...")
    
    # Create subplots (1 row, 2 columns)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(f'{scene_name} - Ground Truth', f'{scene_name} - Predictions')
    )
    
    # Add ground truth trace
    print("  Adding ground truth trace...")
    gt_trace = create_3d_scatter(xyz, colors_gt, label='Ground Truth')
    fig.add_trace(gt_trace, row=1, col=1)
    
    # Add prediction trace
    print("  Adding prediction trace...")
    pred_trace = create_3d_scatter(xyz, colors_pred, label='Predictions')
    fig.add_trace(pred_trace, row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Semantic Segmentation: {scene_name}<br><sub>Ground Truth vs Predictions</sub>',
            x=0.5,
            xanchor='center'
        ),
        height=800,
        width=1600,
        showlegend=True,
        hovermode='closest',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        scene2=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        )
    )
    
    # Save HTML
    print(f"  Saving HTML to {output_html}...")
    fig.write_html(output_html)
    print(f"✓ Visualization saved: {output_html}")
    
    return output_html


def create_single_visualization(xyz, colors, scene_name, title, output_html):
    """Create single 3D visualization"""
    print(f"Creating {title} visualization for {scene_name}...")
    
    fig = go.Figure()
    
    # Add trace
    print("  Adding trace...")
    trace = create_3d_scatter(xyz, colors, label=title)
    fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{title}: {scene_name}',
            x=0.5,
            xanchor='center'
        ),
        height=800,
        width=1000,
        showlegend=True,
        hovermode='closest',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        )
    )
    
    # Save HTML
    print(f"  Saving HTML to {output_html}...")
    fig.write_html(output_html)
    print(f"✓ Visualization saved: {output_html}")
    
    return output_html


def main():
    args = parse_args()
    
    visual_dir = args.visual_dir
    
    if not os.path.exists(visual_dir):
        print(f"Error: Visual directory not found: {visual_dir}")
        return
    
    if args.scene is None:
        print(f"Error: Scene name required. Use --scene <scene_name>")
        print(f"Visual files in: {visual_dir}")
        return
    
    try:
        # Load data
        print(f"\nLoading scene: {args.scene}")
        print(f"  Sample rate: {args.sample_rate} ({int(args.sample_rate*100)}%% of points)")
        xyz, colors_gt, colors_pred = load_scene_data(visual_dir, args.scene, args.sample_rate)
        print(f"  Loaded {len(xyz)} points")
        
        # Compute accuracy
        gt_labels = colors_to_labels(colors_gt)
        pred_labels = colors_to_labels(colors_pred)
        accuracy, per_class_acc = compute_accuracy(gt_labels, pred_labels)
        
        print(f"\n  Overall Accuracy: {accuracy*100:.2f}%")
        print(f"\n  Per-class accuracy:")
        for class_name, stats in per_class_acc.items():
            print(f"    {class_name:15s}: {stats['accuracy']*100:5.2f}% ({stats['correct']:6d}/{stats['count']:6d})")
        
        # Determine output path
        if args.output_html is None:
            output_html = os.path.join(visual_dir, f"{args.scene}_comparison.html")
        else:
            output_html = args.output_html
        
        # Create visualization
        if args.show_gt and args.show_pred:
            create_comparison_visualization(xyz, colors_gt, colors_pred, args.scene, output_html)
        elif args.show_gt:
            create_single_visualization(xyz, colors_gt, args.scene, "Ground Truth", output_html)
        elif args.show_pred:
            create_single_visualization(xyz, colors_pred, args.scene, "Predictions", output_html)
        
        print(f"\n✓ Open this file in your browser or VS Code Simple Browser:")
        print(f"  {os.path.abspath(output_html)}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error visualizing scene: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
