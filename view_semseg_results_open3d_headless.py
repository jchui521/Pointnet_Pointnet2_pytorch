#!/usr/bin/env python3
"""
Enhanced Open3D Visualization with Headless Support
Automatically uses virtual display when available
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

# Try to import virtual display for headless rendering
try:
    from pyvirtualdisplay import Display
    VIRTUAL_DISPLAY_AVAILABLE = True
except ImportError:
    VIRTUAL_DISPLAY_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser('Open3D Semantic Segmentation Visualizer (Headless-Enabled)')
    parser.add_argument('--visual_dir', type=str, required=True,
                       help='Directory with visualization .obj files')
    parser.add_argument('--scene', type=str, default=None,
                       help='Specific scene name (without extension)')
    parser.add_argument('--show_gt', action='store_true', default=True,
                       help='Show ground truth')
    parser.add_argument('--show_pred', action='store_true', default=True,
                       help='Show predictions')
    parser.add_argument('--use_virtual_display', action='store_true', default=VIRTUAL_DISPLAY_AVAILABLE,
                       help=f'Use virtual display (default: {VIRTUAL_DISPLAY_AVAILABLE})')
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


def visualize_scene(pcd_gt, pcd_pred, scene_name, show_gt=True, show_pred=True):
    """Visualize scene with Open3D"""
    
    if show_gt and show_pred:
        print(f"\n[1/2] Showing Ground Truth...")
        print("      Left-click drag: Rotate | Right-click drag: Pan | Scroll: Zoom")
        print("      Press Q or close window to continue...")
        o3d.visualization.draw_geometries(
            [pcd_gt],
            window_name=f"{scene_name} - Ground Truth",
            width=1024,
            height=768,
            point_show_normal=False
        )
        
        print(f"\n[2/2] Showing Predictions...")
        print("      Press Q or close window to finish...")
        o3d.visualization.draw_geometries(
            [pcd_pred],
            window_name=f"{scene_name} - Predictions",
            width=1024,
            height=768,
            point_show_normal=False
        )
    
    elif show_gt:
        print(f"\nShowing Ground Truth only...")
        o3d.visualization.draw_geometries(
            [pcd_gt],
            window_name=f"{scene_name} - Ground Truth",
            width=1024,
            height=768,
            point_show_normal=False
        )
    
    elif show_pred:
        print(f"\nShowing Predictions only...")
        o3d.visualization.draw_geometries(
            [pcd_pred],
            window_name=f"{scene_name} - Predictions",
            width=1024,
            height=768,
            point_show_normal=False
        )


def main():
    args = parse_args()
    
    visual_dir = args.visual_dir
    
    if not os.path.exists(visual_dir):
        print(f"Error: Visual directory not found: {visual_dir}")
        return
    
    if args.scene is None:
        print("Error: --scene argument required")
        print(f"Example: python view_semseg_results_open3d_headless.py --visual_dir {visual_dir} --scene Area_5_conferenceRoom_1")
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
    
    # Setup virtual display if requested and available
    virtual_display = None
    if args.use_virtual_display and VIRTUAL_DISPLAY_AVAILABLE:
        try:
            print("\n[*] Starting virtual display...")
            virtual_display = Display(visible=False, size=(1024, 768))
            virtual_display.start()
            print("    Virtual display started successfully!")
        except Exception as e:
            print(f"    Warning: Could not start virtual display: {e}")
            virtual_display = None
    elif args.use_virtual_display and not VIRTUAL_DISPLAY_AVAILABLE:
        print("\n[!] Virtual display requested but not available")
        print("    Install with: pip install pyvirtualdisplay")
        print("    Or run: python setup_headless_open3d.py")
    
    # Visualize
    print("\n" + "="*70)
    print("Open3D Visualization Controls:")
    print("  Mouse: Left=Rotate, Right=Pan, Scroll=Zoom")
    print("  Keys: Q or close window = Continue")
    print("="*70)
    
    try:
        visualize_scene(pcd_gt, pcd_pred, args.scene, args.show_gt, args.show_pred)
        print("\n✓ Visualization complete!")
    except Exception as e:
        print(f"\n✗ Visualization error: {e}")
        print("\nHeadless environment detected. Using alternative visualization...")
        print(f"OBJ files are saved at: {visual_dir}")
        print("Open them with MeshLab, CloudCompare, or Blender")
    
    finally:
        # Stop virtual display
        if virtual_display is not None:
            try:
                virtual_display.stop()
                print("Virtual display stopped.")
            except:
                pass


if __name__ == '__main__':
    main()
