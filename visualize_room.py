"""Visualize .npy room files from data/stanford_indoor3d/ using Open3D."""

import argparse
import os
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Open3D is required. Install it with: pip install open3d")
    sys.exit(1)

CLASS_NAMES = [
    "ceiling", "floor", "wall", "beam", "column", "window",
    "door", "table", "chair", "sofa", "bookcase", "board", "clutter",
]

# Distinct colors per class (RGB 0-1)
CLASS_COLORS = np.array([
    [0.9, 0.1, 0.1],  # ceiling - red
    [0.6, 0.4, 0.2],  # floor - brown
    [0.8, 0.8, 0.8],  # wall - light gray
    [1.0, 0.6, 0.0],  # beam - orange
    [0.5, 0.0, 0.5],  # column - purple
    [0.0, 0.6, 1.0],  # window - light blue
    [0.0, 0.3, 0.7],  # door - dark blue
    [1.0, 1.0, 0.0],  # table - yellow
    [0.0, 0.8, 0.0],  # chair - green
    [1.0, 0.4, 0.7],  # sofa - pink
    [0.4, 0.2, 0.0],  # bookcase - dark brown
    [0.0, 1.0, 1.0],  # board - cyan
    [0.5, 0.5, 0.5],  # clutter - gray
])


def visualize(npy_path, color_mode="rgb"):
    data = np.load(npy_path)  # N x 7
    points = data[:, :3]
    rgb = data[:, 3:6] / 255.0
    labels = data[:, 6].astype(int)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if color_mode == "label":
        colors = CLASS_COLORS[labels]
    else:
        colors = rgb

    pcd.colors = o3d.utility.Vector3dVector(colors)

    room_name = os.path.splitext(os.path.basename(npy_path))[0]
    mode_str = "semantic labels" if color_mode == "label" else "RGB"
    title = f"{room_name} ({mode_str})"
    print(f"Showing: {title}  ({len(points)} points)")

    if color_mode == "label":
        unique = np.unique(labels)
        print("Classes present:")
        for c in unique:
            print(f"  {c:2d} - {CLASS_NAMES[c]}")

    o3d.visualization.draw_geometries([pcd], window_name=title, width=1280, height=720)


def main():
    parser = argparse.ArgumentParser(description="Visualize S3DIS room point clouds.")
    parser.add_argument("file", nargs="?", help="Path to a specific .npy file.")
    parser.add_argument(
        "--dir", default="data/stanford_indoor3d",
        help="Directory containing .npy files (default: data/stanford_indoor3d).",
    )
    parser.add_argument(
        "--color", choices=["rgb", "label"], default="rgb",
        help="Color mode: 'rgb' for original colors, 'label' for semantic class colors (default: rgb).",
    )
    args = parser.parse_args()

    if args.file:
        visualize(args.file, args.color)
        return

    npy_dir = args.dir
    files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy")])
    if not files:
        print(f"No .npy files found in {npy_dir}")
        return

    print(f"Found {len(files)} rooms. Press 'n' for next, 'q' to quit in the list below.\n")
    for i, f in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {f}")
        visualize(os.path.join(npy_dir, f), args.color)
        if i < len(files) - 1:
            resp = input("Next room? (Enter=yes, q=quit): ").strip().lower()
            if resp == "q":
                break


if __name__ == "__main__":
    main()
