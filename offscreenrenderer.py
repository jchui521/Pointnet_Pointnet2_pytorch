import argparse
import json
import os
import sys
import colorsys
import numpy as np

try:
    import open3d as o3d
    from open3d.visualization.rendering import rendering  # type: ignore[attr-defined]
except ImportError:
    print("Open3D is required. Install it with: pip install open3d")
    sys.exit(1)

os.environ["OPEN3D_CPU_RENDERING"] = "1"

# CLASS_NAMES = [
#     "ceiling", "floor", "wall", "beam", "column", "window",
#     "door", "table", "chair", "sofa", "bookcase", "board", "clutter",
# ]

# # Distinct colors per class (RGB 0-1)
# CLASS_COLORS = np.array([
#     [0.9, 0.1, 0.1],  # ceiling - red
#     [0.6, 0.4, 0.2],  # floor - brown
#     [0.8, 0.8, 0.8],  # wall - light gray
#     [1.0, 0.6, 0.0],  # beam - orange
#     [0.5, 0.0, 0.5],  # column - purple
#     [0.0, 0.6, 1.0],  # window - light blue
#     [0.0, 0.3, 0.7],  # door - dark blue
#     [1.0, 1.0, 0.0],  # table - yellow
#     [0.0, 0.8, 0.0],  # chair - green
#     [1.0, 0.4, 0.7],  # sofa - pink
#     [0.4, 0.2, 0.0],  # bookcase - dark brown
#     [0.0, 1.0, 1.0],  # board - cyan
#     [0.5, 0.5, 0.5],  # clutter - gray
# ])
CLASS_NAMES = ['Elbow',
 'Stairs',
 'Mechanical_Equipment',
 'Conduit_Elbow',
 'Duct',
 'HSS_Channel',
 'Wall',
 'Electrical_Equipment',
 'Conduit',
 'Light',
 'Reducer',
 'Valve',
 'Pipe',
 'Transition',
 'Floor',
 'Receptacle',
 'Tee',
 'Pressure_Gauge',
 'Mullion',
 'Coupling',
 'C_Channel']

CLASS_COLORS = []
n = len(CLASS_NAMES)
for i, name in enumerate(CLASS_NAMES):
    hue = i / n
    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    CLASS_COLORS.append([c for c in rgb])
CLASS_COLORS = np.array(CLASS_COLORS)

def make_pcd_from_numpy(data, color_mode="rgb"):
    pc = data[:, :3]
    rgb = data[:, 3:6]
    labels = data[:, 6].astype(int)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(
        CLASS_COLORS[labels] if color_mode == "label" else rgb
    )
    return pcd, labels, pc

def print_room_info(npy_path, color_mode, n_points, labels):
    room_name = os.path.splitext(os.path.basename(npy_path))[0]
    mode_str = "semantic labels" if color_mode == "label" else "RGB"
    print(f"Showing: {room_name} ({mode_str})  [{n_points} points]")
    if color_mode == "label":
        print("Classes present:")
        for c in np.unique(labels):
            print(f"  {c:2d} - {CLASS_NAMES[c]}")

def take_snapshot(npy, color_mode, output, front=(0, 0, 1), lookat=(0, 0, 0), up=(0, 0, 1), zoom=0.5, w=1280, h=720):
    if type(npy) == str:
        npy = np.load(npy)
    pcd, labels, points = make_pcd_from_numpy(npy, color_mode)
    print_room_info(npy, color_mode, len(points), labels)

    render = rendering.OffscreenRenderer(w, h)
    render.set_camera(front, lookat, up, zoom)
    render.add_geometry(pcd)

    img = render.render_to_image()
    o3d.io.write_image(output, img)
    print(f"Saved: {output}")

    render.destroy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str, default=None)
    parser.add_argument("--color_mode", type=str, default="rgb")
    parser.add_argument("--front", type=float, nargs=3, default=None)
    parser.add_argument("--lookat", type=float, nargs=3, default=None)
    parser.add_argument("--up", type=float, nargs=3, default=None)
    parser.add_argument("--zoom", type=float, default=None)
    parser.add_argument("--w", type=int, default=1280)
    parser.add_argument("--h", type=int, default=720)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    take_snapshot(args.npy_path, args.color_mode, args.output, args.front, args.lookat, args.up, args.zoom, args.w, args.h)

