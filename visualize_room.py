"""Visualize .npy room files from data/stanford_indoor3d/ using Open3D.

Snapshot camera parameters (--front, --lookat, --up, --zoom):
  --front   direction the camera points FROM (e.g. 0 -1 0.5)
  --lookat  point the camera looks AT, defaults to point cloud centroid
  --up      camera up vector (default: 0 0 1)
  --zoom    zoom level (default: 0.5, smaller = more zoomed in)

Example
  python visualize_room.py data/stanford_indoor3d/Area_1_office_1.npy --snapshot office1.png --front 0 0 1 --up 0 1 0 --zoom 0.4
"""

import argparse
import json
import os
import sys
import colorsys

import numpy as np

try:
    import open3d as o3d
    from open3d.visualization import VisualizerWithKeyCallback as Visualizer  # type: ignore[attr-defined]
except ImportError:
    print("Open3D is required. Install it with: pip install open3d")
    sys.exit(1)

# CLASS_NAMES = [
#     "ceiling", "floor", "wall", "beam", "column", "window",
#     "door", "table", "chair", "sofa", "bookcase", "board", "clutter",
# ]

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

CLASS_COLORS = []
n = len(CLASS_NAMES)
for i, name in enumerate(CLASS_NAMES):
    hue = i / n
    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    CLASS_COLORS.append([c for c in rgb])
CLASS_COLORS = np.array(CLASS_COLORS)

def make_pcd(npy_path, color_mode="rgb"):
    data = np.load(npy_path)  # N x 7
    points = data[:, :3]
    rgb = data[:, 3:6] / 255.0
    labels = data[:, 6].astype(int)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(
        CLASS_COLORS[labels] if color_mode == "label" else rgb
    )
    return pcd, labels, points


def print_room_info(npy_path, color_mode, n_points, labels):
    room_name = os.path.splitext(os.path.basename(npy_path))[0]
    mode_str = "semantic labels" if color_mode == "label" else "RGB"
    print(f"Showing: {room_name} ({mode_str})  [{n_points} points]")
    if color_mode == "label":
        print("Classes present:")
        for c in np.unique(labels):
            print(f"  {c:2d} - {CLASS_NAMES[c]}")


def snapshot(pcd, output_path, front, lookat, up, zoom, width=1280, height=720):
    """Render and save an image without leaving a persistent window."""
    vis = Visualizer()
    vis.create_window(visible=True, width=width, height=height)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    ctr.set_front(front)
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path, do_render=True)
    vis.destroy_window()
    print(f"Saved: {output_path}")


def visualize(npy_path, color_mode="rgb", snap_args=None):
    pcd, labels, points = make_pcd(npy_path, color_mode)
    print_room_info(npy_path, color_mode, len(points), labels)

    room_name = os.path.splitext(os.path.basename(npy_path))[0]
    mode_str = "semantic labels" if color_mode == "label" else "RGB"
    title = f"{room_name} ({mode_str})"

    if snap_args is not None:
        front  = snap_args["front"]
        lookat = snap_args["lookat"] if snap_args["lookat"] is not None else points.mean(axis=0).tolist()
        up     = snap_args["up"]
        zoom   = snap_args["zoom"]
        output = snap_args["output"]
        w, h   = snap_args["width"], snap_args["height"]
        snapshot(pcd, output, front, lookat, up, zoom, w, h)
    else:
        def print_camera(vis):
            traj = json.loads(vis.get_view_status())["trajectory"][0]
            f = traj["front"]
            l = traj["lookat"]
            u = traj["up"]
            z = traj["zoom"]
            print(
                f"\n--- camera ---\n"
                f"  --front  {f[0]:.4f} {f[1]:.4f} {f[2]:.4f}\n"
                f"  --lookat {l[0]:.4f} {l[1]:.4f} {l[2]:.4f}\n"
                f"  --up     {u[0]:.4f} {u[1]:.4f} {u[2]:.4f}\n"
                f"  --zoom   {z:.4f}\n"
            )
            return False  # don't close the window

        vis = Visualizer()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(pcd)
        vis.register_key_callback(ord("P"), print_camera)
        print("  [P] print camera params")
        vis.run()
        vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize S3DIS room point clouds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("file", nargs="?", help="Path to a specific .npy file.")
    parser.add_argument(
        "--dir", default="data/stanford_indoor3d",
        help="Directory of .npy files (default: data/stanford_indoor3d).",
    )
    parser.add_argument(
        "--color", choices=["rgb", "label"], default="rgb",
        help="Color mode: rgb or label (default: rgb).",
    )

    # Snapshot options
    snap = parser.add_argument_group("snapshot options")
    snap.add_argument(
        "--snapshot", metavar="OUTPUT.PNG",
        help="Save an image to this path instead of opening an interactive window.",
    )
    snap.add_argument(
        "--front", nargs=3, type=float, default=[0.0, -1.0, 0.5], metavar=("X", "Y", "Z"),
        help="Camera front direction vector (default: 0 -1 0.5).",
    )
    snap.add_argument(
        "--lookat", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
        help="Point the camera looks at (default: centroid of point cloud).",
    )
    snap.add_argument(
        "--up", nargs=3, type=float, default=[0.0, 0.0, 1.0], metavar=("X", "Y", "Z"),
        help="Camera up vector (default: 0 0 1).",
    )
    snap.add_argument(
        "--zoom", type=float, default=0.5,
        help="Zoom level (default: 0.5, smaller = more zoomed in).",
    )
    snap.add_argument("--width",  type=int, default=1280, help="Image width  (default: 1280).")
    snap.add_argument("--height", type=int, default=720,  help="Image height (default: 720).")

    args = parser.parse_args()

    snap_args = None
    if args.snapshot:
        snap_args = {
            "output": args.snapshot,
            "front":  args.front,
            "lookat": args.lookat,
            "up":     args.up,
            "zoom":   args.zoom,
            "width":  args.width,
            "height": args.height,
        }

    if args.file:
        visualize(args.file, args.color, snap_args)
        return

    npy_dir = args.dir
    files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy")])
    if not files:
        print(f"No .npy files found in {npy_dir}")
        return

    print(f"Found {len(files)} rooms.\n")
    for i, f in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {f}")
        out = snap_args
        # Auto-name outputs when batch snapshotting
        if snap_args is not None:
            base, ext = os.path.splitext(snap_args["output"])
            out = {**snap_args, "output": f"{base}_{os.path.splitext(f)[0]}{ext}"}
        visualize(os.path.join(npy_dir, f), args.color, out)
        if out is None and i < len(files) - 1:
            resp = input("Next room? (Enter=yes, q=quit): ").strip().lower()
            if resp == "q":
                break


if __name__ == "__main__":
    main()
