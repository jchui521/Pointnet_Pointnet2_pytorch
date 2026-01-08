"""Open3D semantic segmentation viewer for single S3DIS rooms.

Workflow:
1) Load a single room file (.txt or .npy) with XYZRGBL columns.
2) Optionally run inference with a trained PointNet++ semantic model (PointNet2).
3) Show two Open3D windows sequentially: ground-truth colored points, then predicted colored points.

Controls: use Open3D defaults (drag to rotate, scroll to zoom, Q or ESC to close window).
"""
import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Lazy-install Open3D if missing so users can just run the script.
def _ensure_open3d():
    try:
        import open3d as o3d  # noqa: F401
        return
    except ModuleNotFoundError:
        print("Open3D not found; installing into current interpreter ...")
        cmd = [sys.executable, "-m", "pip", "install", "open3d"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("Failed to install open3d; please install manually with 'pip install open3d'.")


_ensure_open3d()
# Ensure project root and models directory are importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)
import open3d as o3d
import torch

from data_utils.indoor3d_util import g_label2color
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene

# -----------------------------
# Utility helpers
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Open3D semantic viewer")
    parser.add_argument("--scene_file", type=str, required=True, help="Path to single scene file (.txt or .npy) with XYZRGBL columns")
    parser.add_argument("--log_dir", type=str, default="pointnet2_sem_seg_eval", help="Experiment subdir under log/sem_seg containing checkpoints")
    parser.add_argument("--weights", type=str, default=None, help="Optional path to .pth checkpoint (default: best_model.pth inside log_dir)")
    parser.add_argument("--model", type=str, default=None, help="Model module name (default: inferred from log_dir/logs directory)")
    parser.add_argument("--num_point", type=int, default=4096, help="Points per block for inference")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--stride", type=float, default=0.5, help="Block stride for tiling the scene")
    parser.add_argument("--block_size", type=float, default=1.0, help="Block size (meters)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--skip_infer", action="store_true", help="Skip inference and only show ground-truth colors")
    parser.add_argument("--save_pred", type=str, default=None, help="Optional path to save predicted labels as plain-text (one label per line)")
    return parser.parse_args()


def load_scene(scene_file: str) -> Tuple[str, np.ndarray, np.ndarray]:
    if scene_file.endswith(".npy"):
        data = np.load(scene_file)
    else:
        data = np.loadtxt(scene_file)
    if data.shape[1] < 7:
        raise ValueError("Scene file must have at least 7 columns: x y z r g b label")
    scene_points = data[:, :6]
    scene_labels = data[:, 6].astype(int)
    scene_id = Path(scene_file).stem
    return scene_id, scene_points, scene_labels


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.array([g_label2color[int(l)] for l in labels], dtype=np.float32) / 255.0
    return colors


def show_cloud(points: np.ndarray, colors: np.ndarray, title: str) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)
    vis.get_render_option().background_color = np.array([0, 0, 0], dtype=np.float32)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


# -----------------------------
# Inference helpers
# -----------------------------

def make_single_scene_dataset(scene_id: str, scene_points: np.ndarray, scene_labels: np.ndarray,
                              block_points: int, stride: float, block_size: float) -> ScannetDatasetWholeScene:
    class SingleSceneDataset:
        def __init__(self):
            self.block_points = block_points
            self.block_size = block_size
            self.padding = 0.001
            self.stride = stride
            self.scene_points_list = [scene_points]
            self.semantic_labels_list = [scene_labels]
            self.raw_scene_points = [scene_points]
            self.file_list = [scene_id + ".npy"]
            labelweights = np.zeros(13)
            tmp, _ = np.histogram(scene_labels, range(14))
            labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            points = self.scene_points_list[index][:, :6]
            labels = self.semantic_labels_list[index]
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
            grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
            data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]), np.array([])
            for index_y in range(0, grid_y):
                for index_x in range(0, grid_x):
                    s_x = coord_min[0] + index_x * self.stride
                    e_x = min(s_x + self.block_size, coord_max[0])
                    s_x = e_x - self.block_size
                    s_y = coord_min[1] + index_y * self.stride
                    e_y = min(s_y + self.block_size, coord_max[1])
                    s_y = e_y - self.block_size
                    point_idxs = np.where(
                        (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) &
                        (points[:, 1] >= s_y - self.padding) & (points[:, 1] <= e_y + self.padding))[0]
                    if point_idxs.size == 0:
                        continue
                    num_batch = int(np.ceil(point_idxs.size / self.block_points))
                    point_size = int(num_batch * self.block_points)
                    replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                    point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                    point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                    np.random.shuffle(point_idxs)
                    data_batch = points[point_idxs, :]
                    normlized_xyz = np.zeros((point_size, 3))
                    normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                    normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                    normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                    data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                    data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                    data_batch[:, 3:6] /= 255.0
                    data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                    label_batch = labels[point_idxs].astype(int)
                    batch_weight = np.ones_like(label_batch, dtype=np.float32)

                    data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                    label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                    sample_weight = np.hstack([sample_weight, batch_weight]) if sample_weight.size else batch_weight
                    index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
            data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
            label_room = label_room.reshape((-1, self.block_points))
            sample_weight = sample_weight.reshape((-1, self.block_points))
            index_room = index_room.reshape((-1, self.block_points))
            return data_room, label_room, sample_weight, index_room

    return SingleSceneDataset()


def add_vote(vote_label_pool: np.ndarray, point_idx: np.ndarray, pred_label: np.ndarray, weight: np.ndarray) -> np.ndarray:
    B, N = pred_label.shape
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def infer_scene(scene_id: str, scene_points: np.ndarray, scene_labels: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    NUM_CLASSES = 13
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    experiment_dir = Path("log/sem_seg") / args.log_dir
    model_log_dir = experiment_dir / "logs"
    model_name = args.model or os.listdir(model_log_dir)[0].split(".")[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).to(device)

    weights_path = Path(args.weights) if args.weights else experiment_dir / "checkpoints" / "best_model.pth"
    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    dataset = make_single_scene_dataset(scene_id, scene_points, scene_labels,
                                        block_points=args.num_point, stride=args.stride, block_size=args.block_size)

    scene_data, scene_label, scene_smpw, scene_point_index = dataset[0]
    num_blocks = scene_data.shape[0]
    s_batch_num = (num_blocks + args.batch_size - 1) // args.batch_size
    # Use the number of original scene points for the voting pool size, so indices line up.
    num_scene_points = dataset.raw_scene_points[0].shape[0]
    vote_label_pool = np.zeros((num_scene_points, NUM_CLASSES), dtype=np.float32)

    batch_data = np.zeros((args.batch_size, args.num_point, 9))
    batch_label = np.zeros((args.batch_size, args.num_point))
    batch_point_index = np.zeros((args.batch_size, args.num_point))
    batch_smpw = np.zeros((args.batch_size, args.num_point))

    with torch.no_grad():
        for sbatch in range(s_batch_num):
            start_idx = sbatch * args.batch_size
            end_idx = min((sbatch + 1) * args.batch_size, num_blocks)
            real_batch_size = end_idx - start_idx
            batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
            batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
            batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
            batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]

            torch_data = torch.tensor(batch_data[:real_batch_size], dtype=torch.float32, device=device)
            torch_data = torch_data.transpose(2, 1)
            seg_pred, _ = classifier(torch_data)
            batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
            vote_label_pool = add_vote(
                vote_label_pool,
                batch_point_index[0:real_batch_size, ...],
                batch_pred_label[0:real_batch_size, ...],
                batch_smpw[0:real_batch_size, ...],
            )

    pred_label = np.argmax(vote_label_pool, axis=1).astype(int)
    return pred_label


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    scene_id, scene_points, scene_labels = load_scene(args.scene_file)
    gt_colors = labels_to_colors(scene_labels)

    show_cloud(scene_points, gt_colors, f"GT: {scene_id}")

    if args.skip_infer:
        print("Inference skipped; only ground truth displayed.")
        return

    pred_labels = infer_scene(scene_id, scene_points, scene_labels, args)
    pred_colors = labels_to_colors(pred_labels)

    if args.save_pred:
        np.savetxt(args.save_pred, pred_labels.astype(int), fmt="%d")
        print(f"Saved predictions to {args.save_pred}")

    show_cloud(scene_points, pred_colors, f"Pred: {scene_id}")


if __name__ == "__main__":
    main()
