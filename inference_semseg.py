"""
Inference script for RPI semantic segmentation model.
Runs whole-scene inference with voting on .npy point cloud files.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "models"))

from data_utils.RPIDataLoader import DatasetWholeScene

CLASSES_FILE = "data/rpi_custom_dataset_01_raw/classes.txt"
NUM_CLASSES = 21


def parse_args():
    parser = argparse.ArgumentParser("RPI Semantic Segmentation Inference")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Log directory under log/sem_seg_rpi/ (e.g. 2026-02-24_15-43)")
    parser.add_argument("--data_root", type=str, default="data/rpi_data",
                        help="Directory containing .npy scene files [default: data/rpi_data]")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference [default: 32]")
    parser.add_argument("--num_point", type=int, default=4096,
                        help="Points per block [default: 4096]")
    parser.add_argument("--num_votes", type=int, default=3,
                        help="Number of voting passes per scene [default: 3]")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device [default: 0]")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save predictions [default: <log_dir>/predictions]")
    parser.add_argument("--visual", action="store_true", default=False,
                        help="Save colored .obj files for visualization")
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    for b in range(pred_label.shape[0]):
        for n in range(pred_label.shape[1]):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment_dir = Path("log/sem_seg_rpi") / args.log_dir
    if not experiment_dir.exists():
        print(f"ERROR: Log directory not found: {experiment_dir}")
        sys.exit(1)

    checkpoint_path = experiment_dir / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        checkpoint_path = experiment_dir / "checkpoints" / "model.pth"
    if not checkpoint_path.exists():
        print(f"ERROR: No checkpoint found in {experiment_dir / 'checkpoints'}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [line.rstrip() for line in open(CLASSES_FILE)]
    label2class = {i: cls for i, cls in enumerate(classes)}

    print(f"Loading scenes from {args.data_root} ...")
    dataset = DatasetWholeScene(
        root=args.data_root,
        block_points=args.num_point,
        block_size=1.0,
        stride=0.5,
        padding=0.001,
        num_classes=NUM_CLASSES,
    )
    print(f"Loaded {len(dataset)} scenes.")

    model_name = next(
        (f[:-3] for f in os.listdir(str(experiment_dir)) if f.endswith(".py") and "pointnet" in f),
        "pointnet2_sem_seg"
    )
    sys.path.insert(0, str(experiment_dir))
    MODEL = importlib.import_module(model_name)
    sys.path.pop(0)

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(checkpoint_path), weights_only=False)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    print(f"Loaded model '{model_name}' from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    scene_ids = [f[:-4] for f in dataset.file_list]

    with torch.no_grad():
        for scene_idx in range(len(dataset)):
            scene_id = scene_ids[scene_idx]
            print(f"\n[{scene_idx+1}/{len(dataset)}] Inference on: {scene_id}")

            whole_scene_data = dataset.scene_points_list[scene_idx]     # (N, 6)
            vote_label_pool = np.zeros((whole_scene_data.shape[0], NUM_CLASSES))

            for _ in tqdm(range(args.num_votes), desc="Voting"):
                scene_data, _, scene_smpw, scene_point_index = dataset[scene_idx]

                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + args.batch_size - 1) // args.batch_size

                batch_data = np.zeros((args.batch_size, args.num_point, 9))
                batch_point_index = np.zeros((args.batch_size, args.num_point))
                batch_smpw = np.zeros((args.batch_size, args.num_point))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * args.batch_size
                    end_idx = min((sbatch + 1) * args.batch_size, num_blocks)
                    real_batch_size = end_idx - start_idx

                    batch_data[:real_batch_size] = scene_data[start_idx:end_idx]
                    batch_point_index[:real_batch_size] = scene_point_index[start_idx:end_idx]
                    batch_smpw[:real_batch_size] = scene_smpw[start_idx:end_idx]

                    torch_data = torch.Tensor(batch_data).float().cuda()
                    torch_data = torch_data.transpose(2, 1)  # (B, 9, N)

                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(
                        vote_label_pool,
                        batch_point_index[:real_batch_size],
                        batch_pred_label[:real_batch_size],
                        batch_smpw[:real_batch_size],
                    )

            pred_label = np.argmax(vote_label_pool, axis=1)

            np.savetxt(str(output_dir / f"{scene_id}_pred.txt"), pred_label.astype(int), fmt="%d")

            pred_npy = np.concatenate(
                [whole_scene_data, pred_label[:, np.newaxis].astype(np.float32)], axis=1
            )
            np.save(str(output_dir / f"{scene_id}_pred.npy"), pred_npy)

            if args.visual:
                import colorsys
                colors = [
                    [int(c * 255) for c in colorsys.hsv_to_rgb(i / NUM_CLASSES, 0.9, 0.9)]
                    for i in range(NUM_CLASSES)
                ]
                with open(str(output_dir / f"{scene_id}_pred.obj"), "w") as fout:
                    for i in range(whole_scene_data.shape[0]):
                        c = colors[int(pred_label[i])]
                        fout.write("v %f %f %f %d %d %d\n" % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2],
                            c[0], c[1], c[2]
                        ))

    print("\n" + "=" * 50)
    print("INFERENCE COMPLETE")
    print(f"\nPredictions saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
