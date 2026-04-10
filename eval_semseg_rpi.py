"""
Evaluation script for PointNet++ semantic segmentation on the RPI dataset.
Usage:
    python eval_semseg_rpi.py --log_dir <experiment_folder>
Example:
    python eval_semseg_rpi.py --log_dir 2026-04-10_12-00
"""
import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_utils.RPIDataLoader import PointNetDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
NUM_CLASSES = len(classes)
seg_label_to_cat = {i: cls for i, cls in enumerate(classes)}


def parse_args():
    parser = argparse.ArgumentParser("eval_semseg_rpi")
    parser.add_argument("--log_dir", type=str, required=True, help="Experiment folder under log/sem_seg_rpi/")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size [default: 16]")
    parser.add_argument("--num_point", type=int, default=4096, help="Point number [default: 4096]")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers [default: 4]")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use [default: 0]")
    parser.add_argument("--data_root", type=str, default="rpi_data", help="Path to data root [default: rpi_data]")
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment_dir = Path("log/sem_seg_rpi") / args.log_dir
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    log_dir = experiment_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Eval")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(str(log_dir / "eval.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(s):
        logger.info(s)
        print(s)

    log_string(f"Args: {args}")
    log_string(f"NUM_CLASSES: {NUM_CLASSES}  classes: {classes}")

    # Data
    log_string("Loading test data ...")
    test_dataset = PointNetDataset(
        split="test",
        data_root=args.data_root,
        num_point=args.num_point,
        num_classes=NUM_CLASSES,
        block_size=1.0,
        sample_rate=1.0,
        transform=None,
    )
    log_string(f"Number of test samples: {len(test_dataset)}")

    # Use training split only to compute class weights (same as training)
    train_dataset = PointNetDataset(
        split="train",
        data_root=args.data_root,
        num_point=args.num_point,
        num_classes=NUM_CLASSES,
        block_size=1.0,
        sample_rate=1.0,
        transform=None,
    )
    weights = torch.Tensor(train_dataset.labelweights).cuda()

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model_name = os.listdir(str(experiment_dir / "logs"))[0].split(".")[0]
    log_string(f"Loading model: {model_name}")
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()

    checkpoint_path = str(experiment_dir / "checkpoints" / "best_model.pth")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    log_string(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} (best mIoU: {checkpoint.get('class_avg_iou', '?')})")

    classifier = classifier.eval()

    # Eval loop
    with torch.no_grad():
        num_batches = len(test_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0.0
        total_seen_class = [0] * NUM_CLASSES
        total_correct_class = [0] * NUM_CLASSES
        total_iou_deno_class = [0] * NUM_CLASSES

        log_string("---- EVALUATION ----")
        for points, target in tqdm(test_loader, total=num_batches, smoothing=0.9):
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()   # B x N x C
            seg_pred_flat = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()               # B x N
            target_flat = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred_flat, target_flat, trans_feat, weights)
            loss_sum += loss.item()

            pred_val = np.argmax(pred_val, 2)                     # B x N
            correct = np.sum(pred_val == batch_label)
            total_correct += correct
            total_seen += batch_label.size

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum(batch_label == l)
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum((pred_val == l) | (batch_label == l))

    # Metrics
    mIoU = np.mean(
        np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
    )
    overall_acc = total_correct / float(total_seen)
    mean_class_acc = np.mean(
        np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6)
    )

    log_string(f"Eval mean loss:              {loss_sum / num_batches:.6f}")
    log_string(f"Eval overall accuracy:       {overall_acc:.6f}")
    log_string(f"Eval mean class accuracy:    {mean_class_acc:.6f}")
    log_string(f"Eval mIoU:                   {mIoU:.6f}")

    iou_per_class_str = "------- IoU per class --------\n"
    for l in range(NUM_CLASSES):
        iou = total_correct_class[l] / (float(total_iou_deno_class[l]) + 1e-6)
        iou_per_class_str += "class %-20s  seen: %8d  IoU: %.4f\n" % (
            seg_label_to_cat[l], total_seen_class[l], iou
        )
    log_string(iou_per_class_str)

    log_string("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
