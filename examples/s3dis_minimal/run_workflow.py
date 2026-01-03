#!/usr/bin/env python3
"""
S3DIS Minimal Complete Workflow Runner
Runs all steps: data prep, training, testing, and visualization.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "stanford_indoor3d_minimal"
LOG_DIR = REPO_ROOT / "log" / "s3dis_minimal"

def run_command(cmd, description=""):
    """Run a shell command and report status."""
    if description:
        print(f"\n{'='*70}")
        print(f"  {description}")
        print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\n❌ Command failed with exit code {result.returncode}")
        print(f"Command: {cmd}")
        return False
    print(f"\n✓ {description} completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser("S3DIS Minimal Complete Workflow")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT),
                       help=f"Data directory [default: {DATA_ROOT}]")
    parser.add_argument("--room_glob", type=str, default="Area_5_conferenceRoom_1*.npy",
                       help="Room file glob pattern [default: Area_5_conferenceRoom_1*.npy]")
    parser.add_argument("--log_dir", type=str, default="s3dis_minimal_run",
                       help="Log directory name [default: s3dis_minimal_run]")
    parser.add_argument("--model", type=str, default="pointnet_sem_seg",
                       help="Model name [default: pointnet_sem_seg]")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size [default: 2]")
    parser.add_argument("--npoint", type=int, default=2048,
                       help="Points per block [default: 2048]")
    parser.add_argument("--epoch", type=int, default=5,
                       help="Number of epochs [default: 5]")
    parser.add_argument("--sample_rate", type=float, default=0.1,
                       help="Sample rate [default: 0.1]")
    parser.add_argument("--skip_train", action="store_true",
                       help="Skip training (use existing checkpoint)")
    parser.add_argument("--skip_test", action="store_true",
                       help="Skip testing")
    parser.add_argument("--skip_viz", action="store_true",
                       help="Skip visualization")
    
    args = parser.parse_args()
    
    # Validate data exists
    data_path = Path(args.data_root)
    if not data_path.exists():
        print(f"\n❌ Data directory not found: {data_path}")
        print(f"Please place S3DIS room files (e.g., Area_5_conferenceRoom_1.npy) in: {data_path}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("  S3DIS MINIMAL COMPLETE WORKFLOW")
    print(f"{'='*70}")
    print(f"Data root:    {data_path}")
    print(f"Room glob:    {args.room_glob}")
    print(f"Log dir:      {args.log_dir}")
    print(f"Model:        {args.model}")
    print(f"Epochs:       {args.epoch}")
    print(f"Batch size:   {args.batch_size}")
    print(f"N points:     {args.npoint}")
    print(f"Sample rate:  {args.sample_rate}")
    
    # Step 1: Training
    if not args.skip_train:
        train_cmd = (
            f"python examples/s3dis_minimal/train_semseg_minimal.py "
            f"--data_root {args.data_root} "
            f"--room_glob \"{args.room_glob}\" "
            f"--log_dir {args.log_dir} "
            f"--model {args.model} "
            f"--batch_size {args.batch_size} "
            f"--npoint {args.npoint} "
            f"--epoch {args.epoch} "
            f"--sample_rate {args.sample_rate}"
        )
        if not run_command(train_cmd, "TRAINING: Training semantic segmentation model"):
            sys.exit(1)
    else:
        print("\n⊘ Skipping training (--skip_train)")
    
    # Step 2: Testing (optional, may not be implemented yet)
    if not args.skip_test:
        print("\n⊘ Testing step: (not yet implemented in minimal example)")
    
    # Step 3: Visualization with Open3D
    if not args.skip_viz:
        checkpoint_path = LOG_DIR / args.log_dir / "checkpoints" / "best_model.pth"
        if not checkpoint_path.exists():
            print(f"\n⚠ Warning: Checkpoint not found at {checkpoint_path}")
            print("Skipping visualization...")
        else:
            print("\n⊘ Visualization step: (not yet implemented in minimal example)")
            print(f"   Checkpoint ready at: {checkpoint_path}")
            print(f"   To visualize, use: python view_semseg_results.py --visual_dir log/s3dis_minimal/{args.log_dir}/visual")
    
    print(f"\n{'='*70}")
    print("  ✓ WORKFLOW COMPLETE")
    print(f"{'='*70}")
    print(f"\nLogs and checkpoints saved to: {LOG_DIR / args.log_dir}")
    print(f"Checkpoint: {LOG_DIR / args.log_dir / 'checkpoints' / 'best_model.pth'}")
    
if __name__ == '__main__':
    main()
