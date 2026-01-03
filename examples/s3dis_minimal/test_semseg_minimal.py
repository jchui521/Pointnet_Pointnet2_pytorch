#!/usr/bin/env python3
"""
Minimal S3DIS Semantic Segmentation Testing Script
Loads a trained model and generates predictions for visualization.
"""

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

import provider
from pathlib import Path

# S3DIS classes (13 classes)
CLASSES = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 
           'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']


def parse_args():
    parser = argparse.ArgumentParser('S3DIS Minimal Testing')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Path to processed room data')
    parser.add_argument('--room_file', type=str, required=True, help='Room file to test (e.g., Area_5_conferenceRoom_1.npy)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for predictions')
    parser.add_argument('--npoint', type=int, default=4096, help='Points per block')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='Model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    return parser.parse_args()


def load_model(checkpoint_path, model_name, num_classes=13):
    """Load trained model from checkpoint."""
    # Add models directory to path
    import sys
    models_dir = os.path.join(ROOT_DIR, 'models')
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    
    if model_name == 'pointnet_sem_seg':
        from models.pointnet_sem_seg import get_model
    elif model_name == 'pointnet2_sem_seg':
        from models.pointnet2_sem_seg import get_model
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = get_model(num_classes)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def predict_room(model, room_data, npoint=4096, batch_size=8, device='cuda'):
    """
    Predict semantic labels for entire room.
    
    Args:
        model: Trained model
        room_data: [N, 9] array (xyz, rgb, normalized xyz)
        npoint: Points per block
        batch_size: Batch size for inference
    
    Returns:
        predictions: [N] array of predicted labels
        confidences: [N, num_classes] array of prediction probabilities
    """
    model.eval()
    
    N = room_data.shape[0]
    predictions = np.zeros(N, dtype=np.int32)
    confidences = np.zeros((N, 13), dtype=np.float32)
    vote_counts = np.zeros(N, dtype=np.int32)
    
    # Sample blocks from the room
    coords = room_data[:, :3]
    
    # Find coordinate bounds
    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)
    
    # Create sliding window blocks
    stride = 0.5  # 50% overlap
    block_size = 1.0  # 1 meter blocks
    
    blocks = []
    block_indices = []
    
    # Create grid of blocks
    x_min, y_min = coord_min[0], coord_min[1]
    x_max, y_max = coord_max[0], coord_max[1]
    
    x_steps = int(np.ceil((x_max - x_min) / (block_size * stride))) + 1
    y_steps = int(np.ceil((y_max - y_min) / (block_size * stride))) + 1
    
    print(f"Room bounds: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
    print(f"Creating {x_steps}x{y_steps}={x_steps*y_steps} blocks...")
    
    for x_idx in range(x_steps):
        for y_idx in range(y_steps):
            x_center = x_min + x_idx * block_size * stride + block_size / 2
            y_center = y_min + y_idx * block_size * stride + block_size / 2
            
            # Find points in this block
            x_dist = np.abs(coords[:, 0] - x_center)
            y_dist = np.abs(coords[:, 1] - y_center)
            in_block = (x_dist <= block_size / 2) & (y_dist <= block_size / 2)
            
            point_idxs = np.where(in_block)[0]
            
            if len(point_idxs) < 100:  # Skip blocks with too few points
                continue
            
            # Sample or pad to npoint
            if len(point_idxs) >= npoint:
                choice = np.random.choice(len(point_idxs), npoint, replace=False)
            else:
                choice = np.random.choice(len(point_idxs), npoint, replace=True)
            
            selected_idxs = point_idxs[choice]
            selected_data = room_data[selected_idxs, :9].copy()  # Take xyz + rgb + normalized_xyz (9 channels)
            
            # CENTER XY COORDINATES (same as training!)
            selected_data[:, 0] -= x_center  # Center X
            selected_data[:, 1] -= y_center  # Center Y
            # Note: Z is NOT centered in training either
            
            blocks.append(selected_data)
            block_indices.append(selected_idxs)
    
    print(f"Created {len(blocks)} valid blocks")
    
    if len(blocks) == 0:
        print("Warning: No valid blocks created, using full room")
        # Fall back to sampling from full room
        if N >= npoint:
            choice = np.random.choice(N, npoint, replace=False)
        else:
            choice = np.random.choice(N, npoint, replace=True)
        
        selected_data = room_data[choice, :9].copy()  # Take xyz + rgb + normalized_xyz (9 channels)
        # Center around room center
        room_center = coords.mean(axis=0)
        selected_data[:, 0] -= room_center[0]
        selected_data[:, 1] -= room_center[1]
        
        blocks.append(selected_data)
        block_indices.append(choice)
    
    # Process blocks in batches
    num_blocks = len(blocks)
    for batch_start in tqdm(range(0, num_blocks, batch_size), desc="Predicting"):
        batch_end = min(batch_start + batch_size, num_blocks)
        batch_blocks = blocks[batch_start:batch_end]
        batch_idxs = block_indices[batch_start:batch_end]
        
        # Stack blocks into batch
        batch_data = np.stack(batch_blocks, axis=0)  # [B, npoint, 9]
        
        # Convert to tensor - model expects [B, 9, npoint] (xyz + rgb + normalized_xyz)
        points = torch.from_numpy(batch_data).float().transpose(2, 1)  # [B, 9, npoint]
        points = points.to(device)
        
        with torch.no_grad():
            pred, _ = model(points)  # Model outputs [B, npoint, num_classes] already!
            pred_probs = torch.softmax(pred, dim=-1)  # [B, npoint, num_classes]
            pred_labels = pred_probs.argmax(dim=-1)  # [B, npoint]
        
        # Accumulate predictions
        pred_labels = pred_labels.cpu().numpy()
        pred_probs = pred_probs.cpu().numpy()
        
        for i in range(len(batch_blocks)):
            idxs = batch_idxs[i]  # [npoint] array of indices
            labels = pred_labels[i]  # [npoint]
            probs = pred_probs[i]  # [npoint, num_classes]
            
            # Vote-based accumulation
            for j in range(len(idxs)):
                idx = int(idxs[j])  # Convert to Python int
                confidences[idx] += probs[j]
                vote_counts[idx] += 1
    
    # Average confidences and get final predictions
    mask = vote_counts > 0
    confidences[mask] /= vote_counts[mask, np.newaxis]
    predictions = confidences.argmax(axis=1)
    
    return predictions, confidences


def save_predictions(room_file, room_data, predictions, output_dir):
    """Save predictions alongside original data for visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import label colors
    data_utils_dir = Path(__file__).resolve().parents[2] / 'data_utils'
    sys.path.insert(0, str(data_utils_dir))
    from indoor3d_util import g_label2color
    
    # Extract room name
    room_name = Path(room_file).stem
    xyz = room_data[:, :3]  # Original coordinates
    rgb = room_data[:, 3:6]  # Original RGB
    
    # Save original data (xyz + rgb)
    original_path = output_dir / f"{room_name}_original.npy"
    np.save(original_path, room_data[:, :6])  # xyz + rgb
    print(f"Saved original data: {original_path}")
    
    # Save predictions
    pred_path = output_dir / f"{room_name}_pred.npy"
    np.save(pred_path, predictions)
    print(f"Saved predictions: {pred_path}")
    
    # Save predictions as .obj for visualization
    pred_obj_path = output_dir / f"{room_name}_pred.obj"
    with open(pred_obj_path, 'w') as f:
        for i in range(len(xyz)):
            label = int(predictions[i])
            color = g_label2color.get(label, [128, 128, 128])  # Default to gray
            # Normalize color to 0-1 range
            r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
            f.write(f"v {xyz[i,0]} {xyz[i,1]} {xyz[i,2]} {r} {g} {b}\n")
    print(f"Saved prediction .obj: {pred_obj_path}")
    
    # Save ground truth labels if available
    if room_data.shape[1] >= 10:  # Has label column
        gt_labels = room_data[:, -1].astype(np.int32)
        gt_path = output_dir / f"{room_name}_gt.npy"
        np.save(gt_path, gt_labels)
        print(f"Saved ground truth: {gt_path}")
        
        # Save ground truth as .obj for visualization
        gt_obj_path = output_dir / f"{room_name}_gt.obj"
        with open(gt_obj_path, 'w') as f:
            for i in range(len(xyz)):
                label = int(gt_labels[i])
                color = g_label2color.get(label, [128, 128, 128])
                # Normalize color to 0-1 range
                r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
                f.write(f"v {xyz[i,0]} {xyz[i,1]} {xyz[i,2]} {r} {g} {b}\n")
        print(f"Saved ground truth .obj: {gt_obj_path}")
        
        # Compute accuracy
        accuracy = (predictions == gt_labels).mean()
        print(f"\nAccuracy: {accuracy * 100:.2f}%")
        
        # Per-class accuracy
        print("\nPer-class accuracy:")
        for i, cls_name in enumerate(CLASSES):
            mask = gt_labels == i
            if mask.sum() > 0:
                cls_acc = (predictions[mask] == gt_labels[mask]).mean()
                print(f"  {cls_name:12s}: {cls_acc * 100:5.2f}% ({mask.sum():6d} points)")


def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load room data
    room_path = Path(args.data_root) / args.room_file
    if not room_path.exists():
        print(f"Error: Room file not found: {room_path}")
        sys.exit(1)
    
    print(f"\nLoading room data: {room_path}")
    room_data = np.load(room_path).astype(np.float32)
    print(f"Room shape: {room_data.shape}")
    
    # Check if we need to add normalized coordinates
    if room_data.shape[1] == 7:  # xyz(3) + rgb(3) + label(1)
        print("Adding normalized coordinates...")
        xyz = room_data[:, :3]
        rgb = room_data[:, 3:6]
        label = room_data[:, 6:7]
        
        # Normalize RGB to [0, 1] (same as training!)
        rgb = rgb / 255.0
        
        # Normalize coordinates to [0, 1] range
        coord_min = xyz.min(axis=0, keepdims=True)
        coord_max = xyz.max(axis=0, keepdims=True)
        xyz_normalized = (xyz - coord_min) / (coord_max - coord_min + 1e-8)
        
        # Reconstruct data: xyz + rgb + normalized_xyz + label
        room_data = np.concatenate([xyz, rgb, xyz_normalized, label], axis=1)
        print(f"New shape: {room_data.shape}")
    
    print(f"Columns: xyz(3) + rgb_normalized(3) + normalized_xyz(3) + label(1)")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, args.model)
    model = model.to(device)
    print(f"Model loaded: {args.model}")
    
    # Predict
    print(f"\nPredicting semantic labels...")
    predictions, confidences = predict_room(
        model, room_data, 
        npoint=args.npoint, 
        batch_size=args.batch_size,
        device=device
    )
    
    # Set output directory
    if args.output_dir is None:
        checkpoint_dir = Path(args.checkpoint).parent.parent
        args.output_dir = checkpoint_dir / "visual"
    
    # Save results
    print(f"\nSaving results to: {args.output_dir}")
    save_predictions(args.room_file, room_data, predictions, args.output_dir)
    
    print("\n✓ Testing complete!")


if __name__ == '__main__':
    main()
