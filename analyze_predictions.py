#!/usr/bin/env python3
"""
Analyze semantic segmentation predictions
Shows what classes were predicted and their distribution
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add data_utils to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))

from indoor3d_util import g_label2color

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']


def analyze_predictions(visual_dir, scene_name):
    """Analyze prediction results"""
    
    # Load ground truth and predictions
    gt_file = os.path.join(visual_dir, f"{scene_name}_gt.npy")
    pred_file = os.path.join(visual_dir, f"{scene_name}_pred.npy")
    
    if not os.path.exists(gt_file):
        print(f"Error: Ground truth file not found: {gt_file}")
        return
    
    if not os.path.exists(pred_file):
        print(f"Error: Prediction file not found: {pred_file}")
        return
    
    gt_labels = np.load(gt_file)
    pred_labels = np.load(pred_file)
    
    print(f"\n{'='*80}")
    print(f"Semantic Segmentation Analysis: {scene_name}")
    print(f"{'='*80}")
    
    print(f"\nTotal points: {len(gt_labels):,}")
    
    # Overall accuracy
    correct = (gt_labels == pred_labels).sum()
    accuracy = correct / len(gt_labels)
    print(f"Overall accuracy: {accuracy*100:.2f}% ({correct:,}/{len(gt_labels):,} points correct)")
    
    # Ground truth distribution
    print(f"\n{'='*80}")
    print("GROUND TRUTH DISTRIBUTION (What's actually in the scene)")
    print(f"{'='*80}")
    print(f"{'Class':<15} {'Points':<12} {'% of Scene':<12}")
    print(f"{'-'*80}")
    
    for i, cls_name in enumerate(classes):
        count = (gt_labels == i).sum()
        percentage = (count / len(gt_labels)) * 100
        if count > 0:
            print(f"{cls_name:<15} {count:<12,} {percentage:>6.2f}%")
    
    # Prediction distribution
    print(f"\n{'='*80}")
    print("PREDICTION DISTRIBUTION (What the model predicted)")
    print(f"{'='*80}")
    print(f"{'Class':<15} {'Points':<12} {'% of Predictions':<20} {'Found':<10}")
    print(f"{'-'*80}")
    
    for i, cls_name in enumerate(classes):
        count = (pred_labels == i).sum()
        percentage = (count / len(pred_labels)) * 100
        gt_count = (gt_labels == i).sum()
        
        if gt_count > 0:
            found = "YES" if count > 0 else "NO"
            print(f"{cls_name:<15} {count:<12,} {percentage:>6.2f}% {' '*13} {found:<10}")
    
    # Per-class performance
    print(f"\n{'='*80}")
    print("PER-CLASS PERFORMANCE")
    print(f"{'='*80}")
    print(f"{'Class':<15} {'GT Points':<12} {'Predicted':<12} {'Correct':<10} {'Accuracy':<10} {'IoU':<10}")
    print(f"{'-'*80}")
    
    for i, cls_name in enumerate(classes):
        gt_mask = (gt_labels == i)
        pred_mask = (pred_labels == i)
        gt_count = gt_mask.sum()
        pred_count = pred_mask.sum()
        
        if gt_count > 0:
            # Correct predictions for this class
            correct_class = ((gt_labels == i) & (pred_labels == i)).sum()
            class_acc = correct_class / gt_count
            
            # IoU (Intersection over Union)
            intersection = correct_class
            union = ((gt_labels == i) | (pred_labels == i)).sum()
            iou = intersection / union if union > 0 else 0
            
            print(f"{cls_name:<15} {gt_count:<12,} {pred_count:<12,} {correct_class:<10,} {class_acc*100:>6.2f}% {' '*3} {iou:>6.3f}")
    
    # Confusion analysis
    print(f"\n{'='*80}")
    print("COMMON MISCLASSIFICATIONS (Top 5)")
    print(f"{'='*80}")
    print(f"{'GT Class':<15} {'Predicted As':<15} {'Count':<12} {'% of GT':<10}")
    print(f"{'-'*80}")
    
    misclassifications = []
    for gt_cls in range(len(classes)):
        gt_mask = (gt_labels == gt_cls)
        if gt_mask.sum() == 0:
            continue
        
        for pred_cls in range(len(classes)):
            if gt_cls == pred_cls:
                continue
            
            misclass_count = ((gt_labels == gt_cls) & (pred_labels == pred_cls)).sum()
            if misclass_count > 0:
                percentage = (misclass_count / gt_mask.sum()) * 100
                misclassifications.append((gt_cls, pred_cls, misclass_count, percentage))
    
    # Sort by count
    misclassifications.sort(key=lambda x: x[2], reverse=True)
    
    for gt_cls, pred_cls, count, percentage in misclassifications[:10]:
        print(f"{classes[gt_cls]:<15} {classes[pred_cls]:<15} {count:<12,} {percentage:>6.2f}%")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    # Count how many classes were predicted
    predicted_classes = sum(1 for i in range(len(classes)) if (pred_labels == i).sum() > 0)
    actual_classes = sum(1 for i in range(len(classes)) if (gt_labels == i).sum() > 0)
    
    print(f"Classes in scene (GT): {actual_classes}/13")
    print(f"Classes predicted: {predicted_classes}/13")
    print(f"Classes missed: {actual_classes - predicted_classes}")
    
    # Most predicted class
    pred_class_counts = [(i, (pred_labels == i).sum()) for i in range(len(classes))]
    pred_class_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMost predicted class: {classes[pred_class_counts[0][0]]} ({pred_class_counts[0][1]:,} points, {pred_class_counts[0][1]/len(pred_labels)*100:.1f}%)")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Prediction Analysis')
    parser.add_argument('--visual_dir', type=str, required=True)
    parser.add_argument('--scene', type=str, required=True)
    args = parser.parse_args()
    
    analyze_predictions(args.visual_dir, args.scene)
