"""
Twinner01 Testing Script for Custom Semantic Segmentation
Evaluates trained model on test set
"""

import argparse
import os
from twinner01_dataloader import Twinner01Dataset
from twinner01_classes_config import TWINNER01_CLASSES, NUM_CLASSES
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    parser = argparse.ArgumentParser('Twinner01 Semantic Segmentation Testing')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg',
                       help='model name [pointnet_sem_seg, pointnet2_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size [default: 8]')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Log directory (where trained model is saved)')
    parser.add_argument('--npoint', type=int, default=4096,
                       help='Point number [default: 4096]')
    parser.add_argument('--data_root', type=str, default='data/twinner01_custom',
                       help='Data directory [default: data/twinner01_custom]')
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='Test split ratio [default: 0.2]')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                       help='Checkpoint filename [default: best_model.pth]')
    
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETERS'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = Path('./log/twinner01_sem_seg/') / args.log_dir

    '''LOGGING'''
    logger = logging.getLogger("Twinner01 Test")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(experiment_dir) + '/test_log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    log_string('='*60)
    log_string('TWINNER01 SEMANTIC SEGMENTATION TESTING')
    log_string('='*60)
    log_string('PARAMETERS:')
    log_string(args)
    log_string(f'Classes: {TWINNER01_CLASSES}')
    log_string(f'Number of classes: {NUM_CLASSES}')

    '''DATASET'''
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    log_string("Loading test data...")
    TEST_DATASET = Twinner01Dataset(
        split='test',
        data_root=args.data_root,
        num_point=NUM_POINT,
        test_split=args.test_split,
        block_size=1.0,
        sample_rate=1.0,
        transform=None,
        num_classes=NUM_CLASSES
    )

    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    weights = torch.Tensor(TEST_DATASET.labelweights).cuda()
    log_string(f"Test samples: {len(TEST_DATASET)}")

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    
    checkpoint_path = str(experiment_dir) + f'/checkpoints/{args.checkpoint}'
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    log_string(f'Loaded model from {checkpoint_path}')
    log_string(f'Model was trained for {checkpoint["epoch"]} epochs')
    log_string(f'Training accuracy: {checkpoint.get("train_acc", "N/A")}')
    log_string(f'Previous test mIoU: {checkpoint.get("mIoU", "N/A")}')

    '''TESTING'''
    with torch.no_grad():
        num_batches = len(testDataLoader)
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        
        log_string('='*60)
        log_string('Starting evaluation...')
        log_string('='*60)
        
        classifier = classifier.eval()

        for i, (points, target) in tqdm(enumerate(testDataLoader), total=num_batches, desc='Testing'):
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            batch_label = target.cpu().data.numpy()
            
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == batch_label)
            total_correct += correct
            total_seen += (batch_label.shape[0] * batch_label.shape[1])

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

        # Calculate metrics
        test_acc = total_correct / float(total_seen)
        
        iou_per_class = []
        for i in range(NUM_CLASSES):
            iou = total_correct_class[i] / float(total_iou_deno_class[i] + 1e-6)
            iou_per_class.append(iou)
        
        mIoU = np.mean(iou_per_class)
        
        log_string('='*60)
        log_string('TEST RESULTS')
        log_string('='*60)
        log_string(f'Overall Accuracy: {test_acc:.4f}')
        log_string(f'Mean IoU: {mIoU:.4f}')
        log_string('')
        log_string('Per-Class Results:')
        log_string('-'*60)
        
        for i in range(NUM_CLASSES):
            class_acc = total_correct_class[i] / float(total_seen_class[i] + 1e-6)
            log_string(f'{TWINNER01_CLASSES[i]:20s} | IoU: {iou_per_class[i]:.4f} | Acc: {class_acc:.4f} | Points: {total_seen_class[i]}')
        
        log_string('='*60)
        log_string('TESTING COMPLETE!')
        log_string('='*60)
        
        # Save results to file
        results_file = str(experiment_dir) + '/test_results.txt'
        with open(results_file, 'w') as f:
            f.write('Twinner01 Semantic Segmentation Test Results\n')
            f.write('='*60 + '\n')
            f.write(f'Overall Accuracy: {test_acc:.4f}\n')
            f.write(f'Mean IoU: {mIoU:.4f}\n\n')
            f.write('Per-Class Results:\n')
            f.write('-'*60 + '\n')
            for i in range(NUM_CLASSES):
                class_acc = total_correct_class[i] / float(total_seen_class[i] + 1e-6)
                f.write(f'{TWINNER01_CLASSES[i]:20s} | IoU: {iou_per_class[i]:.4f} | Acc: {class_acc:.4f}\n')
        
        log_string(f'Results saved to: {results_file}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
