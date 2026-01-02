"""
Twinner01 Training Script for Custom Semantic Segmentation
Adapted from train_semseg.py for custom datasets
"""

import argparse
import os
from twinner01_dataloader import Twinner01Dataset
from twinner01_classes_config import TWINNER01_CLASSES, NUM_CLASSES
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Twinner01 Semantic Segmentation Training')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', 
                       help='model name [pointnet_sem_seg, pointnet2_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size [default: 8]')
    parser.add_argument('--epoch', default=100, type=int, 
                       help='Epochs to train [default: 100]')
    parser.add_argument('--learning_rate', default=0.001, type=float, 
                       help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', 
                       help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', 
                       help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, 
                       help='Log directory name [default: timestamp]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, 
                       help='Weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, 
                       help='Point number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, 
                       help='LR decay step [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, 
                       help='LR decay rate [default: 0.7]')
    parser.add_argument('--data_root', type=str, default='data/twinner01_custom',
                       help='Data directory [default: data/twinner01_custom]')
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='Test split ratio [default: 0.2]')
    
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETERS'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIRECTORIES'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('twinner01_sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOGGING'''
    logger = logging.getLogger("Twinner01")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    log_string('='*60)
    log_string('TWINNER01 SEMANTIC SEGMENTATION TRAINING')
    log_string('='*60)
    log_string('PARAMETERS:')
    log_string(args)
    log_string(f'Classes: {TWINNER01_CLASSES}')
    log_string(f'Number of classes: {NUM_CLASSES}')

    '''DATASET'''
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    log_string("Loading training data...")
    TRAIN_DATASET = Twinner01Dataset(
        split='train', 
        data_root=args.data_root, 
        num_point=NUM_POINT, 
        test_split=args.test_split,
        block_size=1.0, 
        sample_rate=1.0, 
        transform=None,
        num_classes=NUM_CLASSES
    )
    
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

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True, 
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )
    
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True, 
        drop_last=True
    )
    
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string(f"Training samples: {len(TRAIN_DATASET)}")
    log_string(f"Test samples: {len(TEST_DATASET)}")
    log_string(f"Label weights: {weights.cpu().numpy()}")

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Loaded pretrained model')
    except:
        log_string('No existing model, starting from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    '''TRAINING'''
    log_string('='*60)
    log_string('STARTING TRAINING')
    log_string('='*60)
    
    for epoch in range(start_epoch, args.epoch):
        log_string(f'\nEpoch {epoch+1}/{args.epoch}:')

        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string(f'Learning rate: {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        log_string(f'BN momentum: {momentum}')
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''Training loop'''
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        
        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=num_batches, desc='Training', smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss.item()

        train_acc = total_correct / float(total_seen)
        log_string(f'Training mean loss: {loss_sum / num_batches:.4f}')
        log_string(f'Training accuracy: {train_acc:.4f}')

        '''Evaluation'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('Evaluating...')
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=num_batches, desc='Testing'):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss.item()
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))
            test_acc = total_correct / float(total_seen)
            
            log_string(f'Test mean loss: {loss_sum / num_batches:.4f}')
            log_string(f'Test accuracy: {test_acc:.4f}')
            log_string(f'Test mIoU: {mIoU:.4f}')
            
            # Per-class IoU
            iou_per_class = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
            for i in range(NUM_CLASSES):
                log_string(f'  {TWINNER01_CLASSES[i]}: IoU = {iou_per_class[i]:.4f}, Acc = {total_correct_class[i]/float(total_seen_class[i] + 1e-6):.4f}')

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info(f'Save model... Best mIoU: {best_iou:.4f}')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string(f'Saving at {savepath}')
                state = {
                    'epoch': epoch,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'mIoU': mIoU,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Model saved')

        global_epoch += 1

    log_string('='*60)
    log_string('TRAINING COMPLETE!')
    log_string(f'Best mIoU: {best_iou:.4f}')
    log_string('='*60)


if __name__ == '__main__':
    args = parse_args()
    main(args)
