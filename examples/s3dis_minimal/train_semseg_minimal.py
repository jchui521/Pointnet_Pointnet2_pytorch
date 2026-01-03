"""
Minimal S3DIS smoke test on a tiny subset (e.g., Area_5 conferenceRoom_1).
Place one or a few preprocessed room .npy files under data/stanford_indoor3d_minimal/.
"""

import argparse
import os
import sys
import datetime
import logging
import shutil
import importlib
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / 'models'))
os.chdir(REPO_ROOT)

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
NUM_CLASSES = len(classes)


class MiniS3DISDataset(Dataset):
    """Tiny variant of S3DISDataset that loads a handful of room .npy files and splits them train/test."""

    def __init__(self, split='train', data_root='data/stanford_indoor3d_minimal', room_glob='Area_5_conferenceRoom_1*.npy',
                 num_point=2048, train_fraction=0.7, block_size=1.0, sample_rate=0.1, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform

        data_root = Path(data_root)
        files = sorted(data_root.glob(room_glob))
        if not files:
            raise ValueError(f"No room files matched {room_glob} under {data_root}")

        split_idx = max(1, int(len(files) * train_fraction))
        if split == 'train':
            use_files = files[:split_idx]
        else:
            use_files = files[split_idx:]
        if not use_files:
            use_files = files[-1:]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(NUM_CLASSES)

        for room_file in tqdm(use_files, desc=f"Loading {split} data"):
            room_data = np.load(room_file)
            if room_data.shape[1] != 7:
                raise ValueError(f"Expected shape (N,7) in {room_file}, got {room_data.shape}")
            points, labels = room_data[:, 0:6], room_data[:, 6]
            tmp, _ = np.histogram(labels, range(NUM_CLASSES + 1))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points)
            self.room_labels.append(labels)
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(use_files)):
            room_idxs.extend([index] * max(1, int(round(sample_prob[index] * max(1, num_iter)))))
        self.room_idxs = np.array(room_idxs)

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx % len(self.room_idxs)]
        points = self.room_points[room_idx]
        labels = self.room_labels[room_idx]
        N_points = points.shape[0]

        while True:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        selected_points = points[selected_point_idxs, :]
        current_points = np.zeros((self.num_point, 9))
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


def parse_args():
    parser = argparse.ArgumentParser('Minimal S3DIS train/test')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [pointnet_sem_seg|pointnet2_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epoch', default=10, type=int, help='Epochs to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='Log dir name')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='Points per block')
    parser.add_argument('--step_size', type=int, default=10, help='LR decay step')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='LR decay rate')
    parser.add_argument('--data_root', type=str, default='data/stanford_indoor3d_minimal', help='Path to minimal S3DIS npy files')
    parser.add_argument('--room_glob', type=str, default='Area_5_conferenceRoom_1*.npy', help='Glob to pick minimal rooms')
    parser.add_argument('--train_fraction', type=float, default=0.7, help='Fraction of files for train split')
    parser.add_argument('--sample_rate', type=float, default=0.1, help='Sampling rate vs total points for iteration budget')
    return parser.parse_args()


def main(args):
    def log_string(msg):
        logger.info(msg)
        print(msg)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = REPO_ROOT.joinpath('log', 's3dis_minimal')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.log_dir or timestr)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    args = parse_args()
    logger = logging.getLogger('MiniS3DIS')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir / f"{args.model}.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETERS ...')
    log_string(args)

    log_string('Loading minimal dataset ...')
    TRAIN_DATASET = MiniS3DISDataset(split='train', data_root=args.data_root, room_glob=args.room_glob,
                                     num_point=args.npoint, train_fraction=args.train_fraction, sample_rate=args.sample_rate)
    TEST_DATASET = MiniS3DISDataset(split='test', data_root=args.data_root, room_glob=args.room_glob,
                                    num_point=args.npoint, train_fraction=args.train_fraction, sample_rate=args.sample_rate)

    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    MODEL = importlib.import_module(args.model)
    shutil.copy(str(REPO_ROOT / f'models/{args.model}.py'), str(experiment_dir))
    shutil.copy(str(REPO_ROOT / 'models/pointnet2_utils.py'), str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir / 'checkpoints/best_model.pth'), weights_only=False)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Loaded pretrained model')
    except:
        start_epoch = 0
        classifier = classifier.apply(weights_init)
        log_string('No checkpoint found, starting fresh')

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0

    for epoch in range(start_epoch, args.epoch):
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        momentum = max(momentum, 0.01)
        classifier.train()

        mean_correct = []
        for points, target in train_loader:
            optimizer.zero_grad()
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0] * points.size()[2]))
        train_acc = np.mean(mean_correct)
        log_string(f'Epoch {epoch+1}/{args.epoch} train acc: {train_acc:.4f}, lr: {lr:.6f}')

        classifier.eval()
        total_correct = 0
        total_seen = 0
        with torch.no_grad():
            for points, target in test_loader:
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred, trans_feat = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                target = target.view(-1, 1)[:, 0]
                _ = criterion(seg_pred, target, trans_feat, weights)
                pred_choice = seg_pred.data.max(1)[1]
                correct = pred_choice.eq(target.long().data).cpu().sum()
                total_correct += correct.item()
                total_seen += points.size()[0] * points.size()[2]
        eval_acc = total_correct / float(total_seen)
        log_string(f'Epoch {epoch+1}/{args.epoch} eval acc: {eval_acc:.4f}')

        if eval_acc >= best_acc:
            best_acc = eval_acc
            savepath = experiment_dir / 'checkpoints/best_model.pth'
            state = {
                'epoch': epoch,
                'instance_acc': eval_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string(f'Saved best model at {savepath}')


if __name__ == '__main__':
    args = parse_args()
    main(args)