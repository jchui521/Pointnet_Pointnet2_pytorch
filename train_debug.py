import argparse
import datetime
import importlib
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import provider
from data_utils.RPIDataLoader import PointNetDataset

from models.pointnet2_sem_seg import get_model, get_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

class_file = "classes.txt"
with open(class_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

NUM_WORKERS = 4
BATCH_SIZE = 32
NUM_POINT = 2048
DECAY_RATE = 1e-4
LR_DECAY = 0.7
STEP_SIZE = 10
LR = 0.001
NUM_CLASSES = 47
EPOCHS = 32

data_root = "rpi_data"

ds = PointNetDataset(
    split="test",
    data_root=data_root,
    num_point=NUM_POINT,
    num_classes=NUM_CLASSES,
    block_size=1.0,
    sample_rate=1.0,
    transform=None,
)

dataloader = torch.utils.data.DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    worker_init_fn=lambda id: np.random.seed(id),
)

weights = torch.Tensor(ds.labelweights).cuda()

classifier = get_model(
    num_classes=NUM_CLASSES,
)
criterion = get_loss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, betas=(0.9, 0.999))

for i in range(EPOCHS):
    loss_sum = 0.0
    for batch_id, (points, target) in enumerate(dataloader):
        optimizer.zero_grad()

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
        total_seen += BATCH_SIZE * NUM_POINT
        loss_sum += loss
    print(f"Loss: {loss_sum / len(dataloader)}")