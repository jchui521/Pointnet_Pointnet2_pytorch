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

from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, "models"))

    # class_file = "classes.txt"
    # with open(class_file, "r") as f:
    #     classes = [line.strip() for line in f.readlines()]
    # class2label = {cls: i for i, cls in enumerate(classes)}
    # seg_classes = class2label
    # seg_label_to_cat = {}
    # for i, cat in enumerate(seg_classes.keys()):
    #     seg_label_to_cat[i] = cat

    NUM_WORKERS = 0
    BATCH_SIZE = 16
    NUM_POINT = 2048
    DECAY_RATE = 1e-4
    LR_DECAY = 0.7
    STEP_SIZE = 10
    LR = 0.001
    NUM_CLASSES = 47
    EPOCHS = 1
    SAMPLE_RATE = 1.0

    def inplace_relu(m):
        classname = m.__class__.__name__
        if classname.find("ReLU") != -1:
            m.inplace = True

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    data_root = "rpi_data"

    train_ds = PointNetDataset(
        split="train",
        data_root=data_root,
        num_point=NUM_POINT,
        num_classes=NUM_CLASSES,
        block_size=1.0,
        sample_rate=SAMPLE_RATE,
        transform=None,
    )

    test_ds = PointNetDataset(
        split="test",
        data_root=data_root,
        num_point=NUM_POINT,
        num_classes=NUM_CLASSES,
        block_size=1.0,
        sample_rate=SAMPLE_RATE,
        transform=None,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        pin_memory=True,
        worker_init_fn=lambda id: np.random.seed(id) if NUM_WORKERS > 0 else None,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        pin_memory=True,
        worker_init_fn=lambda id: np.random.seed(id) if NUM_WORKERS > 0 else None,
    )

    weights = torch.Tensor(train_ds.labelweights).cuda()

    classifier = get_model(
        num_classes=NUM_CLASSES,
    ).cuda()
    classifier.apply(inplace_relu)
    classifier.apply(weights_init)

    criterion = get_loss()
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, betas=(0.9, 0.999))

    print("starting training ... ")
    for i in range(EPOCHS):
        # loss_sum = 0.0
        for batch_id, (points, target) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # optimizer.zero_grad()

            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)       

            # loss.backward()
            # optimizer.step()

            # loss_sum += loss
        # print(f"Loss: {loss_sum / len(dataloader)}")
    
        for batch_id, (points, target) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            with torch.no_grad():
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)