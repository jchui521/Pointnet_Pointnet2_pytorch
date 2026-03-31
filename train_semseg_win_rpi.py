import datetime
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

NUM_WORKERS = 4
BATCH_SIZE = 128
NUM_POINT = 8192
DECAY_RATE = 1e-4
LR_DECAY = 0.7
STEP_SIZE = 10
LR = 0.001
NUM_CLASSES = 19
EPOCHS = 1
SAMPLE_RATE = 1.0
# GPUS = "0,1,2,3,4,5,6,7"
GPUS = "0"
log_dir = None

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

def worker_init_fn(worker_id):
    np.random.seed(worker_id + int(time.time()))

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

def log_string(str):
    logger.info(str)
    print(str)

if __name__ == "__main__":
    """HYPER PARAMETER"""
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
    torch.backends.cudnn.benchmark = True

    """CREATE DIR"""
    timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    experiment_dir = Path("./log/")
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath("sem_seg_rpi")
    experiment_dir.mkdir(exist_ok=True)
    if log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath("checkpoints/")
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath("logs/")
    log_dir.mkdir(exist_ok=True)

    """LOG"""
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/%s.txt" % (log_dir, "pointnet2_sem_seg"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("PARAMETER ...")
    log_string(f"Batch Size: {BATCH_SIZE} | Learning Rate: {LR} | Learning Rate Decay: {LR_DECAY} | Step Size: {STEP_SIZE} | Epochs: {EPOCHS} | Sample Rate: {SAMPLE_RATE} \
                | Num Point: {NUM_POINT} | Num Classes: {NUM_CLASSES} | Gpus: {GPUS} | Log Dir: {log_dir}")
    
    """DATA LOADING"""
    root = "rpi_data"

    # print("start loading training data ...")
    # TRAIN_DATASET = PointNetDataset(
    #     split="train",
    #     data_root=root,
    #     num_point=NUM_POINT,
    #     num_classes=NUM_CLASSES,
    #     block_size=1.0,
    #     sample_rate=1.0,
    #     transform=None,
    # )
    print("start loading test data ...")
    TEST_DATASET = PointNetDataset(
        split="test",
        data_root=root,
        num_point=NUM_POINT,
        num_classes=NUM_CLASSES,
        block_size=1.0,
        sample_rate=1.0,
        transform=None,
    )

    # trainDataLoader = torch.utils.data.DataLoader(
    #     TRAIN_DATASET,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=NUM_WORKERS, 
    #     pin_memory=True,
    #     drop_last=True,
    #     persistent_workers=True if NUM_WORKERS > 0 else False,
    #     worker_init_fn=worker_init_fn,
    # )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    # weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    weights = torch.Tensor(TEST_DATASET.labelweights).cuda()

    # log_string(f"Number of training data is: {len(TRAIN_DATASET)}")
    log_string(f"Number of test data is {len(TEST_DATASET)}")
    
    classifier = get_model(NUM_CLASSES).cuda()
    criterion = get_loss().cuda()
    classifier.apply(inplace_relu)
    

    if torch.cuda.device_count() > 1:
        log_string(f"Using {torch.cuda.device_count()} GPUS")
        classifier = torch.nn.DataParallel(classifier)
    
    classifier.apply(weights_init)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=DECAY_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY)

    for epoch in range(EPOCHS):
        log_string(f"**** EPOCH {epoch + 1} / {EPOCHS} ****")
        
        # num_batches = len(trainDataLoader)
        num_batches = len(testDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        # """Training Step"""
        # for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader)):
        for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
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
            correct = np.sum(pred_choice  == batch_label)
            total_correct += correct
            total_seen += BATCH_SIZE * NUM_POINT
            loss_sum += loss
        log_string(f"Mean train loss: {loss_sum / num_batches}")
        log_string(f"Mean train accuracy: {total_correct / float(total_seen)}")
    
        # """Eval Step"""
        # with torch.no_grad():
        #     num_batches = len(testDataLoader)
        #     total_correct = 0
        #     total_seen = 0
        #     loss_sum = 0
        #     labelweights = np.zeros(NUM_CLASSES)
        #     total_seen_class = [0 for _ in range(NUM_CLASSES)]
        #     total_correct_class = [0 for _ in range(NUM_CLASSES)]
        #     total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        #     classifier = classifier.eval()
        #     for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
        #         with torch.no_grad():
        #             points, target = points.float().cuda(), target.long().cuda()
        #             points = points.transpose(2, 1)

        #             seg_pred, trans_feat = classifier(points)
        #             seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

        #             target = target.view(-1, 1)[:, 0]
        #             loss = criterion(seg_pred, target, trans_feat, weights)

        #             pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
        #             correct = np.sum(pred_choice == target.cpu().data.numpy())
        #             total_correct += correct
        #             total_seen += BATCH_SIZE * NUM_POINT
        #             loss_sum += loss
    
        # log_string(f"Mean test loss: {loss_sum / num_batches}")
        # log_string(f"Mean test accuracy: {total_correct / float(total_seen)}")

        scheduler.step()
        torch.cuda.empty_cache()

