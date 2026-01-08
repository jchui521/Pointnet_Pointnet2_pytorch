"""
Author: Benny
Date: Nov 2019
Windows-friendly test harness with NumPy 2.x fixes and optional single-scene testing.
"""
import argparse
import os
from pathlib import Path
import sys
import logging
import importlib
import numpy as np
import torch
from tqdm import tqdm

from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root (log directory name)')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 3]')
    parser.add_argument('--scene_file', type=str, default=None, help='Optional single scene .txt to evaluate (raw Stanford3dDataset file)')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def load_single_scene(scene_file):
    if scene_file.endswith('.npy'):
        data = np.load(scene_file)
    else:
        data = np.loadtxt(scene_file)
    assert data.shape[1] >= 7, "Scene file must have at least 7 columns (x y z r g b label)"
    scene_points = data[:, :6]
    scene_labels = data[:, 6].astype(int)
    scene_id = Path(scene_file).stem
    return scene_id, scene_points, scene_labels


def make_single_scene_dataset(scene_id, scene_points, scene_labels, block_points=4096, stride=0.5, block_size=1.0, padding=0.001):
    class SingleSceneDataset:
        def __init__(self):
            self.block_points = block_points
            self.block_size = block_size
            self.padding = padding
            self.stride = stride
            self.scene_points_list = [scene_points]
            self.semantic_labels_list = [scene_labels]
            self.raw_scene_points = [scene_points]
            self.file_list = [scene_id + '.npy']
            labelweights = np.zeros(13)
            tmp, _ = np.histogram(scene_labels, range(14))
            labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            points = self.scene_points_list[index][:, :6]
            labels = self.semantic_labels_list[index]
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
            grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
            data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]), np.array([])
            for index_y in range(0, grid_y):
                for index_x in range(0, grid_x):
                    s_x = coord_min[0] + index_x * self.stride
                    e_x = min(s_x + self.block_size, coord_max[0])
                    s_x = e_x - self.block_size
                    s_y = coord_min[1] + index_y * self.stride
                    e_y = min(s_y + self.block_size, coord_max[1])
                    s_y = e_y - self.block_size
                    point_idxs = np.where(
                        (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) &
                        (points[:, 1] >= s_y - self.padding) & (points[:, 1] <= e_y + self.padding))[0]
                    if point_idxs.size == 0:
                        continue
                    num_batch = int(np.ceil(point_idxs.size / self.block_points))
                    point_size = int(num_batch * self.block_points)
                    replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                    point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                    point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                    np.random.shuffle(point_idxs)
                    data_batch = points[point_idxs, :]
                    normlized_xyz = np.zeros((point_size, 3))
                    normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                    normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                    normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                    data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                    data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                    data_batch[:, 3:6] /= 255.0
                    data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                    label_batch = labels[point_idxs].astype(int)
                    batch_weight = np.zeros_like(label_batch, dtype=np.float32) + 1.0

                    data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                    label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                    sample_weight = np.hstack([sample_weight, batch_weight]) if sample_weight.size else batch_weight
                    index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
            data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
            label_room = label_room.reshape((-1, self.block_points))
            sample_weight = sample_weight.reshape((-1, self.block_points))
            index_room = index_room.reshape((-1, self.block_points))
            return data_room, label_room, sample_weight, index_room

    return SingleSceneDataset()


def main(args):
    def log_string(str_msg):
        logger.info(str_msg)
        print(str_msg)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = Path(experiment_dir).joinpath('visual')
    visual_dir.mkdir(exist_ok=True, parents=True)

    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 13
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    # Dataset selection: single scene or full split
    if args.scene_file:
        scene_id, scene_points, scene_labels = load_single_scene(args.scene_file)
        TEST_DATASET_WHOLE_SCENE = make_single_scene_dataset(scene_id, scene_points, scene_labels, block_points=NUM_POINT)
        log_string(f"Single scene evaluation: {scene_id}")
    else:
        root = 'data/stanford_indoor3d/'
        TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
        log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_id_list = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id_list = [x[:-4] for x in scene_id_list]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            log_string(f"Inference [{batch_idx + 1}/{num_batches}] {scene_id_list[batch_idx]} ...")
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id_list[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id_list[batch_idx] + '_gt.obj'), 'w')

            scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            scene_label_flat = scene_label.reshape(-1)
            vote_label_pool = np.zeros((scene_label_flat.shape[0], NUM_CLASSES))
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
            batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

            for sbatch in range(s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                batch_data[:, :, 3:6] /= 1.0

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().cuda()
                torch_data = torch_data.transpose(2, 1)
                seg_pred, _ = classifier(torch_data)
                batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                           batch_pred_label[0:real_batch_size, ...],
                                           batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)

            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((scene_label_flat == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (scene_label_flat == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (scene_label_flat == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float64) + 1e-6)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id_list[batch_idx], tmp_iou))
            log_string(str(iou_map))
            log_string('----------------------------')

            filename = os.path.join(visual_dir, scene_id_list[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')

            if args.visual:
                raw_points_list = getattr(TEST_DATASET_WHOLE_SCENE, 'raw_scene_points', None)
                scene_points_vis = raw_points_list[batch_idx] if raw_points_list is not None else TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx][:, :3]
                for i in range(scene_label_flat.shape[0]):
                    color = g_label2color[pred_label[i]]
                    color_gt = g_label2color[scene_label_flat[i]]
                    fout.write('v %f %f %f %d %d %d\n' % (
                        scene_points_vis[i, 0], scene_points_vis[i, 1], scene_points_vis[i, 2], color[0], color[1], color[2]))
                    fout_gt.write('v %f %f %f %d %d %d\n' % (
                        scene_points_vis[i, 0], scene_points_vis[i, 1], scene_points_vis[i, 2], color_gt[0], color_gt[1], color_gt[2]))
                fout.close()
                fout_gt.close()

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
