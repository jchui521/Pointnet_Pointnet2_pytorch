import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from torch.utils.data import DataLoader

# import provider

class PointNetDataset(Dataset):
    def __init__(
        self,
        split=None,
        data_root="data/rpi_data",
        num_point=4096,
        num_classes=13,
        block_size=1.0,
        sample_rate=1.0,
        transform=None,
    ):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.split = split
        rooms = sorted(os.listdir(data_root))
        if split == "train":
            rooms_split = [
                room for room in rooms if "train" in room
            ]
        elif split == "test":
            rooms_split = [
                room for room in rooms if "test" in room
            ]
        else:
            rooms_split = rooms
        
        print(f"Loading data from {data_root} for {split} set")
        print(f"Params: num_point={num_point}, block_size={block_size}, sample_rate={sample_rate}")

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        self.room_x_sort = []       # indices that sort each room by x-coordinate
        self.room_x_sorted = []     # pre-sorted x values (avoids O(N) recompute per __getitem__)
        num_point_all = []
        labelweights = np.zeros(num_classes)

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(num_classes + 1))
            labelweights += tmp
            coord_min, coord_max = (
                np.amin(points, axis=0)[:3],
                np.amax(points, axis=0)[:3],
            )
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            x_sort = np.argsort(points[:, 0])
            self.room_x_sort.append(x_sort)
            self.room_x_sorted.append(points[x_sort, 0])
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.where(
            labelweights == 0,
            0.0,
            np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        )
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)

        self.valid_centers = []
        grid_res = 0.1  # 10cm grid resolution for prefix-sum queries
        for room_idx in tqdm(range(len(rooms_split))):
            points = self.room_points[room_idx]
            N = points.shape[0]
            target = max(1, int(round(sample_prob[room_idx] * num_iter)))

            # Build 2D prefix-sum grid so block-count queries are O(1) instead of O(N)
            x_min, y_min = points[:, 0].min(), points[:, 1].min()
            xi = ((points[:, 0] - x_min) / grid_res).astype(np.int32)
            yi = ((points[:, 1] - y_min) / grid_res).astype(np.int32)
            x_bins, y_bins = xi.max() + 1, yi.max() + 1
            grid = np.zeros((x_bins, y_bins), dtype=np.int32)
            np.add.at(grid, (xi, yi), 1)
            prefix = grid.cumsum(axis=0).cumsum(axis=1)

            half = self.block_size / 2.0

            def block_count(cx, cy):
                ix0 = max(0, int((cx - half - x_min) / grid_res))
                ix1 = min(x_bins - 1, int((cx + half - x_min) / grid_res))
                iy0 = max(0, int((cy - half - y_min) / grid_res))
                iy1 = min(y_bins - 1, int((cy + half - y_min) / grid_res))
                if ix0 > ix1 or iy0 > iy1:
                    return 0
                total = prefix[ix1, iy1]
                if ix0 > 0: total -= prefix[ix0 - 1, iy1]
                if iy0 > 0: total -= prefix[ix1, iy0 - 1]
                if ix0 > 0 and iy0 > 0: total += prefix[ix0 - 1, iy0 - 1]
                return int(total)

            batch_size = min(N, max(target * 3, 100))
            candidate_idxs = np.random.choice(N, size=batch_size, replace=N < batch_size)
            found = 0
            for ci in candidate_idxs:
                cx, cy = points[ci, 0], points[ci, 1]
                if block_count(cx, cy) > 1024:
                    self.valid_centers.append((room_idx, points[ci, :3]))
                    found += 1
                    if found >= target:
                        break
        print("Totally {} samples in {} set.".format(len(self.valid_centers), split))

    def __getitem__(self, idx):
        room_idx, center = self.valid_centers[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N

        half = self.block_size / 2.0
        x_sort = self.room_x_sort[room_idx]
        sorted_x = self.room_x_sorted[room_idx]  # pre-computed, O(1)
        lo = np.searchsorted(sorted_x, center[0] - half, side='left')
        hi = np.searchsorted(sorted_x, center[0] + half, side='right')
        candidates = x_sort[lo:hi]
        y = points[candidates, 1]
        mask = (y >= center[1] - half) & (y <= center[1] + half)
        point_idxs = candidates[mask]

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(
                point_idxs, self.num_point, replace=False
            )
        else:
            selected_point_idxs = np.random.choice(
                point_idxs, self.num_point, replace=True
            )

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        coord_range = self.room_coord_max[room_idx] - self.room_coord_min[room_idx]
        coord_range = np.where(coord_range == 0, 1.0, coord_range)  # avoid div-by-zero
        current_points[:, 6] = (selected_points[:, 0] - self.room_coord_min[room_idx][0]) / coord_range[0]
        current_points[:, 7] = (selected_points[:, 1] - self.room_coord_min[room_idx][1]) / coord_range[1]
        current_points[:, 8] = (selected_points[:, 2] - self.room_coord_min[room_idx][2]) / coord_range[2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 2] = selected_points[:, 2] - center[2]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(
                current_points, current_labels
            )

        # if self.split == "train":
            # points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])

        return current_points, current_labels

    def __len__(self):
        return len(self.valid_centers)

class DatasetWholeScene:
    # prepare to give prediction on each points
    def __init__(
        self,
        root,
        block_points=4096,
        stride=0.5,
        block_size=1.0,
        padding=0.001,
        num_classes=21,
    ):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.stride = stride
        self.scene_points_num = []
        self.file_list = [
            d for d in os.listdir(root)
        ]
        self.scene_points_list = []
        self.room_coord_min, self.room_coord_max = [], []
        self.semantic_labels_list = []
        for file in self.file_list:
            data = np.load(os.path.join(root, file))
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, :6])

        labelweights = np.zeros(num_classes)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(num_classes + 1))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:, :6]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(
            np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride)
            + 1
        )
        grid_y = int(
            np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride)
            + 1
        )
        data_room = np.array([])

        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding)
                    & (points[:, 0] <= e_x + self.padding)
                    & (points[:, 1] >= s_y - self.padding)
                    & (points[:, 1] <= e_y + self.padding)
                )[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = (
                    False if (point_size - point_idxs.size <= point_idxs.size) else True
                )
                point_idxs_repeat = np.random.choice(
                    point_idxs, point_size - point_idxs.size, replace=replace
                )
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

                data_room = (
                    np.vstack([data_room, data_batch]) if data_room.size else data_batch
                )
                
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        
        return data_room

    def __len__(self):
        return len(self.scene_points_list)



if __name__ == "__main__":
    # data_root = "rpi_data"
    data_root = "data/rpi_data"
    num_point, block_size, sample_rate, num_classes = 4096, 1.0, 0.01, 47

    point_data = PointNetDataset(
        split="train",
        data_root=data_root,
        num_point=num_point,
        num_classes=num_classes,
        block_size=block_size,
        sample_rate=sample_rate,
        transform=None,
    )

    print("point data size:", point_data.__len__())
    print("point data 0 shape:", point_data.__getitem__(0)[0].shape)
    print("point label 0 shape:", point_data.__getitem__(0)[1].shape)

    data_loader = DataLoader(
        point_data,
        batch_size=16,
        shuffle=True,
    )

    print("---------")
    print("data_loader size:", len(data_loader))
    print("data_loader 0 shape:", data_loader.dataset.__getitem__(0)[0].shape)
    
    # point_data = DatasetWholeScene(
    #     root=data_root,
    #     block_points=num_point,
    #     num_classes=num_classes,
    #     block_size=block_size,
    # )

    # print("point data size:", point_data.__len__())
    # print("point data 0 shape:", point_data.__getitem__(0)[0].shape)
    # print("point label 0 shape:", point_data.__getitem__(0)[1].shape)


