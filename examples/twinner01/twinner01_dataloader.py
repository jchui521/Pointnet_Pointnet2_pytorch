"""
Twinner01 Custom DataLoader for Semantic Segmentation
Based on S3DISDataLoader but adapted for custom data
"""

import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class Twinner01Dataset(Dataset):
    """
    Custom dataset loader for Twinner01 project
    
    Data directory structure should be:
        data/twinner01_custom/
            scene_01.npy
            scene_02.npy
            scene_03.npy
            ...
    
    Each .npy file contains: (N, 7) array
        Columns: [X, Y, Z, R, G, B, Label]
    """
    
    def __init__(self, split='train', data_root='data/twinner01_custom', 
                 num_point=4096, test_split=0.2, block_size=1.0, 
                 sample_rate=1.0, transform=None, num_classes=6):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.num_classes = num_classes
        
        # Load all scene files
        if not os.path.exists(data_root):
            raise ValueError(f"Data directory not found: {data_root}")
        
        all_files = sorted([f for f in os.listdir(data_root) if f.endswith('.npy')])
        
        if len(all_files) == 0:
            raise ValueError(f"No .npy files found in {data_root}")
        
        # Split into train/test
        num_test = max(1, int(len(all_files) * test_split))
        if split == 'train':
            scene_files = all_files[:-num_test]
        else:
            scene_files = all_files[-num_test:]
        
        print(f"Loading {len(scene_files)} scenes for {split} set from {data_root}")
        
        # Load all scenes
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(self.num_classes)
        
        for scene_file in tqdm(scene_files, desc=f"Loading {split} data"):
            scene_path = os.path.join(data_root, scene_file)
            scene_data = np.load(scene_path)  # (N, 7): xyzrgbl
            
            if scene_data.shape[1] != 7:
                print(f"Warning: {scene_file} has shape {scene_data.shape}, expected (N, 7)")
                continue
            
            points, labels = scene_data[:, 0:6], scene_data[:, 6]  # xyzrgb (N, 6); labels (N,)
            
            # Count label frequencies for weighting
            tmp, _ = np.histogram(labels, range(self.num_classes + 1))
            labelweights += tmp
            
            # Get coordinate bounds
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            
            self.room_points.append(points)
            self.room_labels.append(labels)
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        
        # Calculate label weights for balanced training
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / (labelweights + 1e-10), 1 / 3.0)
        print(f"Label weights: {self.labelweights}")
        
        # Sample scenes based on their size
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(scene_files)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        
        print(f"Total {len(self.room_idxs)} samples in {split} set.")
    
    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # (N, 6)
        labels = self.room_labels[room_idx]  # (N,)
        N_points = points.shape[0]
        
        # Sample a random block from the scene
        max_attempts = 100
        for attempt in range(max_attempts):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
                                  (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size >= 512:  # Ensure enough points
                break
            if attempt == max_attempts - 1:
                # Fallback: use all points if we can't find a good block
                point_idxs = np.arange(N_points)
        
        # Sample points from block
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
        
        # Normalize
        selected_points = points[selected_point_idxs, :]  # (num_point, 6)
        current_points = np.zeros((self.num_point, 9))  # (num_point, 9)
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0  # Normalize RGB
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        
        return current_points, current_labels
    
    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    # Test the dataloader
    print("Testing Twinner01Dataset...")
    
    try:
        dataset = Twinner01Dataset(
            split='train',
            data_root='data/twinner01_custom',
            num_point=4096,
            num_classes=6
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")
        
        # Test loading a sample
        points, labels = dataset[0]
        print(f"\nSample data:")
        print(f"Points shape: {points.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("\nTo create sample data, run:")
        print("python twinner01_prepare_data.py --mode sample")
