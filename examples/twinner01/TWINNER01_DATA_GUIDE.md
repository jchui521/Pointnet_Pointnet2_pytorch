# Twinner01 Custom Semantic Segmentation Setup Guide

## Overview
This guide walks you through setting up custom semantic segmentation for your own objects using the Twinner01 pipeline.

---

## Required Data Characteristics

> Tip: Install Open3D (already in `requirements.txt`) to visualize your scenes and labels after processing.

### Data Format
Your point cloud data must be in **NumPy format (.npy)** with this structure:

```
Shape: (N, 7)
Where N = number of points in the scene
Columns: [X, Y, Z, R, G, B, Label]
```

**Column Details:**
- **X, Y, Z**: 3D coordinates (float32) - spatial position of each point
- **R, G, B**: Color values (0-255, uint8 or float32) - appearance information
- **Label**: Class ID (int) - semantic label (0 to num_classes-1)

### Data Requirements
1. **Minimum points per scene**: 512 (preferably 4096+)
2. **Multiple scenes**: At least 5-10 scenes for training
3. **Labeled data**: Every point must have a label
4. **Consistent labeling**: Same object type → same label across all scenes
5. **Coordinate system**: Any coordinate system works (will be normalized)

---

## Step-by-Step Setup

### Step 1: Define Your Classes

Edit `twinner01_classes_config.py`:

```python
TWINNER01_CLASSES = [
    'background',      # Class 0
    'my_object_1',     # Class 1 - e.g., 'car'
    'my_object_2',     # Class 2 - e.g., 'tree'
    'my_object_3',     # Class 3 - e.g., 'building'
    # Add all your classes
]
```

**Important**: 
- Label 0 is typically background/unknown
- Order matters - indices must match your data labels
- Keep names descriptive

### Step 2: Prepare Your Data

#### Option A: You Have Raw Point Cloud Files

If your data is in TXT, CSV, or other format:

```powershell
# Single file
python twinner01_prepare_data.py --mode single `
  --input "path/to/your/scene.txt" `
  --output "data/twinner01_custom/scene_01.npy"

# Batch process directory
python twinner01_prepare_data.py --mode batch `
  --input "path/to/raw_data/" `
  --output "data/twinner01_custom/" `
  --pattern "*.txt"
```

**Expected input format** (TXT/CSV):
```
X, Y, Z, Label           # 4 columns (colors added automatically)
OR
X, Y, Z, R, G, B, Label  # 7 columns (complete)
```

#### Option B: Create Sample Data for Testing

```powershell
# Create dummy data to test pipeline
python twinner01_prepare_data.py --mode sample `
  --output "data/twinner01_custom/sample_scene_01.npy" `
  --num_points 10000 `
  --num_classes 6
```

#### Option C: Convert Existing NumPy Data

If you already have .npy files but wrong format:

```python
import numpy as np

# Load your data
data = np.load('your_data.npy')

# Reshape if needed to (N, 7)
# Example: if you have separate xyz, rgb, labels
xyz = data['coordinates']  # (N, 3)
rgb = data['colors']       # (N, 3)
labels = data['labels']    # (N,)

combined = np.hstack([xyz, rgb, labels.reshape(-1, 1)])
np.save('data/twinner01_custom/scene_01.npy', combined.astype(np.float32))
```

### Step 3: Organize Your Data

Create directory structure:

```
data/
  twinner01_custom/
    scene_01.npy    # Training/test scenes
    scene_02.npy
    scene_03.npy
    scene_04.npy
    scene_05.npy
    ...
```

**Recommendation**:
- **Minimum**: 5 scenes (3 train, 2 test)
- **Good**: 20+ scenes (16 train, 4 test)
- **Optimal**: 100+ scenes

### Step 4: Test Your Data Loader

```powershell
python twinner01_dataloader.py
```

Expected output:
```
Loading 3 scenes for train set...
Label weights: [1.2, 0.9, 1.1, ...]
Total 500 samples in train set.
Dataset loaded successfully!
Points shape: (4096, 9)
Labels shape: (4096,)
```

---

## Data Specifications Summary

| Property | Requirement |
|----------|-------------|
| **File format** | .npy (NumPy array) |
| **Array shape** | (N, 7) per scene |
| **Columns** | [X, Y, Z, R, G, B, Label] |
| **Coordinate type** | float32 |
| **Color range** | 0-255 (will be normalized) |
| **Label type** | int (0 to num_classes-1) |
| **Min points/scene** | 512 (4096+ recommended) |
| **Min scenes** | 5 for testing, 20+ for real training |
| **Coordinate system** | Any (will be centered/normalized) |

---

## Common Data Sources

### If You're Starting from Scratch

1. **LiDAR scans**: Export as XYZ + Label
2. **Photogrammetry**: CloudCompare → annotate → export
3. **Synthetic data**: Blender/Unity → export point clouds
4. **Existing datasets**: Convert from PCD, PLY, LAS formats

### Converting Other Formats

```python
# From .ply or .pcd files (use open3d)
import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("scene.ply")
xyz = np.asarray(pcd.points)
rgb = np.asarray(pcd.colors) * 255  # open3d uses 0-1

# You need to add labels manually or from another source
labels = your_label_array  # (N,)

data = np.hstack([xyz, rgb, labels.reshape(-1, 1)])
np.save('data/twinner01_custom/scene.npy', data.astype(np.float32))
```

---

## Next Steps

After data preparation:

1. **Verify data**: Run `twinner01_dataloader.py` to test
2. **Update config**: Edit `twinner01_classes_config.py` with actual classes
3. **Create training script**: Next step - adapt train_semseg.py
4. **Train model**: Run training on your custom data
5. **Test model**: Evaluate on held-out test scenes

---

## Troubleshooting

**"No .npy files found"**
→ Run `twinner01_prepare_data.py` first

**"Wrong shape"**
→ Ensure data is (N, 7) not (7, N) or other

**"Labels out of range"**
→ Check labels are 0 to num_classes-1

**"Not enough points"**
→ Combine multiple small scenes or use denser scans

---

## Files Created

1. `twinner01_classes_config.py` - Class definitions
2. `twinner01_prepare_data.py` - Data conversion tool
3. `twinner01_dataloader.py` - PyTorch dataset loader
4. `TWINNER01_DATA_GUIDE.md` - This guide

**Next**: Training script creation
