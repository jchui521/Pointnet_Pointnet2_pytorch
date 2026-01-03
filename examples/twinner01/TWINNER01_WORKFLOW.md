# Twinner01 Complete Workflow Guide

## Overview
Complete pipeline for training custom semantic segmentation models on your annotated point cloud data.

**Runtime dependencies**: PyTorch, Open3D (already listed in `requirements.txt`).

---

## Data Format: Annotated .xyz Files (S3DIS-style)

### Directory Structure
```
raw_data/
    scene_01/
        chair_001.txt
        chair_002.txt
        table_001.txt
        floor_001.txt
        wall_001.txt
        wall_002.txt
    scene_02/
        ...
    scene_03/
        ...
```

### File Format
Each `.txt` file contains points for **one object instance**:
```
X Y Z R G B
1.23 4.56 0.12 200 150 100
1.24 4.57 0.13 201 151 101
...
```

**Rules**:
- Space-separated values (6 columns)
- X, Y, Z: 3D coordinates (meters, any scale)
- R, G, B: Color values (0-255)
- Filename prefix = class name (e.g., `chair_001.txt` → class "chair")

---

## Complete Workflow

### Step 1: Define Your Classes

Edit **[twinner01_classes_config.py](twinner01_classes_config.py)**:

```python
TWINNER01_CLASSES = [
    'background',   # Always have this for unknown/uncategorized
    'floor',
    'wall', 
    'chair',
    'table',
    'door',
    # Add all your object types
]
```

**Important**: Order matches label IDs (0, 1, 2, 3...)

---

### Step 2: Prepare Your Annotated Data

Convert annotated .xyz files to training format:

```powershell
python twinner01_prepare_data.py --mode scenes `
  --input raw_data `
  --output data/twinner01_custom
```

**What this does**:
- Reads all `.txt` files from each scene directory
- Maps filenames to class labels (using your config)
- Combines all objects into one `.npy` file per scene
- Saves to `data/twinner01_custom/scene_XX.npy`

**Output**: 
```
data/twinner01_custom/
    scene_01.npy  (N, 7) array
    scene_02.npy
    scene_03.npy
    ...
```

---

### Step 3: Verify Data

Test the dataloader:

```powershell
python twinner01_quickstart.py
```

Should show:
- ✓ Dataset loaded
- ✓ All classes present
- ✓ Sample shapes correct

---

### Step 4: Train Model

```powershell
# Basic training
python twinner01_train.py `
  --model pointnet2_sem_seg `
  --epoch 100 `
  --batch_size 8 `
  --log_dir my_experiment

# With custom parameters
python twinner01_train.py `
  --model pointnet2_sem_seg `
  --epoch 200 `
  --batch_size 4 `
  --learning_rate 0.001 `
  --npoint 4096 `
  --data_root data/twinner01_custom `
  --test_split 0.2 `
  --log_dir experiment_v1
```

**Parameters**:
- `--model`: `pointnet_sem_seg` (simpler) or `pointnet2_sem_seg` (better)
- `--epoch`: Training epochs (100-200 recommended)
- `--batch_size`: 4-16 depending on GPU memory
- `--npoint`: Points per sample (4096 default)
- `--test_split`: Fraction of data for testing (0.2 = 20%)
- `--log_dir`: Experiment name

**Output**: Model saved to `log/twinner01_sem_seg/<log_dir>/checkpoints/best_model.pth`

**Training time**: ~1-2 hours on GPU (depends on data size)

---

### Step 5: Test Model

```powershell
python twinner01_test.py `
  --log_dir my_experiment `
  --model pointnet2_sem_seg
```

**Output**:
- Overall accuracy
- Mean IoU (Intersection over Union)
- Per-class IoU and accuracy
- Results saved to `log/twinner01_sem_seg/<log_dir>/test_results.txt`

---

## Example: Complete Pipeline

```powershell
# 1. Activate environment
.\SET_UP_ENVIRONMENT.ps1

# 2. Edit classes
# Edit twinner01_classes_config.py with your classes

# 3. Prepare data
python twinner01_prepare_data.py --mode scenes `
  --input "C:\MyData\annotated_scenes" `
  --output "data/twinner01_custom"

# 4. Verify
python twinner01_quickstart.py

# 5. Train
python twinner01_train.py `
  --model pointnet2_sem_seg `
  --epoch 150 `
  --batch_size 8 `
  --log_dir my_model_v1

# 6. Test
python twinner01_test.py `
  --log_dir my_model_v1 `
  --model pointnet2_sem_seg
```

---

## Data Requirements Summary

| Aspect | Requirement |
|--------|------------|
| **Input format** | .txt files with X Y Z R G B (space-separated) |
| **Structure** | One .txt file per object instance |
| **Naming** | `<classname>_<id>.txt` (e.g., chair_01.txt) |
| **Min scenes** | 5 (3 train, 2 test minimum) |
| **Recommended** | 20+ scenes for good results |
| **Points/scene** | 1000+ (more is better) |
| **Classes** | Define all in twinner01_classes_config.py |

---

## Troubleshooting

**"Unknown class" warning**:
- Add missing class names to `TWINNER01_CLASSES` in config
- Or rename files to match existing classes

**Out of memory**:
- Reduce `--batch_size` to 4 or 2
- Reduce `--npoint` to 2048

**Low accuracy**:
- Train longer (--epoch 200+)
- Use more training data (20+ scenes)
- Use pointnet2_sem_seg model (better than pointnet)

**No valid blocks found**:
- Check if scenes have enough points (need 512+)
- Verify coordinate scales are reasonable (not too large/small)

---

## Files Reference

| File | Purpose |
|------|---------|
| [twinner01_classes_config.py](twinner01_classes_config.py) | Define your object classes |
| [twinner01_prepare_data.py](twinner01_prepare_data.py) | Convert .txt files to .npy format |
| [twinner01_dataloader.py](twinner01_dataloader.py) | PyTorch dataset loader |
| [twinner01_train.py](twinner01_train.py) | Training script |
| [twinner01_test.py](twinner01_test.py) | Testing/evaluation script |
| [twinner01_quickstart.py](twinner01_quickstart.py) | Quick setup test |
| [TWINNER01_DATA_GUIDE.md](TWINNER01_DATA_GUIDE.md) | Data format details |
| [TWINNER01_WORKFLOW.md](TWINNER01_WORKFLOW.md) | This guide |

---

## Next Steps

1. ✅ Prepare your annotated .xyz files
2. ✅ Update class configuration
3. ✅ Run data preparation
4. ✅ Train model
5. ✅ Evaluate results
6. 🔜 Use trained model for inference on new scenes

**Ready to train on your custom data!**
