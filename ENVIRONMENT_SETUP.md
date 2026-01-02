# Environment Setup Guide

This guide helps you quickly configure and activate the PointNet/PointNet++ environment in PowerShell.

## Quick Start

### Option 1: Run Setup Script (Recommended)
```powershell
# Navigate to project directory
cd c:\_LOCAL\GitHub\Pointnet_Pointnet2_pytorch

# Run the setup script
.\setup_env.ps1
```

This will:
- Activate the virtual environment
- Display GPU/CUDA information
- Show available quick commands

---

## Option 2: Add to PowerShell Profile (Persistent)

For persistent configuration across all PowerShell sessions:

### Installation Steps

1. **Open PowerShell as Administrator** and run:
   ```powershell
   notepad $PROFILE
   ```

2. **Copy content** from `powershell_profile.ps1` and paste into the opened file

3. **Save and close** the notepad

4. **Reload profile**:
   ```powershell
   . $PROFILE
   ```

### Available Commands After Profile Setup

| Command | Description |
|---------|-------------|
| `pointnet` | Activate PointNet environment and navigate to project |
| `pnpy <script.py>` | Run Python script using virtual environment |
| `pn-test-cls` | Run classification test |
| `pn-test-partseg` | Run part segmentation test |
| `pn-test-semseg` | Run semantic segmentation test |

### Examples
```powershell
# Activate environment
pointnet

# Run Python script
pnpy train_classification.py --model pointnet2_cls_ssg --epoch 10

# Quick tests
pn-test-cls
pn-test-partseg
pn-test-semseg
```

---

## Option 3: Manual Activation

```powershell
# Navigate to project
cd c:\_LOCAL\GitHub\Pointnet_Pointnet2_pytorch

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run Python commands
python test_classification.py --log_dir pointnet2_msg_normals --use_normals
```

---

## Environment Details

- **Python**: 3.9.10 (Virtual Environment: `.venv`)
- **PyTorch**: 2.6.0+cu124
- **CUDA**: 12.4 (Compatible with CUDA 12.8)
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB)

### Key Dependencies
- `torch==2.6.0+cu124`
- `numpy==2.0.2`
- `scikit-learn==1.6.1`
- `h5py==3.14.0`
- `tqdm`

---

## Troubleshooting

### PowerShell Execution Policy Error
If you get an error about execution policies:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Python Not Found
Make sure you're in the project directory and the `.venv` folder exists:
```powershell
cd c:\_LOCAL\GitHub\Pointnet_Pointnet2_pytorch
Test-Path .\.venv\Scripts\python.exe
```

### CUDA Not Available
Check NVIDIA driver:
```powershell
nvidia-smi
```

Verify PyTorch CUDA:
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

---

## Common Commands Reference

### Classification
```powershell
# Test pre-trained model
python test_classification.py --log_dir pointnet2_msg_normals --use_normals

# Train new model
python train_classification.py --model pointnet2_cls_ssg --log_dir my_model --batch_size 16 --epoch 50
```

### Part Segmentation
```powershell
# Test pre-trained model
python test_partseg.py --log_dir pointnet2_part_seg_msg --normal

# Train new model
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir my_partseg
```

### Semantic Segmentation
```powershell
# Test with visualization
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual

# Train new model
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir my_semseg
```

---

## Git Configuration

The environment is configured to exclude large data files:

**Excluded from Git** (in `.gitignore`):
- `data/` - Datasets (ModelNet40, ShapeNet, S3DIS)
- `log/` - Training outputs and checkpoints
- `.venv/` - Python virtual environment

**Included in Git**:
- All Python source code
- Model architectures
- Data loaders
- Documentation

---

## Additional Resources

- **Original Repository**: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- **Your Repository**: https://github.com/Drshelden/Pointnet_Pointnet2_pytorch
- **MeshLab** (for visualization): http://www.meshlab.net/

---

## Notes

- Use `--batch_size 16` or lower for 4GB GPU
- Pre-trained models are in `log/` directory
- Datasets should be in `data/` directory
- Results achieve: 92.78% (classification), 85.56% (part seg), 82.76% (semantic seg)
