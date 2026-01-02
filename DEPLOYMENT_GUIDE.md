# Deployment Guide for NVIDIA BREV/LAUNCHABLES

## Prerequisites
- BREV or LAUNCHABLES account
- GPU instance (T4, A10G, or better recommended)
- Git repository access

## Step-by-Step Deployment

### 1. Create BREV/LAUNCHABLES Instance
1. Log into your BREV dashboard at https://console.brev.dev
2. Click "New Instance"
3. Select:
   - **Template**: PyTorch or Ubuntu 22.04
   - **GPU**: T4 (16GB) or A10G (24GB) minimum
   - **Region**: Choose nearest to you
4. Click "Create Instance"

### 2. Connect to Instance
```bash
# Option A: Using BREV CLI
brev open <instance-name>

# Option B: Using SSH (BREV provides command)
ssh <username>@<instance-ip>
```

### 3. Clone Repository
```bash
cd ~
git clone https://github.com/drshelden/Pointnet_Pointnet2_pytorch.git
cd Pointnet_Pointnet2_pytorch
```

### 4. Setup Python Environment
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 5. Verify CUDA Setup
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 6. Verify Data
```bash
# Check if data directories exist
ls -la data/modelnet40_normal_resampled/
ls -la data/shapenetcore_partanno_segmentation_benchmark_v0_normal/
ls -la data/stanford_indoor3d/

# Check data file counts
echo "ModelNet40 files: $(find data/modelnet40_normal_resampled -name '*.txt' | wc -l)"
echo "ShapeNet files: $(find data/shapenetcore_partanno_segmentation_benchmark_v0_normal -name '*.txt' | wc -l)"
echo "S3DIS files: $(find data/stanford_indoor3d -name '*.npy' | wc -l)"
```

### 7. Train Models (Optional but Recommended)
```bash
# Train classification model (ModelNet40)
python train_classification.py \
  --model pointnet2_cls_ssg \
  --log_dir log/classification \
  --epoch 200 \
  --batch_size 24 \
  --learning_rate 0.001

# Train part segmentation model (ShapeNet)
python train_partseg.py \
  --model pointnet2_part_seg_ssg \
  --log_dir log/part_seg \
  --epoch 200 \
  --batch_size 16

# Train semantic segmentation model (S3DIS)
python train_semseg.py \
  --model pointnet2_sem_seg \
  --log_dir log/sem_seg \
  --epoch 100 \
  --batch_size 8 \
  --test_area 5
```

### 8. Run Test Scripts

#### Test Classification
```bash
python test_classification.py \
  --log_dir log/classification \
  --model pointnet2_cls_ssg \
  --batch_size 24
```

Expected output: Accuracy metrics on ModelNet40 test set

#### Test Part Segmentation
```bash
python test_partseg.py \
  --log_dir log/part_seg \
  --model pointnet2_part_seg_ssg \
  --batch_size 16
```

Expected output: mIoU (mean Intersection over Union) per category

#### Test Semantic Segmentation
```bash
python test_semseg.py \
  --log_dir log/sem_seg \
  --model pointnet2_sem_seg \
  --batch_size 8 \
  --test_area 5
```

Expected output: mIoU and accuracy on S3DIS Area 5

### 9. Monitor GPU Usage
```bash
# In a separate terminal/tmux session
watch -n 1 nvidia-smi
```

### 10. Download Results
```bash
# From your local machine
scp -r <instance>:~/Pointnet_Pointnet2_pytorch/log ./results
```

## Troubleshooting

### Out of Memory Errors
- Reduce `--batch_size` (try 8, 4, or even 2)
- Use smaller model variant (`ssg` instead of `msg`)

### CUDA Not Available
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Data Not Found
- Ensure data directories are uploaded or downloaded
- Check file paths in test scripts match data locations

### Missing Checkpoints
- Run training scripts first to generate model checkpoints
- Or download pre-trained weights and place in log directories

## Performance Tips
- Use `tmux` or `screen` for long-running training jobs
- Enable mixed precision training with `--use_amp` flag (if available)
- Monitor GPU memory with `nvidia-smi dmon -s mu`

## Cost Optimization
- Stop instances when not in use
- Use spot instances for training (if supported)
- Download important results regularly
