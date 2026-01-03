# S3DIS Minimal Example (Area 5 / conferenceRoom_1)

Lightweight smoke-test for semantic segmentation on a tiny S3DIS subset. Place a single preprocessed room file (e.g., `Area_5_conferenceRoom_1.npy`) under `data/stanford_indoor3d_minimal/` and run the minimal trainer.

## Quick Start

### 1. Generate synthetic test data
```powershell
# First, create sample room data (50k points)
python examples/s3dis_minimal/generate_synthetic_data.py
```

### 2. Run the complete end-to-end workflow (training):

**PowerShell:**
```powershell
cd examples/s3dis_minimal
.\run_workflow.ps1 -Epoch 5 -BatchSize 2 -LogDir "my_test"
```

**Python:**
```bash
python examples/s3dis_minimal/run_workflow.py --epoch 5 --batch_size 2 --log_dir my_test
```

## Full Steps

1. Ensure you have at least one preprocessed S3DIS room file in `data/stanford_indoor3d_minimal/`, e.g. `Area_5_conferenceRoom_1.npy` (shape `(N, 7)` → `[X, Y, Z, R, G, B, Label]`).

2. **Run complete workflow** (recommended):
   ```powershell
   # PowerShell from repo root
   .\examples\s3dis_minimal\run_workflow.ps1 -Epoch 10 -BatchSize 4
   ```

3. **Or run individual steps manually**:
   ```powershell
   # Training
   python examples/s3dis_minimal/train_semseg_minimal.py \
     --data_root data/stanford_indoor3d_minimal \
     --room_glob "Area_5_conferenceRoom_1.npy" \
     --npoint 2048 --batch_size 2 --epoch 5 --sample_rate 0.1 \
     --log_dir my_test
   ```

4. Logs/checkpoints land in `log/s3dis_minimal/<log_dir>/`.

## Workflow Script Options

```powershell
# Full training with custom settings
.\run_workflow.ps1 -Epoch 20 -BatchSize 4 -NPoint 4096 -LogDir "longer_run" -Model pointnet2_sem_seg

# Skip training (use existing checkpoint)
.\run_workflow.ps1 -SkipTrain -SkipViz

# View available options
Get-Help .\run_workflow.ps1 -Full
```

This path keeps the original S3DIS pipelines intact while offering a minimal, quick-to-run smoke test.
