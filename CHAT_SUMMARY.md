# Twinner01 Semantic Segmentation – Expanded Chat Summary

## Context
We collaborated to stand up a custom semantic segmentation workflow on PointNet/PointNet++, tailored for Twinner01 point clouds, with Open3D visualization, streamlined data prep, and clarified run commands.

## Core Outcomes
- End-to-end path: data prep → training → testing → Open3D visualization.
- Class mapping centralized in `twinner01_classes_config.py` to keep labels consistent across prep, train, and viz.
- Open3D visualizer (`view_semseg_results.py`) replaces legacy C++/cv2 flow; outputs sync with model runs.
- Environment bootstrapped via `requirements.txt` and `SET_UP_ENVIRONMENT.ps1`.

## Back-and-Forth Highlights
- Aligned data expectations: S3DIS-style folders, instance files as `X Y Z R G B` with filenames carrying the class prefix (e.g., `chair_001.txt`).
- Confirmed conversion target: `data/twinner01_custom/scene_xx.npy` shaped `(N, 7)` → `[X, Y, Z, R, G, B, Label]`.
- Clarified train/test invocation for PointNet++ semantic segmentation, including adjustable batch size and epochs for GPU fit.
- Swapped to Open3D viewing: sequential GT then Pred windows; emphasized **Q** (or close) to advance; mouse for orbit/zoom/pan.
- Located artifacts: logs and `.obj` meshes under `log/sem_seg/<log_dir>/visual/`.

## How to Run
### Environment
```powershell
pip install -r requirements.txt
./SET_UP_ENVIRONMENT.ps1
```

### Prepare Data (scenes)
```powershell
python twinner01_prepare_data.py --mode scenes \
  --input raw_data \
  --output data/twinner01_custom
```

### Train (PointNet++ semantic segmentation)
```powershell
python twinner01_train.py \
  --model pointnet2_sem_seg \
  --epoch 100 \
  --batch_size 8 \
  --log_dir my_exp
```

### Test
```powershell
python twinner01_test.py \
  --log_dir my_exp \
  --model pointnet2_sem_seg
```

### Visualize (Open3D)
```powershell
# List available scenes
python view_semseg_results.py --visual_dir log/sem_seg/pointnet2_sem_seg/visual

# View a specific scene (GT then Pred windows)
python view_semseg_results.py --visual_dir log/sem_seg/pointnet2_sem_seg/visual --scene Area_5_office_1
```

## Data and Labels
- Input: per-scene folders, per-object files with `X Y Z R G B`.
- Label assignment: filename prefix must match class names defined in `twinner01_classes_config.py`; label 0 commonly reserved for background.
- Output: merged scene arrays saved as `.npy` ready for the dataloader.

## Operational Notes
- Reduce `--batch_size` and/or `--npoint` on constrained GPUs.
- Ensure Open3D installed (in `requirements.txt`); use the PowerShell helper for Windows setup.
- Artifacts for inspection live in `log/sem_seg/<log_dir>/visual/` alongside any exported meshes.
- Adjust class colors or mappings centrally in `twinner01_classes_config.py` to keep training, testing, and visualization in sync.

## Next Steps
- Integrate your custom class list in `twinner01_classes_config.py` and regenerate data.
- Run a short epoch sanity check (`--epoch 5`, small `--batch_size`) before long training.
- Use the Open3D viewer to validate label colors and alignment after each training run.
