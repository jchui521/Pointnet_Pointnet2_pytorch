# Original PointNet/PointNet++ Examples

This folder holds the original training/testing entrypoints for ModelNet classification, ShapeNet part segmentation, and S3DIS semantic segmentation. They are placed under `examples/original/*` to keep the repo root tidy while retaining the same functionality.

## Layout
- `classification/` – ModelNet40/10 training and evaluation.
- `part_seg/` – ShapeNet part segmentation training.
- `sem_seg/` – S3DIS semantic segmentation training.

Each subfolder contains a `train.py` (and `test.py` where applicable) plus `__init__.py` so they can be imported or run directly.

## Backward compatibility
Top-level wrappers (`train_classification.py`, `test_classification.py`, `train_partseg.py`, `train_semseg.py`) delegate to these files, so existing commands still work.
