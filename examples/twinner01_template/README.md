# Twinner01 Template (OH-1 baseline)

A self-contained template for Twinner01 runs. Copy this folder (rename it, e.g., `twinner01_oh2`) to spawn a new variant with its own class map and configs.

## What's here
- `twinner01_classes_config.py` – edit class names/colors; this file is loaded first by the wrappers.
- `train.py`, `test.py`, `prepare_data.py`, `quickstart.py`, `view_results.py` – wrappers that prefer the template config and call the original entrypoints in the repo root.

## How to use
1. (Optional) Duplicate this folder to start a new variant: `examples/twinner01_oh2/`.
2. Edit `twinner01_classes_config.py` inside your copy.
3. Run from repo root (PowerShell example):
   - `python examples/twinner01_template/prepare_data.py --mode scenes --input raw_data --output data/twinner01_custom`
   - `python examples/twinner01_template/train.py --log_dir oh1_baseline`
   - `python examples/twinner01_template/test.py --log_dir oh1_baseline`
   - `python examples/twinner01_template/view_results.py --visual_dir log/twinner01_sem_seg/oh1_baseline/visual`

Because the wrappers insert this folder on `sys.path` before the repo root, your template class config is honored without editing the shared scripts.
