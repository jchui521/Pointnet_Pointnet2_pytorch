# Open3D Visualization Setup Guide

## Overview

The S3DIS minimal workflow includes Open3D visualization to view semantic segmentation predictions vs ground truth in 3D. However, Open3D's `draw_geometries()` requires a GPU context and display server, which doesn't work in headless/remote environments.

## Solutions

### Solution 1: Use Generated .OBJ Files (Recommended for Headless)

The easiest solution is to use the saved `.obj` files with a 3D viewer:

**Files Generated:**
```
log/s3dis_minimal/s3dis_minimal_demo/visual/
├── Area_5_conferenceRoom_1_gt.obj      # Ground truth with colors
├── Area_5_conferenceRoom_1_pred.obj    # Predictions with colors
├── Area_5_conferenceRoom_1_original.npy
└── Area_5_conferenceRoom_1_gt.npy
```

**Viewers (Free & Open Source):**

1. **MeshLab** (Recommended)
   - Download: http://www.meshlab.net/
   - Supports: Windows, Linux, macOS
   - Usage: File → Open → Select .obj file
   - Features: Rotate (right-click), Pan (middle-click), Zoom (scroll)

2. **CloudCompare**
   - Download: https://www.cloudcompare.org/
   - Supports: Windows, Linux, macOS
   - Usage: File → Open → Select .obj file
   - Features: Great for point cloud comparison

3. **Blender**
   - Download: https://www.blender.org/
   - Supports: Windows, Linux, macOS
   - Usage: File → Import → OBJ
   - Features: Powerful, can create side-by-side views

### Solution 2: Matplotlib Visualization (Works Everywhere)

Generate static 3D visualizations as PNG images:

```powershell
python view_semseg_results_matplotlib.py `
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" `
  --scene "Area_5_conferenceRoom_1" `
  --sample_rate 0.1
```

**Output:**
- `Area_5_conferenceRoom_1_gt_visualization.png` - Ground truth 3D view
- `Area_5_conferenceRoom_1_pred_visualization.png` - Predictions 3D view  
- `Area_5_conferenceRoom_1_comparison.png` - Side-by-side comparison

### Solution 3: Virtual Display on Linux/macOS

If you're on Linux, use Xvfb (X Virtual Framebuffer):

```bash
# Install Xvfb
sudo apt-get install xvfb  # Ubuntu/Debian
sudo yum install xorg-x11-server-Xvfb  # RedHat/CentOS

# Run with virtual display
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
python view_semseg_results_open3d_headless.py \
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" \
  --scene "Area_5_conferenceRoom_1"
```

### Solution 4: Remote Display with SSH

If you have SSH access to a machine with display:

```bash
# Forward display to your local machine
ssh -X user@remote-host
# Then run visualization scripts
python view_semseg_results_open3d_headless.py ...
```

## Quick Start

### Windows (Local Machine with GPU)

```powershell
# Setup headless support (optional)
python setup_headless_open3d.py

# Run visualization
python view_semseg_results_open3d_headless.py `
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" `
  --scene "Area_5_conferenceRoom_1" `
  --show_gt --show_pred
```

**Controls:**
- Left-click + drag: Rotate view
- Right-click + drag: Pan/translate
- Scroll wheel: Zoom in/out
- Q or close window: Next view

### Linux (Local Machine with GPU)

```bash
python view_semseg_results_open3d_headless.py \
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" \
  --scene "Area_5_conferenceRoom_1" \
  --show_gt --show_pred
```

### Headless/Remote Environment

**Option A: Use Matplotlib (Easiest)**
```powershell
python view_semseg_results_matplotlib.py \
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" \
  --scene "Area_5_conferenceRoom_1" \
  --sample_rate 0.1
```

**Option B: Open OBJ files with MeshLab**
1. Open MeshLab
2. File → Open
3. Navigate to `log/s3dis_minimal/s3dis_minimal_demo/visual/`
4. Select `Area_5_conferenceRoom_1_gt.obj` or `Area_5_conferenceRoom_1_pred.obj`

## Comparison: Visualization Methods

| Method | Works Headless | Interactive | Cross-Platform | Setup |
|--------|---|---|---|---|
| **Open3D (GPU)** | ❌ No | ✅ Yes | ✅ All | GPU/Display required |
| **Open3D (Xvfb)** | ✅ Yes | ✅ Yes | ❌ Linux only | Complex |
| **Matplotlib** | ✅ Yes | ❌ Static | ✅ All | Easy |
| **OBJ + MeshLab** | ✅ Yes | ✅ Yes | ✅ All | Install MeshLab |
| **Plotly (HTML)** | ✅ Yes | ✅ Yes | ✅ All | Easy |

## Available Visualization Scripts

### 1. `view_semseg_results_open3d.py`
**Type:** Interactive Open3D viewer  
**Works In:** Local machine with GPU/display  
**Command:**
```powershell
python view_semseg_results_open3d.py \
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" \
  --scene "Area_5_conferenceRoom_1"
```

### 2. `view_semseg_results_open3d_headless.py`
**Type:** Interactive Open3D with fallback  
**Works In:** Most environments (graceful degradation)  
**Command:**
```powershell
python view_semseg_results_open3d_headless.py \
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" \
  --scene "Area_5_conferenceRoom_1"
```

### 3. `view_semseg_results_matplotlib.py`
**Type:** Static PNG visualization  
**Works In:** Everywhere (headless, remote, etc.)  
**Command:**
```powershell
python view_semseg_results_matplotlib.py \
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" \
  --scene "Area_5_conferenceRoom_1" \
  --sample_rate 0.1
```

### 4. `view_semseg_results_plotly.py`
**Type:** Interactive HTML visualization  
**Works In:** Everywhere (opens in browser/VS Code)  
**Command:**
```powershell
python view_semseg_results_plotly.py \
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" \
  --scene "Area_5_conferenceRoom_1" \
  --sample_rate 0.1
```

## File Locations

```
Project Root/
├── view_semseg_results.py                  # Basic Open3D viewer
├── view_semseg_results_open3d.py           # Enhanced Open3D
├── view_semseg_results_open3d_headless.py  # Headless-enabled
├── view_semseg_results_matplotlib.py       # Matplotlib (PNG output)
├── view_semseg_results_plotly.py          # Plotly (HTML output)
├── setup_headless_open3d.py                 # Setup script
└── log/s3dis_minimal/s3dis_minimal_demo/visual/
    ├── Area_5_conferenceRoom_1_gt.obj
    ├── Area_5_conferenceRoom_1_pred.obj
    ├── Area_5_conferenceRoom_1_gt_visualization.png
    ├── Area_5_conferenceRoom_1_pred_visualization.png
    └── Area_5_conferenceRoom_1_comparison.png
```

## Troubleshooting

### Problem: "GLFW Error: WGL: Failed to make context current"
**Cause:** No GPU context or display available  
**Solution:** Use matplotlib or Plotly visualization instead

### Problem: "ModuleNotFoundError: No module named 'pyvirtualdisplay'"
**Cause:** Virtual display library not installed  
**Solution:**
```powershell
pip install pyvirtualdisplay
# Or run setup script
python setup_headless_open3d.py
```

### Problem: Window opens but immediately closes
**Cause:** Headless environment  
**Solution:** Use matplotlib or save screenshots before window closes

### Problem: "ImportError: No module named 'open3d'"
**Cause:** Open3D not installed  
**Solution:**
```powershell
pip install open3d
```

## Recommended Workflow

For different environments:

### Local Windows/Mac/Linux Machine
✅ **Use:** `view_semseg_results_open3d_headless.py`  
```powershell
python view_semseg_results_open3d_headless.py \
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" \
  --scene "Area_5_conferenceRoom_1"
```

### Remote/Headless Server
✅ **Use:** `view_semseg_results_matplotlib.py` (generates PNG files)  
```powershell
python view_semseg_results_matplotlib.py \
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" \
  --scene "Area_5_conferenceRoom_1" \
  --sample_rate 0.2
```

### VS Code Remote/SSH
✅ **Use:** `view_semseg_results_plotly.py` (HTML in Simple Browser)  
```powershell
python view_semseg_results_plotly.py \
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" \
  --scene "Area_5_conferenceRoom_1" \
  --sample_rate 0.1
```

### Manual 3D Viewer
✅ **Use:** OBJ files with MeshLab  
1. Download MeshLab from http://www.meshlab.net/
2. Open: `log/s3dis_minimal/s3dis_minimal_demo/visual/Area_5_conferenceRoom_1_gt.obj`
3. Compare with: `log/s3dis_minimal/s3dis_minimal_demo/visual/Area_5_conferenceRoom_1_pred.obj`

## Summary

Open3D visualization works best on local machines with GPU and display. For headless/remote environments, use:
- **Matplotlib** for static PNG visualizations
- **Plotly** for interactive HTML visualizations
- **MeshLab** to manually view OBJ files

All methods show the same semantic segmentation results - choose the one that works best for your environment!
