# Quick Reference: Visualization Options

## What You Have Now ✅

After running the S3DIS minimal workflow, you have:

```
log/s3dis_minimal/s3dis_minimal_demo/visual/
├── Area_5_conferenceRoom_1_gt.obj          ← Ground Truth (1M+ colored points)
├── Area_5_conferenceRoom_1_pred.obj        ← Predictions (1M+ colored points)  
├── Area_5_conferenceRoom_1_gt.npy
├── Area_5_conferenceRoom_1_pred.npy
└── Area_5_conferenceRoom_1_original.npy
```

## Pick Your Visualization Method

### 🖥️ **Local Windows/Mac/Linux with GPU** (Easiest)

**Use:** Open3D interactive viewer
```powershell
python view_semseg_results_open3d_headless.py `
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" `
  --scene "Area_5_conferenceRoom_1"
```
**Result:** Opens interactive 3D window, shows GT then Pred side-by-side

---

### ☁️ **Remote/Headless/SSH Server** (No Display)

**Option A: Generate PNG images (Easiest)**
```powershell
python view_semseg_results_matplotlib.py `
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" `
  --scene "Area_5_conferenceRoom_1" `
  --sample_rate 0.15
```
**Result:** Creates 3 PNG files you can download/view

**Option B: Interactive HTML (Works in browser/VS Code)**
```powershell
python view_semseg_results_plotly.py `
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" `
  --scene "Area_5_conferenceRoom_1" `
  --sample_rate 0.1
```
**Result:** HTML file that opens in Simple Browser

---

### 📊 **Manual 3D Viewer (Always Works)**

1. Download **MeshLab** (free): http://www.meshlab.net/
2. Open MeshLab
3. File → Open → `Area_5_conferenceRoom_1_gt.obj` (or `_pred.obj`)
4. Use mouse to rotate/pan/zoom

**Alternative viewers:**
- CloudCompare: https://www.cloudcompare.org/
- Blender: https://www.blender.org/ (overkill but works)

---

## Quick Comparison

| Method | Setup Time | Works Headless | Interactive | Image Quality |
|--------|-----------|---|---|---|
| Open3D | 0 min | ❌ | ✅ | High |
| Matplotlib | 1 min | ✅ | ❌ | Good |
| Plotly | 2 min | ✅ | ✅ | Medium |
| MeshLab | 5 min | ✅ | ✅ | High |

---

## What You're Looking At

### Ground Truth (GT)
- Your data's actual semantic labels
- 13 classes color-coded:
  - Ceiling (light gray)
  - Floor (brown)
  - Wall (gray)
  - Beam, Column, Window, Door, Table, Chair, Sofa, Bookcase, Board, Clutter

### Predictions (Pred)
- Model's predicted semantic labels
- Same colors as ground truth
- Compare visually: where do predictions match GT?

---

## Recommended: Pick One 👇

### 💻 **"I have a local machine"**
```powershell
python view_semseg_results_open3d_headless.py `
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" `
  --scene "Area_5_conferenceRoom_1"
```

### 🌐 **"I'm on a remote server"**
```powershell
python view_semseg_results_matplotlib.py `
  --visual_dir "log/s3dis_minimal/s3dis_minimal_demo/visual" `
  --scene "Area_5_conferenceRoom_1"
```

### 🎨 **"I want best quality visualization"**
Download MeshLab and open:
```
log/s3dis_minimal/s3dis_minimal_demo/visual/Area_5_conferenceRoom_1_gt.obj
log/s3dis_minimal/s3dis_minimal_demo/visual/Area_5_conferenceRoom_1_pred.obj
```

---

## Files Generated

| File | Content | Size |
|------|---------|------|
| `*_gt.obj` | Ground truth, 1M+ colored points | ~60MB |
| `*_pred.obj` | Predictions, 1M+ colored points | ~60MB |
| `*_gt.npy` | Binary GT labels | ~4MB |
| `*_pred.npy` | Binary predicted labels | ~4MB |
| `*_visualization.png` | Static 3D view (matplotlib) | ~2MB |
| `*_comparison.png` | Side-by-side GT vs Pred | ~4MB |

---

## Troubleshooting

**Q: Windows shows "GLFW Error: WGL Failed"**  
A: Normal on headless. Use Matplotlib or MeshLab instead.

**Q: "ModuleNotFoundError: matplotlib"**  
A: `pip install matplotlib`

**Q: "ModuleNotFoundError: plotly"**  
A: `pip install plotly`

**Q: Where are my files?**  
A: `log/s3dis_minimal/s3dis_minimal_demo/visual/`

---

## Next Steps

1. ✅ Complete S3DIS training/testing (you did this!)
2. 📺 Visualize results (pick one method above)
3. 🔄 Improve model (more epochs, different architecture)
4. 📈 Train on more data (more rooms/areas)

---

**Full Guide:** See `OPEN3D_VISUALIZATION_GUIDE.md` for detailed info.
