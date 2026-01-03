@echo off
REM Open3D Headless Visualization Wrapper for Windows

REM Check if pyvirtualdisplay is installed
python -c "import pyvirtualdisplay" 2>nul
if errorlevel 1 (
    echo Installing pyvirtualdisplay...
    pip install pyvirtualdisplay
)

REM Run with virtual display
python -c "from pyvirtualdisplay import Display; disp = Display(); disp.start(); exec(open('view_semseg_results_open3d.py').read())"
