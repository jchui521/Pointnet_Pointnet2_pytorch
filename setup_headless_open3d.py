#!/usr/bin/env python3
"""
Open3D Headless Visualization Setup
Configures Open3D to work without GPU/display in headless environments
"""

import os
import sys
import subprocess

def setup_headless_open3d():
    """Setup Open3D for headless rendering"""
    
    print("="*70)
    print("Setting up Open3D for Headless/Remote Environments")
    print("="*70)
    
    # Check OS
    is_windows = sys.platform.startswith('win')
    is_linux = sys.platform.startswith('linux')
    is_macos = sys.platform.startswith('darwin')
    
    print(f"\nDetected OS: {sys.platform}")
    
    if is_windows:
        setup_windows_headless()
    elif is_linux:
        setup_linux_headless()
    elif is_macos:
        setup_macos_headless()
    else:
        print(f"Unknown OS: {sys.platform}")
    
    print("\n" + "="*70)
    print("Setup Complete!")
    print("="*70)


def setup_windows_headless():
    """Setup Open3D for Windows headless environment"""
    print("\n[Windows] Setting up headless configuration...")
    
    print("\nOption 1: Virtual Display (Recommended)")
    print("  Install: pyvirtualdisplay")
    print("  Run: pip install pyvirtualdisplay")
    
    print("\nOption 2: Use OSMesa for software rendering")
    print("  (Requires Mesa3D installation)")
    
    print("\nOption 3: Run on local machine with display")
    print("  The .obj files are saved in:")
    print("    log/s3dis_minimal/s3dis_minimal_demo/visual/")
    print("  Open them with:")
    print("    - MeshLab (free)")
    print("    - CloudCompare (free)")
    print("    - Blender")
    
    # Create windows batch script for virtual display
    batch_script = """@echo off
REM Open3D Headless Visualization Wrapper for Windows

REM Check if pyvirtualdisplay is installed
python -c "import pyvirtualdisplay" 2>nul
if errorlevel 1 (
    echo Installing pyvirtualdisplay...
    pip install pyvirtualdisplay
)

REM Run with virtual display
python -c "from pyvirtualdisplay import Display; disp = Display(); disp.start(); exec(open('view_semseg_results_open3d.py').read())"
"""
    
    with open('run_open3d_headless.bat', 'w') as f:
        f.write(batch_script)
    
    print("\n  Created: run_open3d_headless.bat")


def setup_linux_headless():
    """Setup Open3D for Linux headless environment"""
    print("\n[Linux] Setting up headless configuration...")
    
    print("\nOption 1: Xvfb (Virtual Framebuffer) - Recommended")
    print("  Install: xvfb")
    print("  Ubuntu/Debian: sudo apt-get install xvfb")
    print("  RedHat/CentOS: sudo yum install xorg-x11-server-Xvfb")
    
    print("\nOption 2: Use OSMesa for software rendering")
    print("  Install: libosmesa6-dev")
    print("  Ubuntu/Debian: sudo apt-get install libosmesa6-dev")
    
    # Create bash script for Xvfb
    bash_script = """#!/bin/bash
# Open3D Headless Visualization Wrapper for Linux

# Check if Xvfb is available
if command -v Xvfb &> /dev/null; then
    echo "Starting virtual display..."
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    XVFB_PID=$!
    export DISPLAY=:99
    
    python view_semseg_results_open3d.py "$@"
    
    # Kill Xvfb
    kill $XVFB_PID 2>/dev/null
else
    echo "Xvfb not found. Install with: sudo apt-get install xvfb"
    echo "Running without display (windows will not appear)..."
    python view_semseg_results_open3d.py "$@"
fi
"""
    
    with open('run_open3d_headless.sh', 'w') as f:
        f.write(bash_script)
    
    os.chmod('run_open3d_headless.sh', 0o755)
    print("\n  Created: run_open3d_headless.sh")
    print("  Run: chmod +x run_open3d_headless.sh && ./run_open3d_headless.sh")


def setup_macos_headless():
    """Setup Open3D for macOS headless environment"""
    print("\n[macOS] Setting up headless configuration...")
    
    print("\nOpen3D requires a display on macOS")
    print("Options:")
    print("  1. Run on a machine with display")
    print("  2. Use VNC remote desktop")
    print("  3. Forward display via SSH: ssh -Y user@host")
    
    print("\nAlternatively, open the OBJ files with:")
    print("  - Preview.app (built-in)")
    print("  - Meshlab (free)")
    print("  - CloudCompare (free)")


def install_virtual_display():
    """Install virtual display package"""
    print("\n" + "="*70)
    print("Installing Virtual Display Support")
    print("="*70)
    
    try:
        print("\nInstalling pyvirtualdisplay...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyvirtualdisplay'])
        print("✓ pyvirtualdisplay installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install: {e}")
        return False
    
    return True


if __name__ == '__main__':
    setup_headless_open3d()
    
    # Ask if user wants to install virtual display
    if not sys.platform.startswith('linux'):
        response = input("\n\nWould you like to install pyvirtualdisplay for virtual display support? (y/n): ")
        if response.lower() == 'y':
            install_virtual_display()
