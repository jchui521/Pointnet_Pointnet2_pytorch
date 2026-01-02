#!/bin/bash
# Remote Setup Script for BREV/LAUNCHABLES
# Run this script after cloning the repository on the remote instance

set -e

echo "========================================="
echo "PointNet/PointNet2 Remote Setup"
echo "========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Verify CUDA
echo ""
echo "========================================="
echo "CUDA Verification"
echo "========================================="
python3 -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Check data
echo ""
echo "========================================="
echo "Data Verification"
echo "========================================="
if [ -d "data/modelnet40_normal_resampled" ]; then
    echo "✓ ModelNet40 data found"
    echo "  Files: $(find data/modelnet40_normal_resampled -name '*.txt' | wc -l)"
else
    echo "✗ ModelNet40 data not found"
fi

if [ -d "data/shapenetcore_partanno_segmentation_benchmark_v0_normal" ]; then
    echo "✓ ShapeNet data found"
else
    echo "✗ ShapeNet data not found"
fi

if [ -d "data/stanford_indoor3d" ]; then
    echo "✓ S3DIS data found"
    echo "  Files: $(find data/stanford_indoor3d -name '*.npy' | wc -l)"
else
    echo "✗ S3DIS data not found"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Train models or run tests"
echo "3. Monitor with: nvidia-smi -l 1"
