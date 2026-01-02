#!/bin/bash
# Run All Test Scripts
# Run this after training models or with pre-trained checkpoints

set -e

echo "========================================="
echo "Running All PointNet/PointNet2 Tests"
echo "========================================="

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Test Classification
echo ""
echo "========================================="
echo "Test 1: Classification (ModelNet40)"
echo "========================================="
if [ -d "log/classification" ] && [ -n "$(find log/classification -name '*.pth' 2>/dev/null)" ]; then
    python test_classification.py \
      --log_dir log/classification \
      --model pointnet2_cls_ssg \
      --batch_size 24
    echo "✓ Classification test completed"
else
    echo "✗ No trained classification model found in log/classification"
    echo "  Run: python train_classification.py first"
fi

# Test Part Segmentation
echo ""
echo "========================================="
echo "Test 2: Part Segmentation (ShapeNet)"
echo "========================================="
if [ -d "log/part_seg" ] && [ -n "$(find log/part_seg -name '*.pth' 2>/dev/null)" ]; then
    python test_partseg.py \
      --log_dir log/part_seg \
      --model pointnet2_part_seg_ssg \
      --batch_size 16
    echo "✓ Part segmentation test completed"
else
    echo "✗ No trained part segmentation model found in log/part_seg"
    echo "  Run: python train_partseg.py first"
fi

# Test Semantic Segmentation
echo ""
echo "========================================="
echo "Test 3: Semantic Segmentation (S3DIS)"
echo "========================================="
if [ -d "log/sem_seg" ] && [ -n "$(find log/sem_seg -name '*.pth' 2>/dev/null)" ]; then
    python test_semseg.py \
      --log_dir log/sem_seg \
      --model pointnet2_sem_seg \
      --batch_size 8 \
      --test_area 5
    echo "✓ Semantic segmentation test completed"
else
    echo "✗ No trained semantic segmentation model found in log/sem_seg"
    echo "  Run: python train_semseg.py first"
fi

echo ""
echo "========================================="
echo "All Tests Complete!"
echo "========================================="
echo "Check log/ directories for detailed results"
