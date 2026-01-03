# PointNet/PointNet++ Environment Setup Script
# Run this script to configure your PowerShell environment

Write-Host "Setting up PointNet/PointNet++ environment..." -ForegroundColor Cyan

# Set project root
$ProjectRoot = "c:\_LOCAL\GitHub\Pointnet_Pointnet2_pytorch"
Set-Location $ProjectRoot

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& "$ProjectRoot\.venv\Scripts\Activate.ps1"

# Set Python path (for convenience)
$env:PYTHON_PATH = "$ProjectRoot\.venv\Scripts\python.exe"

# Display environment info
Write-Host "`n============================================" -ForegroundColor Yellow
Write-Host "Environment Ready!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Yellow
Write-Host "Project Root: $ProjectRoot" -ForegroundColor White
Write-Host "Python: $env:PYTHON_PATH" -ForegroundColor White
Write-Host "`nGPU Info:" -ForegroundColor Cyan
& $env:PYTHON_PATH -c "import torch, open3d; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('Open3D:', open3d.__version__)"

Write-Host "`n============================================" -ForegroundColor Yellow
Write-Host "Quick Commands:" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Yellow
Write-Host "Classification Test:   " -NoNewline; Write-Host "python test_classification.py --log_dir pointnet2_msg_normals --use_normals" -ForegroundColor White
Write-Host "Part Segmentation:     " -NoNewline; Write-Host "python test_partseg.py --log_dir pointnet2_part_seg_msg --normal" -ForegroundColor White
Write-Host "Semantic Segmentation: " -NoNewline; Write-Host "python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual" -ForegroundColor White
Write-Host "`nTraining Example:      " -NoNewline; Write-Host "python train_classification.py --model pointnet2_cls_ssg --log_dir my_model --batch_size 16 --epoch 10" -ForegroundColor White
Write-Host "============================================`n" -ForegroundColor Yellow
