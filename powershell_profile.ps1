# PointNet/PointNet++ PowerShell Profile Configuration
# Add this content to your PowerShell profile for persistent environment setup

# ========================================
# To install, run this command:
# notepad $PROFILE
# Then copy and paste the content below
# ========================================

# Function to activate PointNet environment
function Start-PointNetEnv {
    $ProjectRoot = "c:\_LOCAL\GitHub\Pointnet_Pointnet2_pytorch"
    
    if (Test-Path $ProjectRoot) {
        Set-Location $ProjectRoot
        & "$ProjectRoot\.venv\Scripts\Activate.ps1"
        Write-Host "PointNet environment activated!" -ForegroundColor Green
    } else {
        Write-Host "Error: PointNet project not found at $ProjectRoot" -ForegroundColor Red
    }
}

# Alias for quick access
Set-Alias -Name pointnet -Value Start-PointNetEnv

# Function to run Python scripts in the virtual environment
function Invoke-PointNetPython {
    param([string]$Script, [string]$Args)
    $PythonExe = "c:\_LOCAL\GitHub\Pointnet_Pointnet2_pytorch\.venv\Scripts\python.exe"
    
    if ($Script) {
        & $PythonExe $Script $Args
    } else {
        & $PythonExe
    }
}

# Alias for Python execution
Set-Alias -Name pnpy -Value Invoke-PointNetPython

# Quick test commands
function Test-PointNetClassification {
    Set-Location "c:\_LOCAL\GitHub\Pointnet_Pointnet2_pytorch"
    & ".\.venv\Scripts\python.exe" test_classification.py --log_dir pointnet2_msg_normals --use_normals
}

function Test-PointNetPartSeg {
    Set-Location "c:\_LOCAL\GitHub\Pointnet_Pointnet2_pytorch"
    & ".\.venv\Scripts\python.exe" test_partseg.py --log_dir pointnet2_part_seg_msg --normal
}

function Test-PointNetSemSeg {
    Set-Location "c:\_LOCAL\GitHub\Pointnet_Pointnet2_pytorch"
    & ".\.venv\Scripts\python.exe" test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual
}

# Aliases for test commands
Set-Alias -Name pn-test-cls -Value Test-PointNetClassification
Set-Alias -Name pn-test-partseg -Value Test-PointNetPartSeg
Set-Alias -Name pn-test-semseg -Value Test-PointNetSemSeg

Write-Host "PointNet commands loaded! Type 'pointnet' to activate environment." -ForegroundColor Cyan
