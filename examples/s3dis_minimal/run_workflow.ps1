<#
.SYNOPSIS
S3DIS Minimal Complete Workflow Runner (PowerShell)
Runs all steps: training, testing, and visualization.

.PARAMETER DataRoot
Path to the minimal S3DIS data directory (default: data/stanford_indoor3d_minimal)

.PARAMETER RoomGlob
Glob pattern to select room files (default: Area_5_conferenceRoom_1*.npy)

.PARAMETER LogDir
Name of the logging directory (default: s3dis_minimal_run)

.PARAMETER Model
Model name [pointnet_sem_seg|pointnet2_sem_seg] (default: pointnet_sem_seg)

.PARAMETER Epoch
Number of training epochs (default: 5)

.PARAMETER BatchSize
Batch size (default: 2)

.PARAMETER NPoint
Points per block (default: 2048)

.PARAMETER SampleRate
Sampling rate (default: 0.1)

.PARAMETER SkipTrain
Skip the training step

.PARAMETER SkipTest
Skip the testing step

.PARAMETER SkipViz
Skip the visualization step

.EXAMPLE
.\run_workflow.ps1 -Epoch 10 -BatchSize 4

.EXAMPLE
.\run_workflow.ps1 -LogDir "my_s3dis_test" -SkipTrain

#>

param(
    [string]$DataRoot = "data/stanford_indoor3d_minimal",
    [string]$RoomGlob = "Area_5_conferenceRoom_1*.npy",
    [string]$LogDir = "s3dis_minimal_run",
    [string]$Model = "pointnet_sem_seg",
    [int]$Epoch = 5,
    [int]$BatchSize = 2,
    [int]$NPoint = 2048,
    [double]$SampleRate = 0.1,
    [switch]$SkipTrain,
    [switch]$SkipTest,
    [switch]$SkipViz
)

# Get repo root
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$DataPath = Join-Path $RepoRoot $DataRoot
$LogPath = Join-Path $RepoRoot "log" "s3dis_minimal" $LogDir

# Banner
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  S3DIS MINIMAL COMPLETE WORKFLOW" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Data root:    $DataPath"
Write-Host "Room glob:    $RoomGlob"
Write-Host "Log dir:      $LogDir"
Write-Host "Model:        $Model"
Write-Host "Epochs:       $Epoch"
Write-Host "Batch size:   $BatchSize"
Write-Host "N points:     $NPoint"
Write-Host "Sample rate:  $SampleRate"
Write-Host ""

# Check data exists
if (-not (Test-Path $DataPath)) {
    Write-Host "❌ Data directory not found: $DataPath" -ForegroundColor Red
    Write-Host "Please place S3DIS room files (e.g., Area_5_conferenceRoom_1.npy) there." -ForegroundColor Yellow
    exit 1
}

# Check for room files
$RoomFiles = @(Get-ChildItem -Path $DataPath -Filter $RoomGlob -ErrorAction SilentlyContinue)
if ($RoomFiles.Count -eq 0) {
    Write-Host "⚠️  No room files found matching: $RoomGlob" -ForegroundColor Yellow
    Write-Host "Available files in $DataPath`:" -ForegroundColor Yellow
    Get-ChildItem -Path $DataPath | ForEach-Object { Write-Host "  - $($_.Name)" }
    exit 1
}

Write-Host "Found $($RoomFiles.Count) room file(s):" -ForegroundColor Green
$RoomFiles | ForEach-Object { Write-Host "  - $($_.Name)" }
Write-Host ""

# Step 1: Training
if (-not $SkipTrain) {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  STEP 1: TRAINING SEMANTIC SEGMENTATION MODEL" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    
    $TrainCmd = @(
        "python", "examples/s3dis_minimal/train_semseg_minimal.py",
        "--data_root", $DataRoot,
        "--room_glob", $RoomGlob,
        "--log_dir", $LogDir,
        "--model", $Model,
        "--batch_size", $BatchSize,
        "--npoint", $NPoint,
        "--epoch", $Epoch,
        "--sample_rate", $SampleRate
    )
    
    Push-Location $RepoRoot
    & $TrainCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Training failed!" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Pop-Location
    
    Write-Host ""
    Write-Host "✓ Training completed!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "⊘ Skipping training (--SkipTrain)" -ForegroundColor Yellow
    Write-Host ""
}

# Step 2: Testing
if (-not $SkipTest) {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  STEP 2: TESTING (Not yet implemented)" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "⊘ Testing step not yet available in minimal example" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "⊘ Skipping testing (--SkipTest)" -ForegroundColor Yellow
    Write-Host ""
}

# Step 3: Visualization
if (-not $SkipViz) {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  STEP 3: VISUALIZATION (Not yet implemented)" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "⊘ Visualization step not yet available in minimal example" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "⊘ Skipping visualization (--SkipViz)" -ForegroundColor Yellow
    Write-Host ""
}

# Final summary
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  ✓ WORKFLOW COMPLETE" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Logs and checkpoints saved to:" -ForegroundColor Green
Write-Host "  $LogPath"
Write-Host ""
Write-Host "Best model checkpoint:" -ForegroundColor Green
Write-Host "  $LogPath\checkpoints\best_model.pth"
Write-Host ""

