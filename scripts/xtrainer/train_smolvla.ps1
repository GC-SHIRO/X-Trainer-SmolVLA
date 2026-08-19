[CmdletBinding()]
param(
    [string]$DatasetRoot,
    [string]$OutputDir,
    [string]$Device,
    [Nullable[int]]$BatchSize,
    [Nullable[int]]$Steps,
    [string]$ResumeCheckpoint,
    [switch]$SkipValidation,
    [Alias("h", "help")]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
Usage: .\scripts\xtrainer\train_smolvla.ps1 -DatasetRoot PATH [options]

Train SmolVLA with configs/xtrainer/train_smolvla.yaml and the standard
lerobot-train loop. The v2.1 dataset is validated before training by default.

Required:
  -DatasetRoot PATH          Local LeRobot Dataset v2.1 directory.

Options:
  -OutputDir PATH            Override the training output directory.
  -Device DEVICE             Override policy.device (for example: cuda).
  -BatchSize N               Override batch_size.
  -Steps N                   Override steps.
  -ResumeCheckpoint PATH     Resume from a checkpoint train_config.json or
                             pretrained_model directory. Its saved config is used.
  -SkipValidation            Do not run validate_dataset_v21.py before training.
  -Help, --help              Show this help message.
"@ | Write-Output
}

if ($RemainingArgs) {
    if ($RemainingArgs.Count -eq 1 -and $RemainingArgs[0] -eq "--help") {
        $Help = $true
    }
    else {
        throw "Unknown argument(s): $($RemainingArgs -join ' ')"
    }
}
if ($Help) {
    Show-Usage
    exit 0
}

if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    Write-Error "-DatasetRoot is required."
    Show-Usage
    exit 2
}
if (-not (Test-Path -LiteralPath $DatasetRoot -PathType Container)) {
    Write-Error "Dataset root does not exist or is not a directory: $DatasetRoot"
    exit 2
}

$DatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$TrainConfig = Join-Path $RepoRoot "configs/xtrainer/train_smolvla.yaml"
$Validator = Join-Path $PSScriptRoot "validate_dataset_v21.py"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "python was not found; activate the LeRobot environment first."
    exit 127
}
if (-not (Get-Command lerobot-train -ErrorAction SilentlyContinue)) {
    Write-Error "lerobot-train was not found; install/activate LeRobot first."
    exit 127
}

if (-not $SkipValidation) {
    & python $Validator --root $DatasetRoot
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$TrainArgs = @()
if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    $TrainArgs += "--config_path=$TrainConfig"
}
else {
    $TrainArgs += "--resume=true"
    $TrainArgs += "--config_path=$ResumeCheckpoint"
}
$TrainArgs += "--dataset.root=$DatasetRoot"

if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $TrainArgs += "--output_dir=$OutputDir"
}
if (-not [string]::IsNullOrWhiteSpace($Device)) {
    $TrainArgs += "--policy.device=$Device"
}
if ($null -ne $BatchSize) {
    $TrainArgs += "--batch_size=$BatchSize"
}
if ($null -ne $Steps) {
    $TrainArgs += "--steps=$Steps"
}

& lerobot-train @TrainArgs
exit $LASTEXITCODE
