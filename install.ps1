[CmdletBinding()]
param(
    [string]$EnvironmentName = "fusion",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$requirementsPath = Join-Path $repoRoot "requirements.txt"
$verifyScript = Join-Path $repoRoot "scripts\verify_install.py"

function Find-Conda {
    $command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    throw "Conda was not found. Install Miniconda or Anaconda, then run this installer again."
}

if (-not (Test-Path -LiteralPath $requirementsPath)) {
    throw "requirements.txt was not found at $requirementsPath"
}

$conda = Find-Conda
$environmentList = & $conda env list --json | ConvertFrom-Json
$environmentPath = $environmentList.envs |
    Where-Object { (Split-Path -Leaf $_) -ieq $EnvironmentName } |
    Select-Object -First 1

if (-not $environmentPath) {
    Write-Host "Creating conda environment '$EnvironmentName' with Python 3.12..."
    & $conda create --name $EnvironmentName "python=3.12" --yes
    if ($LASTEXITCODE -ne 0) {
        throw "Conda could not create the '$EnvironmentName' environment."
    }
}

$pythonArgs = @("run", "--name", $EnvironmentName, "--no-capture-output", "python")
$env:PYTHONNOUSERSITE = "1"

Write-Host "Upgrading pip build tools..."
& $conda @pythonArgs -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip build tools."
}

Write-Host "Installing UltraDarkFusion dependencies..."
& $conda @pythonArgs -m pip install --no-user --requirement $requirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

Write-Host "Verifying the application environment..."
& $conda @pythonArgs $verifyScript
if ($LASTEXITCODE -ne 0) {
    throw "Installation verification failed."
}

Write-Host ""
Write-Host "UltraDarkFusion installation completed."
Write-Host "Launch it with run_darkfusion.bat."
if (-not $SkipModels) {
    Write-Host "Optional SAM3 and GroundingDINO files are described in MODEL_SETUP.md."
}
