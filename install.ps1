[CmdletBinding()]
param(
    [string]$EnvironmentName = "fusion",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$requirementsPath = Join-Path $repoRoot "requirements.txt"
$verifyScript = Join-Path $repoRoot "scripts\verify_install.py"
$modelBundleUrl = "https://drive.google.com/file/d/1j9Y-WpUDjPt67_U43lafO-7dTkxLJuPS/view?usp=sharing"

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
    Write-Host "Creating conda environment '$EnvironmentName' with security-patched Python 3.12..."
    & $conda create --name $EnvironmentName "python=3.12.15" --yes
    if ($LASTEXITCODE -ne 0) {
        throw "Conda could not create the '$EnvironmentName' environment."
    }
}

$pythonArgs = @("run", "--name", $EnvironmentName, "--no-capture-output", "python")
$env:PYTHONNOUSERSITE = "1"
if ($environmentPath) {
    $activePython = Join-Path $environmentPath 'python.exe'
    $activeProcesses = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -eq $activePython
    })
    if ($activeProcesses.Count -gt 0) {
        throw "The '$EnvironmentName' environment has running Python jobs. Close DarkFusion and finish training, then rerun this installer to update the same environment safely."
    }
}

$previousPipConfigFile = $env:PIP_CONFIG_FILE
try {
    # --isolated skips user settings, but still reads machine-wide pip.ini.
    # Use only this repository's indexes, including in build subprocesses.
    $env:PIP_CONFIG_FILE = "nul"

    $privatePython = (& $conda @pythonArgs -c 'import sys; print(sys.executable)' | Select-Object -Last 1).Trim()
    & (Join-Path $repoRoot 'scripts\update-windows-ml.ps1') -PythonPath $privatePython

    Write-Host "Upgrading pip build tools..."
    & $conda @pythonArgs -m pip --isolated install --index-url https://pypi.org/simple --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Could not upgrade pip build tools."
    }

    Write-Host "Installing UltraDarkFusion dependencies..."
    & $conda @pythonArgs -m pip --isolated install --index-url https://pypi.org/simple --no-user --requirement $requirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }
}
finally {
    $env:PIP_CONFIG_FILE = $previousPipConfigFile
}

Write-Host "Verifying the application environment..."
$privatePython = (& $conda @pythonArgs -c 'import sys; print(sys.executable)' | Select-Object -Last 1).Trim()
& (Join-Path $repoRoot 'scripts\install-windows-ml-runtime.ps1') -PythonPath $privatePython
& $conda @pythonArgs $verifyScript
if ($LASTEXITCODE -ne 0) {
    throw "Installation verification failed."
}

Write-Host ""
Write-Host "UltraDarkFusion installation completed."
Write-Host "Launch it with run_darkfusion.bat."
if (-not $SkipModels) {
    Write-Host ""
    Write-Host "Required for complete SAM3 and GroundingDINO functionality:"
    Write-Host $modelBundleUrl
    Write-Host "Extract the downloaded Sam folder into:"
    Write-Host (Join-Path $repoRoot "UltraDarkFusion")
    Write-Host "See MODEL_SETUP.md for file paths and the SHA-256 checksum."
}
