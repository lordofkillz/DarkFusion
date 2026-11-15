[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [string]$Version = 'latest',
    [string]$CacheDirectory,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$metadataProbe = @'
import importlib.metadata as m, json, sys
print(json.dumps(dict(version=m.version('ultralytics'), prefix=sys.prefix, executable=sys.executable)))
'@
$metadata = & $PythonPath -I -c $metadataProbe | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $metadata.version) { throw 'Could not inspect the selected Ultralytics installation.' }
$pins = @(Get-Content -LiteralPath (Join-Path $repoRoot 'requirements.txt') | Where-Object { $_ -match '^ultralytics==[^;\s]+$' })
if ($pins.Count -ne 1) { throw 'Expected one pinned Ultralytics version in requirements.txt.' }
$releaseVersion = ($pins[0] -split '==')[1]
$latestVersion = (Invoke-RestMethod -Uri 'https://pypi.org/pypi/ultralytics/json' -TimeoutSec 20).info.version
$targetVersion = if ($Version -eq 'latest') { $latestVersion } elseif ($Version -eq 'release') { $releaseVersion } else { $Version }
if ($targetVersion -notmatch '^\d+\.\d+\.\d+$') { throw 'Use latest, release, or a stable numeric Ultralytics version.' }
if ($CheckOnly) {
    [PSCustomObject]@{ Installed=$metadata.version; ReleasePin=$releaseVersion; Latest=$latestVersion; Target=$targetVersion } | ConvertTo-Json
    return
}

function Assert-EnvironmentIdle {
    $jobs = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -ieq $PythonPath -or $_.ExecutablePath -ieq $metadata.executable
    })
    if ($jobs.Count) { throw "Close DarkFusion and Python jobs before updating. Running process IDs: $($jobs.ProcessId -join ', '). No installed packages were changed." }
}
Assert-EnvironmentIdle
if ($metadata.version -eq $targetVersion) {
    Write-Host "Ultralytics $targetVersion is already installed. No update needed."
    return
}
if (-not $CacheDirectory) { $CacheDirectory = Join-Path $repoRoot 'UltraDarkFusion\.darkfusion_cache\ultralytics-update' }
$operation = Join-Path $CacheDirectory ([Guid]::NewGuid().ToString('N'))
$candidateWheels = Join-Path $operation 'candidate-wheels'
$recoveryWheels = Join-Path $operation 'recovery-wheels'
$candidate = Join-Path $operation 'candidate-package'
New-Item -ItemType Directory -Path $candidateWheels,$recoveryWheels,$candidate -Force | Out-Null
$metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $operation 'previous-version.json') -Encoding UTF8
$verifyFile = Join-Path $PSScriptRoot 'verify_install.py'
$stageProbe = @'
import runpy, sys
sys.path.insert(0, sys.argv[1])
script=sys.argv[2]
sys.argv=[script, '--ultralytics-compat']
runpy.run_path(script, run_name='__main__')
'@
$installedProbe = @'
import importlib.metadata as m, sys
assert m.version('ultralytics') == sys.argv[1]
'@
$changed = $false
try {
    # Prepare both wheels before touching the active package. Torch, CUDA,
    # TensorRT and Windows ML dependencies are never upgraded by this script.
    & $PythonPath -I -m pip --isolated download --index-url https://pypi.org/simple --only-binary=:all: --no-deps --dest $candidateWheels "ultralytics==$targetVersion"
    if ($LASTEXITCODE -ne 0) { throw 'Candidate download failed; installation unchanged.' }
    & $PythonPath -I -m pip --isolated download --index-url https://pypi.org/simple --only-binary=:all: --no-deps --dest $recoveryWheels "ultralytics==$($metadata.version)"
    if ($LASTEXITCODE -ne 0) { throw 'Recovery download failed; installation unchanged.' }
    & $PythonPath -I -m pip --isolated install --no-index --find-links $candidateWheels --no-deps --no-compile --target $candidate "ultralytics==$targetVersion"
    if ($LASTEXITCODE -ne 0) { throw 'Candidate staging failed; installation unchanged.' }
    & $PythonPath -I -c $stageProbe $candidate $verifyFile *> (Join-Path $operation 'candidate-compatibility.log')
    if ($LASTEXITCODE -ne 0) { throw "Candidate compatibility checks failed. Installation unchanged. See $operation\candidate-compatibility.log" }
    Assert-EnvironmentIdle
    $changed = $true
    & $PythonPath -I -m pip --isolated install --no-user --no-index --find-links $candidateWheels --no-deps --force-reinstall "ultralytics==$targetVersion"
    if ($LASTEXITCODE -ne 0) { throw 'Ultralytics installation failed.' }
    & $PythonPath -I -c $installedProbe $targetVersion
    if ($LASTEXITCODE -ne 0) { throw 'Installed version verification failed.' }
    & $PythonPath -I $verifyFile --ultralytics-compat *> (Join-Path $operation 'installed-compatibility.log')
    if ($LASTEXITCODE -ne 0) { throw 'Installed compatibility checks failed.' }
    & $PythonPath -I $verifyFile *> (Join-Path $operation 'installation-verification.log')
    if ($LASTEXITCODE -ne 0) { throw 'DarkFusion installation verification failed.' }
    Write-Host "Ultralytics $targetVersion installed and checked. Previous wheel and logs: $operation"
    Write-Host 'Complete real-model/GPU acceptance before changing the release pin in requirements.txt.'
}
catch {
    $failure = $_
    if ($changed) {
        Write-Warning 'Restoring the previous Ultralytics package after the failed update...'
        & $PythonPath -I -m pip --isolated install --no-user --no-index --find-links $recoveryWheels --no-deps --force-reinstall "ultralytics==$($metadata.version)"
        if ($LASTEXITCODE -ne 0) { Write-Warning "Recovery failed. Previous wheel remains in $recoveryWheels" }
        else {
            & $PythonPath -I -c $installedProbe $metadata.version
            if ($LASTEXITCODE -ne 0) { Write-Warning 'Recovery version could not be verified.' }
        }
    }
    throw $failure
}
