[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [string]$CacheDirectory,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$metadataProbe = @'
import importlib.metadata as m, json, sys
names = ('onnxruntime', 'onnxruntime-gpu', 'onnxruntime-directml', 'onnxruntime-openvino', 'onnxruntime-windowsml')
found = {}
for name in names:
    try: found[name] = m.version(name)
    except m.PackageNotFoundError: pass
dependencies = {}
for name in ('wasdk-Microsoft.Windows.AI.MachineLearning', 'wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap', 'winrt-runtime', 'numpy'):
    try: dependencies[name] = m.version(name)
    except m.PackageNotFoundError: pass
print(json.dumps(dict(prefix=sys.prefix, executable=sys.executable, packages=found, dependencies=dependencies)))
'@
$metadata = & $PythonPath -I -c $metadataProbe | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $metadata.prefix) { throw 'Could not inspect the selected Python environment.' }
$targetPython = [IO.Path]::GetFullPath([string]$metadata.executable)

function Assert-EnvironmentIdle {
    $jobs = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -ieq $targetPython -or $_.ExecutablePath -ieq $PythonPath
    })
    if ($jobs.Count) {
        throw "Close DarkFusion and finish training before updating this environment. Running Python process IDs: $($jobs.ProcessId -join ', '). No runtime packages were changed."
    }
}

$pins = Get-Content -LiteralPath (Join-Path $repoRoot 'requirements.txt')
$packageNames = @('onnxruntime-windowsml', 'wasdk-Microsoft.Windows.AI.MachineLearning[all]',
    'wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap', 'winrt-runtime', 'numpy')
$requirements = @($packageNames | ForEach-Object {
    $pattern = '^' + [regex]::Escape($_) + '==([^;\s]+)'
    $match = @($pins | Where-Object { $_ -match $pattern })
    if ($match.Count -ne 1) { throw "Expected one release requirement for $_." }
    [regex]::Match($match[0], $pattern).Value
})
$expectedVersion = $requirements[0].Split('=')[-1]
$previous = @($metadata.packages.PSObject.Properties | ForEach-Object { "$($_.Name)==$($_.Value)" })
$overlap = @($metadata.packages.PSObject.Properties | Where-Object { $_.Name -ne 'onnxruntime-windowsml' })
$dependenciesMatch = $true
foreach ($requirement in $requirements[1..($requirements.Count - 1)]) {
    $name,$version = $requirement -split '=='
    $name = $name -replace '\[.*\]',''
    $installedDependency = $metadata.dependencies.PSObject.Properties[$name]
    if (-not $installedDependency -or $installedDependency.Value -ne $version) { $dependenciesMatch = $false }
}
$verify = @'
import importlib.metadata as m, sys
sys.path.insert(0, sys.argv[1])
from darkfusion_windows_ml import get_ort
ort = get_ort()
assert m.version('onnxruntime-windowsml') == sys.argv[2]
for name in ('onnxruntime', 'onnxruntime-gpu', 'onnxruntime-directml', 'onnxruntime-openvino'):
    try: m.version(name)
    except m.PackageNotFoundError: continue
    raise RuntimeError(name + ' still overlaps Windows ML')
assert 'DmlExecutionProvider' in ort.get_available_providers()
assert ort.__version__ == '.'.join(sys.argv[2].split('.')[:3])
print('Windows ML and DirectML verified in the selected environment.')
'@
if ($CheckOnly) {
    $metadata | ConvertTo-Json -Depth 4
    return
}
Assert-EnvironmentIdle
if (-not $overlap.Count -and $dependenciesMatch -and $metadata.packages.'onnxruntime-windowsml' -eq $expectedVersion) {
    Write-Host 'The release Windows ML runtime is already installed without overlapping ORT packages.'
    & (Join-Path $PSScriptRoot 'install-windows-ml-runtime.ps1') -PythonPath $PythonPath -CacheDirectory $CacheDirectory
    & $PythonPath -I -c $verify (Join-Path $repoRoot 'UltraDarkFusion') $expectedVersion
    if ($LASTEXITCODE -eq 0) { return }
    Write-Warning 'The installed ORT module needs repair; reinstalling the release package.'
}
if (-not $CacheDirectory) {
    $CacheDirectory = Join-Path $repoRoot 'UltraDarkFusion\.darkfusion_cache\windows-ml-update'
}
$operationDirectory = Join-Path $CacheDirectory ([Guid]::NewGuid().ToString('N'))
$newWheels = Join-Path $operationDirectory 'release-wheels'
$oldWheels = Join-Path $operationDirectory 'restore-wheels'
New-Item -ItemType Directory -Path $newWheels,$oldWheels -Force | Out-Null
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $operationDirectory 'previous-runtime.json') -Encoding UTF8
$savedPipConfig = $env:PIP_CONFIG_FILE
$savedNoUser = $env:PYTHONNOUSERSITE
$packagesRemoved = $false
try {
    $env:PIP_CONFIG_FILE = 'nul'
    $env:PYTHONNOUSERSITE = '1'
    # Download and resolve both the replacement and recovery packages before
    # removing anything. This update does not reinstall Torch or training tools.
    & $PythonPath -I -m pip --isolated download --index-url https://pypi.org/simple --cache-dir (Join-Path $operationDirectory 'http-cache') --only-binary=:all: --dest $newWheels @requirements
    if ($LASTEXITCODE -ne 0) { throw 'Windows ML download failed; the original runtime was not changed.' }
    if ($previous.Count) {
        & $PythonPath -I -m pip --isolated download --index-url https://pypi.org/simple --cache-dir (Join-Path $operationDirectory 'http-cache') --only-binary=:all: --no-deps --dest $oldWheels @previous
        if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the recovery packages; the original runtime was not changed.' }
    }
    Assert-EnvironmentIdle
    if ($previous.Count) {
        $packagesRemoved = $true
        $variants = @($metadata.packages.PSObject.Properties.Name)
        & $PythonPath -I -m pip --isolated uninstall --yes @variants
        if ($LASTEXITCODE -ne 0) { throw 'Could not remove the old ONNX Runtime package.' }
    }
    & $PythonPath -I -m pip --isolated install --no-user --no-index --find-links $newWheels @requirements
    if ($LASTEXITCODE -ne 0) { throw 'Windows ML installation failed.' }
    & (Join-Path $PSScriptRoot 'install-windows-ml-runtime.ps1') -PythonPath $PythonPath -CacheDirectory $operationDirectory
    & $PythonPath -I -c $verify (Join-Path $repoRoot 'UltraDarkFusion') $expectedVersion
    if ($LASTEXITCODE -ne 0) { throw 'Windows ML verification failed.' }
    Write-Host "ONNX Runtime was replaced in $($metadata.prefix). Recovery packages remain in $operationDirectory."
}
catch {
    $failure = $_
    if ($packagesRemoved -and $previous.Count) {
        Write-Warning 'Restoring the previous ONNX Runtime package after the failed update...'
        & $PythonPath -I -m pip --isolated uninstall --yes onnxruntime-windowsml
        & $PythonPath -I -m pip --isolated install --no-user --no-index --find-links $oldWheels --no-deps --force-reinstall @previous
        if ($LASTEXITCODE -ne 0) { Write-Warning "Automatic recovery failed. Recovery wheels are preserved at $oldWheels." }
    }
    throw $failure
}
finally {
    $env:PIP_CONFIG_FILE = $savedPipConfig
    $env:PYTHONNOUSERSITE = $savedNoUser
}
