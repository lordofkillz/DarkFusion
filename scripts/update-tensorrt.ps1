[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [string]$CacheDirectory,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$pins = @(Get-Content -LiteralPath (Join-Path $repoRoot 'requirements.txt') | Where-Object {
    $_ -match '^tensorrt-cu(12|13)==[^;\s]+$'
})
if ($pins.Count -ne 1) { throw 'Expected one pinned CUDA-specific TensorRT family in requirements.txt.' }
$requirement = $pins[0]
$family,$version = $requirement -split '=='
$expected = @($family, "$family-bindings", "$family-libs")
$probe = @'
import importlib.metadata as m, json, re, sys
packages = {}
for dist in m.distributions():
    name = re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower()
    if name == 'tensorrt' or re.fullmatch(r'tensorrt-cu(?:12|13)(?:-bindings|-libs)?', name):
        packages[name] = dist.version
print(json.dumps(dict(executable=sys.executable, prefix=sys.prefix, packages=packages)))
'@
$metadata = & $PythonPath -I -c $probe | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $metadata.prefix) { throw 'Could not inspect TensorRT packages.' }
if ($CheckOnly) { $metadata | ConvertTo-Json -Depth 4; return }

function Assert-EnvironmentIdle {
    $jobs = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -ieq $PythonPath -or $_.ExecutablePath -ieq $metadata.executable
    })
    if ($jobs.Count) { throw "Close Python jobs before updating TensorRT. Running process IDs: $($jobs.ProcessId -join ', ')." }
}

$previous = @($metadata.packages.PSObject.Properties | ForEach-Object { "$($_.Name)==$($_.Value)" })
$names = @($metadata.packages.PSObject.Properties.Name)
$matching = $names.Count -eq 3
foreach ($name in $expected) {
    $item = $metadata.packages.PSObject.Properties[$name]
    if (-not $item -or $item.Value -ne $version) { $matching = $false }
}
$verify = @'
import importlib.metadata as m, sys
import tensorrt
family, expected = sys.argv[1:]
for name in (family, family + '-bindings', family + '-libs'):
    assert m.version(name) == expected, (name, m.version(name), expected)
assert tensorrt.__version__ == expected, (tensorrt.__version__, expected)
print('TensorRT ' + expected + ' verified (' + family + ').')
'@
Assert-EnvironmentIdle
if ($matching) {
    & $PythonPath -I -c $verify $family $version
    if ($LASTEXITCODE -eq 0) { return }
    Write-Warning 'TensorRT module needs repair despite matching package metadata.'
}
if (-not $CacheDirectory) { $CacheDirectory = Join-Path $repoRoot 'UltraDarkFusion\.darkfusion_cache\tensorrt-update' }
$operation = Join-Path $CacheDirectory ([Guid]::NewGuid().ToString('N'))
$release = Join-Path $operation 'release-wheels'
$recovery = Join-Path $operation 'recovery-wheels'
New-Item -ItemType Directory -Path $release,$recovery -Force | Out-Null
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $operation 'previous-runtime.json') -Encoding UTF8
$savedConfig = $env:PIP_CONFIG_FILE
$removed = $false
try {
    $env:PIP_CONFIG_FILE = 'nul'
    # Fetch the complete replacement and exact old packages before any removal.
    # TensorRT metapackages may be source distributions; pip wheel builds those
    # small wrappers while retaining prebuilt bindings/native libraries.
    & $PythonPath -I -m pip --isolated wheel --index-url https://pypi.org/simple --wheel-dir $release $requirement
    if ($LASTEXITCODE -ne 0) { throw 'TensorRT replacement download failed; no packages removed.' }
    if ($previous.Count) {
        & $PythonPath -I -m pip --isolated wheel --index-url https://pypi.org/simple --no-deps --wheel-dir $recovery @previous
        if ($LASTEXITCODE -ne 0) { throw 'TensorRT recovery download failed; no packages removed.' }
    }
    Assert-EnvironmentIdle
    if ($names.Count) {
        $removed = $true
        & $PythonPath -I -m pip --isolated uninstall --yes @names
        if ($LASTEXITCODE -ne 0) { throw 'Could not remove overlapping TensorRT packages.' }
    }
    & $PythonPath -I -m pip --isolated install --no-user --no-index --find-links $release $requirement
    if ($LASTEXITCODE -ne 0) { throw 'TensorRT installation failed.' }
    & $PythonPath -I -c $verify $family $version
    if ($LASTEXITCODE -ne 0) { throw 'TensorRT runtime verification failed.' }
    Write-Host "TensorRT updated. Recovery packages remain in $operation."
    Write-Host 'Engine files built with another TensorRT version may need re-exporting from the original model.'
}
catch {
    $failure = $_
    if ($removed -and $previous.Count) {
        Write-Warning 'Restoring the previous TensorRT packages after the failed update...'
        & $PythonPath -I -m pip --isolated uninstall --yes $family "$family-bindings" "$family-libs"
        & $PythonPath -I -m pip --isolated install --no-user --no-index --find-links $recovery --no-deps --force-reinstall @previous
        if ($LASTEXITCODE -ne 0) { Write-Warning "Automatic recovery failed. Recovery wheels: $recovery" }
    }
    throw $failure
}
finally { $env:PIP_CONFIG_FILE = $savedConfig }
