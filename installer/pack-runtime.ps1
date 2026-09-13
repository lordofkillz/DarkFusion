[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$EnvironmentPath,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [string]$BuildPython = "python"
)

$ErrorActionPreference = "Stop"
$environmentRoot = (Resolve-Path -LiteralPath $EnvironmentPath).Path
$runtimePython = Join-Path $environmentRoot "python.exe"
if (-not (Test-Path -LiteralPath $runtimePython -PathType Leaf)) {
    throw "EnvironmentPath must point to a complete Windows Python environment."
}
if (Test-Path -LiteralPath $OutputPath) {
    throw "OutputPath already exists. Choose a new archive path."
}
$archivePath = [IO.Path]::GetFullPath($OutputPath)
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($archivePath)) | Out-Null

& $runtimePython -I -m pip check
if ($LASTEXITCODE -ne 0) { throw "The source environment has conflicting dependencies." }
& $BuildPython -I -m conda_pack.cli --prefix $environmentRoot --output $archivePath --format zip --arcroot runtime --compress-level 1
if ($LASTEXITCODE -ne 0) { throw "Runtime packaging failed. See the conda-pack error above." }
Write-Host "Runtime package created: $archivePath"
