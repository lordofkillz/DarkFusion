[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$BuildDirectory)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)
$fixture = Join-Path $BuildDirectory ('smoke-fixtures\Setup O''Neil & $Budget; ' + [char]0x03A9 + ' ' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $fixture -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $BuildDirectory 'DarkFusionSetup.exe') -Destination $fixture
Set-Content -LiteralPath (Join-Path $fixture 'payload.zip') -Value 'disposable test payload'
Set-Content -LiteralPath (Join-Path $fixture 'payload.json') -Value '{}'
$backend = @'
param([string]$PackagePath, [string]$ManifestPath, [string]$InstallDirectory, [string]$LogPath, [switch]$DesktopShortcut, [switch]$StartMenuShortcut)
$ErrorActionPreference = 'Stop'
[ordered]@{PackagePath=$PackagePath; ManifestPath=$ManifestPath; InstallDirectory=$InstallDirectory; LogPath=$LogPath; Desktop=[bool]$DesktopShortcut; StartMenu=[bool]$StartMenuShortcut} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'captured.json') -Encoding UTF8
'Native fixture backend started.' | Set-Content -LiteralPath $LogPath
$exitFile = Join-Path $PSScriptRoot 'exit-code.txt'
if (Test-Path -LiteralPath $exitFile) { exit ([int](Get-Content -LiteralPath $exitFile)) }
exit 0
'@
Set-Content -LiteralPath (Join-Path $fixture 'install-standalone.ps1') -Value $backend -Encoding UTF8

function ConvertTo-WindowsArgument([string]$Value) {
    return '"' + (($Value -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
}
function Invoke-Setup([string[]]$Arguments) {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = Join-Path $fixture 'DarkFusionSetup.exe'
    $info.WorkingDirectory = $fixture
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.Arguments = ($Arguments | ForEach-Object { ConvertTo-WindowsArgument $_ }) -join ' '
    $process = [Diagnostics.Process]::Start($info)
    if (-not $process.WaitForExit(30000)) { $process.Kill(); throw 'Native setup fixture exceeded 30 seconds.' }
    $result = $process.ExitCode
    $process.Dispose()
    return $result
}

$destination = Join-Path $fixture 'Install O''Neil & $Budget; trailing folder\'
$exitCode = Invoke-Setup @('--quiet', '--install-dir', $destination)
if ($exitCode -ne 0) { throw "Quiet setup success fixture returned $exitCode." }
$captured = Get-Content -LiteralPath (Join-Path $fixture 'captured.json') -Raw | ConvertFrom-Json
if ($captured.InstallDirectory -cne $destination) { throw "Destination did not survive native-to-PowerShell argument parsing: $($captured.InstallDirectory)" }
if ($captured.PackagePath -cne (Join-Path $fixture 'payload.zip')) { throw 'Package path was changed.' }
if ($captured.ManifestPath -cne (Join-Path $fixture 'payload.json')) { throw 'Manifest path was changed.' }
if (-not $captured.StartMenu -or $captured.Desktop) { throw 'Quiet shortcut defaults are incorrect.' }
if (-not (Test-Path -LiteralPath $captured.LogPath -PathType Leaf)) { throw 'Explicit backend log path was not usable.' }
Set-Content -LiteralPath (Join-Path $fixture 'exit-code.txt') -Value '23'
if ((Invoke-Setup @('--quiet', '--install-dir', $destination)) -ne 23) { throw 'Backend failure exit code was not preserved.' }
if ((Invoke-Setup @('--quiet', '--unknown')) -ne 2) { throw 'Invalid CLI argument was accepted.' }
if ((Invoke-Setup @('--quiet', '--install-dir', 'relative-folder')) -ne 3) { throw 'Relative install directory was accepted.' }
Remove-Item -LiteralPath (Join-Path $fixture 'payload.zip')
if ((Invoke-Setup @('--quiet', '--install-dir', $destination)) -ne 3) { throw 'Incomplete installer files were accepted.' }
Write-Host 'PASS: native setup Unicode/special-character paths, quiet defaults, explicit log, backend failure, invalid arguments, relative paths, missing payload.'
Write-Host "Disposable fixture retained at $fixture"
