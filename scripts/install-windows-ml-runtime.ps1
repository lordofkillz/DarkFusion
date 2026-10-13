[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [string]$CacheDirectory
)

$ErrorActionPreference = 'Stop'
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
if (-not $CacheDirectory) {
    $CacheDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) 'UltraDarkFusion\.darkfusion_cache\windows-ml-setup'
}
# Windows PowerShell 5.1 strips embedded double quotes from native arguments.
# Python single quotes survive both 5.1 and PowerShell 7 argument passing.
$probe = 'from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import initialize; runtime = initialize(); runtime(); print(''Windows App SDK runtime is ready.'')'
$probeReady = $false
try {
    & $PythonPath -I -c $probe *> $null
    $probeReady = $LASTEXITCODE -eq 0
}
catch { $probeReady = $false }
if ($probeReady) { return }

Write-Host 'Installing the Microsoft Windows App SDK runtime required by Windows ML...'
New-Item -ItemType Directory -Path $CacheDirectory -Force | Out-Null
$installerPath = Join-Path $CacheDirectory ('WindowsAppRuntimeInstall-x64-' + [Guid]::NewGuid().ToString('N') + '.exe')
# Must match the wasdk 2.3 projections pinned in requirements.txt. The signed
# Microsoft installer deploys the shared runtime, not Python packages/drivers.
$download = 'https://aka.ms/windowsappsdk/2.3/latest/windowsappruntimeinstall-x64.exe'
Invoke-WebRequest -Uri $download -OutFile $installerPath -UseBasicParsing
# A native setup launched from PowerShell 7 can inherit its PSModulePath while
# running Windows PowerShell 5.1. Load this shell's built-in security module,
# rather than an incompatible module selected from the parent's search path.
$securityModule = Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1'
Import-Module -Name $securityModule -ErrorAction Stop
$signature = Microsoft.PowerShell.Security\Get-AuthenticodeSignature -LiteralPath $installerPath
if ($signature.Status -ne 'Valid' -or -not $signature.SignerCertificate -or
    $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation(?:,|$)') {
    throw 'The Windows App SDK installer does not have a valid Microsoft signature.'
}
$installProcess = Start-Process -FilePath $installerPath -ArgumentList '--quiet' -WindowStyle Hidden -Wait -PassThru
if ($installProcess.ExitCode -notin @(0,3010)) {
    throw "Windows App SDK installation failed (exit $($installProcess.ExitCode))."
}
& $PythonPath -I -c $probe
if ($LASTEXITCODE -ne 0) {
    throw 'The matching Windows App SDK runtime is not ready. Restart Windows if setup requested it, then retry.'
}
