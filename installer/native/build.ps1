[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'build'),
    [switch]$RunTests
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

# Import the selected Visual Studio toolchain only into this build process.
# No global environment variables, Python environments, or user settings change.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw 'Install Visual Studio 2022 Build Tools with Desktop development with C++ (MSVC and Windows SDK).'
}
$installationPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installationPath) { throw 'No MSVC x64 toolchain was found. Install Desktop development with C++.' }
$developerCommand = Join-Path $installationPath 'Common7\Tools\VsDevCmd.bat'
$toolchainInfo = New-Object Diagnostics.ProcessStartInfo
$toolchainInfo.FileName = $env:ComSpec
$toolchainInfo.Arguments = '/d /s /c ""{0}" -no_logo -arch=amd64 -host_arch=amd64 >nul && set"' -f $developerCommand
$toolchainInfo.UseShellExecute = $false
$toolchainInfo.CreateNoWindow = $true
$toolchainInfo.RedirectStandardOutput = $true
$toolchainProcess = [Diagnostics.Process]::Start($toolchainInfo)
$developerEnvironment = $toolchainProcess.StandardOutput.ReadToEnd() -split '\r?\n'
$toolchainProcess.WaitForExit()
$toolchainExitCode = $toolchainProcess.ExitCode
$toolchainProcess.Dispose()
if ($toolchainExitCode -ne 0) { throw 'Visual Studio toolchain initialization failed.' }
foreach ($line in $developerEnvironment) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
}

$compiler = (Get-Command cl.exe -ErrorAction Stop).Source
$common = @('/nologo', '/std:c++17', '/EHsc', '/O2', '/MT', '/W4', '/utf-8', '/DUNICODE', '/D_UNICODE', '/D_WIN32_WINNT=0x0A00')
$libraries = @('user32.lib', 'gdi32.lib', 'shell32.lib', 'ole32.lib', 'comctl32.lib', 'uuid.lib')
$manifest = Join-Path $PSScriptRoot 'app.manifest'

Push-Location $PSScriptRoot
try {
    foreach ($target in @(@{ Source = 'setup.cpp'; Name = 'DarkFusionSetup' }, @{ Source = 'launcher.cpp'; Name = 'DarkFusion' })) {
        $exe = Join-Path $OutputDirectory ($target.Name + '.exe')
        $obj = Join-Path $OutputDirectory ($target.Name + '.obj')
        & $compiler @common $target.Source "/Fo$obj" "/Fe$exe" /link /SUBSYSTEM:WINDOWS /MANIFEST:EMBED "/MANIFESTINPUT:$manifest" @libraries
        if ($LASTEXITCODE -ne 0) { throw "Native build failed: $($target.Name)" }
    }
    if ($RunTests) {
        $testExe = Join-Path $OutputDirectory 'native smoke.exe'
        & $compiler @common 'native-smoke.cpp' "/Fo$(Join-Path $OutputDirectory 'native-smoke.obj')" "/Fe$testExe" /link /SUBSYSTEM:CONSOLE @libraries
        if ($LASTEXITCODE -ne 0) { throw 'Native smoke test build failed.' }
        & $testExe
        if ($LASTEXITCODE -ne 0) { throw 'Native argument/environment smoke test failed.' }
        & (Join-Path $PSScriptRoot 'test-setup.ps1') -BuildDirectory $OutputDirectory
    }
} finally {
    Pop-Location
}
Write-Host "Built DarkFusionSetup.exe and DarkFusion.exe in $OutputDirectory"
