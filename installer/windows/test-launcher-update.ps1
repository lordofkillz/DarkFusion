[CmdletBinding()]
param(
    [string]$LauncherPath,
    [string]$OutputDirectory = (Join-Path $env:TEMP 'DarkFusion-launcher-tests')
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($LauncherPath)) { $LauncherPath = Join-Path $PSScriptRoot '../native/build/DarkFusion.exe' }
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$backend = Join-Path $PSScriptRoot 'install-standalone.ps1'
$errors = $null
$tree = [Management.Automation.Language.Parser]::ParseFile($backend, [ref]$null, [ref]$errors)
if ($errors.Count) { throw 'The installer backend does not parse.' }
foreach ($definition in $tree.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }, $false)) {
    . ([scriptblock]::Create($definition.Extent.Text))
}
$root = [IO.Path]::GetFullPath((Join-Path $OutputDirectory ('Launcher fixtures ' + [char]0x03a9 + ' ' + [guid]::NewGuid().ToString('N'))))
[void][IO.Directory]::CreateDirectory($root)
$encoding = New-Object Text.UTF8Encoding($false)
$script:InstallLog = Join-Path $root 'test.log'
$script:ExtractionStarted = $false
$results = @()

function Get-TestHash([byte[]]$Bytes) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($sha.ComputeHash($Bytes)).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Pass([string]$Name) {
    $script:results += [pscustomobject]@{ test = $Name; passed = $true }
    Write-Host "PASS: $Name"
}

$originalBytes = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $LauncherPath).ProviderPath)
$sourceDirectory = Join-Path $root 'Verified setup files'
[void][IO.Directory]::CreateDirectory($sourceDirectory)
$validLauncher = Join-Path $sourceDirectory 'DarkFusion.exe'
[IO.File]::WriteAllBytes($validLauncher, $originalBytes)
$launcher = Read-LauncherUpdate $validLauncher
if ($launcher.Hash -cne (Get-TestHash $originalBytes)) { throw 'The launcher snapshot hash differs.' }
$script:Destination = Join-Path $root 'Snapshot App'
[void][IO.Directory]::CreateDirectory($script:Destination)
$target = Join-Path $script:Destination 'DarkFusion.exe'
[IO.File]::WriteAllText($target, 'older extracted launcher', $encoding)
$failure = $null
try { Install-LauncherUpdate $launcher } catch { $failure = $_.Exception.Message }
if (-not $failure -or [IO.File]::ReadAllText($target) -cne 'older extracted launcher') {
    throw 'Launcher replacement was allowed before extraction.'
}
Pass 'replacement requires a newly extracted installation'

# Changing the input after validation cannot change the bytes installed.
[IO.File]::WriteAllText($validLauncher, 'changed source', $encoding)
$script:ExtractionStarted = $true
Install-LauncherUpdate $launcher
if ((Get-TestHash ([IO.File]::ReadAllBytes($target))) -cne $launcher.Hash) { throw 'The launcher snapshot was not preserved.' }
Pass 'validated launcher snapshot installed exactly'
[IO.File]::WriteAllBytes($validLauncher, $originalBytes)

# Build an inert runtime process that accepts every argument and returns success.
# This tests backend sequencing/state without starting Python or the application.
$stub = Join-Path $root 'runtime-fixture.exe'
Add-Type -TypeDefinition 'public static class DarkFusionLauncherFixtureRuntime { public static int Main(string[] args) { return 0; } }' `
    -OutputAssembly $stub -OutputType ConsoleApplication
$oldLauncherBytes = $encoding.GetBytes('original archive launcher')
$entries = @(
    [pscustomobject]@{ Name = 'runtime/python.exe'; Bytes = [IO.File]::ReadAllBytes($stub) },
    [pscustomobject]@{ Name = 'runtime/Scripts/conda-unpack-script.py'; Bytes = $encoding.GetBytes('# inert relocation fixture') },
    [pscustomobject]@{ Name = 'app/scripts/verify_install.py'; Bytes = $encoding.GetBytes('# inert verifier fixture') },
    [pscustomobject]@{ Name = 'DarkFusion.exe'; Bytes = $oldLauncherBytes }
)
$package = Join-Path $root 'runtime-fixture.zip'
$stream = [IO.File]::Open($package, [IO.FileMode]::CreateNew)
$archive = New-Object IO.Compression.ZipArchive($stream, [IO.Compression.ZipArchiveMode]::Create, $true)
[long]$unpacked = 0
try {
    foreach ($item in $entries) {
        $entry = $archive.CreateEntry($item.Name)
        $output = $entry.Open()
        try { $output.Write($item.Bytes, 0, $item.Bytes.Length) } finally { $output.Dispose() }
        $unpacked += $item.Bytes.Length
    }
}
finally { $archive.Dispose(); $stream.Dispose() }
$packageHash = Get-TestHash ([IO.File]::ReadAllBytes($package))
$manifest = [ordered]@{
    schema_version = 1; product = 'DarkFusion'; version = '5.2.0'; source_commit = '0' * 40
    archive_size_bytes = ([IO.FileInfo]$package).Length; unpacked_size_bytes = $unpacked; sha256 = $packageHash
}
$manifestPath = Join-Path $root 'runtime-fixture.json'
[IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json), $encoding)
$badLauncher = Join-Path $root 'invalid-launcher.exe'
[IO.File]::WriteAllBytes($badLauncher, (New-Object byte[] 128))
$badPe = Join-Path $root 'invalid-pe.exe'
$badPeBytes = [byte[]]$originalBytes.Clone()
$badPeBytes[[BitConverter]::ToInt32($badPeBytes, 60)] = 0
[IO.File]::WriteAllBytes($badPe, $badPeBytes)
$linkedDirectory = Join-Path $root 'Linked setup files'
New-Item -ItemType Junction -Path $linkedDirectory -Target $sourceDirectory | Out-Null
$powershell = Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'
foreach ($case in @('valid', 'offline', 'missing', 'linked', 'invalid', 'invalid_pe', 'relative', 'inside_destination')) {
    $destination = Join-Path $root ($case + ' App ' + [char]0x00e9 + [char]0x03a9)
    $log = Join-Path $root ($case + '.log')
    $parameters = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $backend,
        '-PackagePath', $package, '-ManifestPath', $manifestPath, '-InstallDirectory', $destination, '-LogPath', $log)
    $candidate = switch ($case) {
        'valid' { $validLauncher }
        'missing' { Join-Path $root 'missing.exe' }
        'linked' { Join-Path $linkedDirectory 'DarkFusion.exe' }
        'invalid' { $badLauncher }
        'invalid_pe' { $badPe }
        'relative' { 'relative.exe' }
        'inside_destination' { Join-Path $destination 'DarkFusion.exe' }
    }
    if ($case -ne 'offline') { $parameters += @('-LauncherPath', $candidate) }
    if ($case -notin @('valid', 'offline')) { $parameters += @('-DesktopShortcut', '-StartMenuShortcut') }
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $backendOutput = & $powershell @parameters 2>&1
        $backendExit = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $oldPreference }
    $logText = if (Test-Path -LiteralPath $log) { [IO.File]::ReadAllText($log) } else { [string]::Join([Environment]::NewLine, $backendOutput) }
    $statePath = Join-Path $destination 'install-state.json'
    if ($case -in @('valid', 'offline')) {
        if ($backendExit -ne 0 -or -not (Test-Path -LiteralPath $statePath)) { throw "Backend $case failed: $logText" }
        $state = [IO.File]::ReadAllText($statePath) | ConvertFrom-Json
        if ($state.status -cne 'complete' -or $state.package_sha256 -cne $packageHash) { throw 'Installation state lost runtime provenance.' }
        $installedBytes = [IO.File]::ReadAllBytes((Join-Path $destination 'DarkFusion.exe'))
        if ($case -eq 'valid') {
            if ($state.launcher_sha256 -cne $launcher.Hash -or (Get-TestHash $installedBytes) -cne $launcher.Hash) {
                throw 'Backend did not install and record the new launcher.'
            }
        }
        elseif ($state.PSObject.Properties['launcher_sha256'] -or (Get-TestHash $installedBytes) -cne (Get-TestHash $oldLauncherBytes)) {
            throw 'Offline installation unexpectedly changed the archive launcher.'
        }
    }
    else {
        if ($backendExit -ne 1 -or (Test-Path -LiteralPath $destination) -or
            $logText.Contains('SUCCESS:') -or $logText.Contains('Created shortcut:')) {
            throw "Backend $case did not fail before installation/shortcuts: $logText"
        }
        $expected = switch ($case) {
            'missing' { 'Setup file is missing' }
            'linked' { 'Linked folders or files are not supported' }
            'invalid' { 'not a valid Windows executable' }
            'invalid_pe' { 'not a valid 64-bit Windows executable' }
            'relative' { 'Launcher path must be an absolute path' }
            'inside_destination' { 'separate from the setup package' }
        }
        if (-not $logText.Contains($expected)) { throw "Backend $case failed for an unexpected reason: $logText" }
    }
    Pass "backend $case"
}
[IO.File]::WriteAllText((Join-Path $root 'results.json'), ($results | ConvertTo-Json), $encoding)
Write-Host "Disposable fixtures retained at $root"
