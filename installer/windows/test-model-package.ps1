[CmdletBinding()]
param([string]$OutputDirectory = (Join-Path $env:TEMP 'DarkFusion-model-tests'))

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$scriptFile = Join-Path $PSScriptRoot 'install-standalone.ps1'
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($scriptFile, [ref]$null, [ref]$errors)
if ($errors.Count) { throw 'The installer backend did not parse.' }
$functions = @('Write-InstallLog', 'Test-PathWithin', 'Assert-NoReparseAncestors',
    'Ensure-ExtractionDirectory', 'Get-ManifestInteger', 'Open-ModelPackage', 'Install-ModelFiles')
foreach ($definition in $ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    if ($definition.Name -in $functions) { . ([scriptblock]::Create($definition.Extent.Text)) }
}
$root = [IO.Path]::GetFullPath((Join-Path $OutputDirectory ('Model fixtures ' + [char]0x03A9 + ' ' + [guid]::NewGuid().ToString('N'))))
[void][IO.Directory]::CreateDirectory($root)
# Retain these small, uniquely named fixtures for review; never remove user files.
$script:InstallLog = Join-Path $root 'test.log'
$encoding = New-Object Text.UTF8Encoding($false)

function Get-TestHash([byte[]]$Data) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($sha.ComputeHash($Data)).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function New-TestPackage([string]$Case, [string]$Directory) {
    $samBytes = $encoding.GetBytes('SAM checkpoint fixture')
    $dinoBytes = $encoding.GetBytes('GroundingDINO checkpoint fixture')
    $entries = @(
        [pscustomobject]@{ Name = 'Sam/sam3.pt'; Bytes = $samBytes },
        [pscustomobject]@{ Name = 'Sam/groundingdino_swint_ogc.pth'; Bytes = $dinoBytes },
        [pscustomobject]@{ Name = 'Sam/config.json'; Bytes = $encoding.GetBytes('bundle support file must not replace runtime support file') }
    )
    if ($Case -eq 'duplicate') { $entries += $entries[0] }
    if ($Case -eq 'traversal') { $entries += [pscustomobject]@{ Name = '../escape.txt'; Bytes = $samBytes } }
    if ($Case -eq 'missing') { $entries = $entries | Where-Object { $_.Name -ne 'Sam/sam3.pt' } }
    $packageFile = Join-Path $Directory 'models.zip'
    $stream = [IO.File]::Open($packageFile, [IO.FileMode]::CreateNew)
    $zip = New-Object IO.Compression.ZipArchive($stream, [IO.Compression.ZipArchiveMode]::Create, $true)
    try {
        foreach ($item in $entries) {
            $entry = $zip.CreateEntry($item.Name)
            if ($Case -eq 'symlink' -and $item.Name -eq 'Sam/sam3.pt') { $entry.ExternalAttributes = -1610612736 }
            $output = $entry.Open()
            try { $output.Write($item.Bytes, 0, $item.Bytes.Length) } finally { $output.Dispose() }
        }
    }
    finally { $zip.Dispose(); $stream.Dispose() }
    $manifest = [ordered]@{
        name = 'DarkFusion-models.zip'
        size_bytes = ([IO.FileInfo]$packageFile).Length
        sha256 = Get-TestHash ([IO.File]::ReadAllBytes($packageFile))
        unpacked_size_bytes = $samBytes.Length + $dinoBytes.Length
        files = @(
            [ordered]@{ archive_path = 'Sam/sam3.pt'; target_path = 'app/UltraDarkFusion/Sam/sam3.pt'; size_bytes = $samBytes.Length; sha256 = Get-TestHash $samBytes },
            [ordered]@{ archive_path = 'Sam/groundingdino_swint_ogc.pth'; target_path = 'app/UltraDarkFusion/Sam/groundingdino_swint_ogc.pth'; size_bytes = $dinoBytes.Length; sha256 = Get-TestHash $dinoBytes }
        )
    }
    if ($Case -eq 'archive_hash') { $manifest.sha256 = '0' * 64 }
    if ($Case -eq 'model_hash') { $manifest.files[0].sha256 = '0' * 64 }
    if ($Case -eq 'target') { $manifest.files[0].target_path = '../escape.pt' }
    $manifestFile = Join-Path $Directory 'models.json'
    [IO.File]::WriteAllText($manifestFile, ($manifest | ConvertTo-Json -Depth 5), $encoding)
    return [pscustomobject]@{ Package = $packageFile; Manifest = $manifestFile; Data = $manifest }
}

$results = @()
foreach ($case in @('success', 'archive_hash', 'model_hash', 'duplicate', 'traversal', 'symlink', 'missing', 'target', 'runtime_collision')) {
    $caseRoot = Join-Path $root $case
    [void][IO.Directory]::CreateDirectory($caseRoot)
    $fixture = New-TestPackage $case $caseRoot
    $script:Destination = Join-Path $caseRoot 'App'
    $support = Join-Path $script:Destination 'app/UltraDarkFusion/Sam/config.json'
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($support))
    [IO.File]::WriteAllText($support, 'original runtime support', $encoding)
    $script:ValidatedDirectories = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    [void]$script:ValidatedDirectories.Add($script:Destination)
    $runtimeNames = New-Object 'System.Collections.Generic.Dictionary[string,bool]' ([StringComparer]::OrdinalIgnoreCase)
    if ($case -eq 'runtime_collision') { $runtimeNames.Add('app/UltraDarkFusion/Sam/sam3.pt', $false) }
    $package = $null
    $failure = $null
    try {
        $package = Open-ModelPackage $fixture.Package $fixture.Manifest $runtimeNames
        Install-ModelFiles $package
    }
    catch { $failure = $_.Exception.Message }
    finally { if ($package) { $package.Archive.Dispose(); $package.Stream.Dispose() } }
    if ($case -eq 'success') {
        if ($failure) { throw "Valid model package failed: $failure" }
        foreach ($file in $fixture.Data.files) {
            $installed = Join-Path $script:Destination $file.target_path
            if ((Get-TestHash ([IO.File]::ReadAllBytes($installed))) -cne $file.sha256) { throw 'Installed checkpoint content differs from the manifest.' }
        }
    }
    elseif (-not $failure) { throw "The invalid $case fixture was accepted." }
    if ([IO.File]::ReadAllText($support) -cne 'original runtime support') { throw 'Runtime support files were overwritten.' }
    if (Test-Path -LiteralPath (Join-Path $caseRoot 'escape.txt')) { throw 'An archive entry escaped the installation folder.' }
    $results += [pscustomobject]@{ test = $case; passed = $true; expected_failure = $failure }
    Write-Host "PASS: $case"
}

# Exercise the actual backend's failure handling with a tiny inert runtime ZIP.
# Corrupt models must fail before any runtime executable can run or state is saved.
$runtimeFile = Join-Path $root 'runtime-fixture.zip'
$stream = [IO.File]::Open($runtimeFile, [IO.FileMode]::CreateNew)
$zip = New-Object IO.Compression.ZipArchive($stream, [IO.Compression.ZipArchiveMode]::Create, $true)
$runtimeBytes = $encoding.GetBytes('inert fixture; never execute')
$runtimePaths = @('runtime/python.exe', 'runtime/Scripts/conda-unpack-script.py', 'app/scripts/verify_install.py', 'DarkFusion.exe')
try {
    foreach ($path in $runtimePaths) {
        $entry = $zip.CreateEntry($path)
        $output = $entry.Open()
        try { $output.Write($runtimeBytes, 0, $runtimeBytes.Length) } finally { $output.Dispose() }
    }
}
finally { $zip.Dispose(); $stream.Dispose() }
$runtimeManifest = [ordered]@{
    schema_version = 1; product = 'DarkFusion'; version = '5.2.0'; source_commit = '0' * 40
    archive_size_bytes = ([IO.FileInfo]$runtimeFile).Length
    unpacked_size_bytes = $runtimeBytes.Length * $runtimePaths.Count
    sha256 = Get-TestHash ([IO.File]::ReadAllBytes($runtimeFile))
}
$runtimeManifestFile = Join-Path $root 'runtime-fixture.json'
[IO.File]::WriteAllText($runtimeManifestFile, ($runtimeManifest | ConvertTo-Json), $encoding)
foreach ($case in @('archive_hash', 'model_hash')) {
    $caseRoot = Join-Path $root $case
    $destination = Join-Path $caseRoot 'Actual backend App'
    $logFile = Join-Path $caseRoot 'backend.log'
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $backendOutput = & (Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe') `
            -NoProfile -ExecutionPolicy Bypass -File $scriptFile -PackagePath $runtimeFile `
            -ManifestPath $runtimeManifestFile -InstallDirectory $destination -LogPath $logFile `
            -ModelPackagePath (Join-Path $caseRoot 'models.zip') -ModelManifestPath (Join-Path $caseRoot 'models.json') 2>&1
        $backendExit = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $oldPreference }
    if ($backendExit -ne 1) { throw "The actual backend returned unexpected exit code $backendExit for $case." }
    $log = [IO.File]::ReadAllText($logFile)
    $expected = if ($case -eq 'archive_hash') { 'The model package checksum does not match' } else { 'The installed model checksum does not match' }
    if (-not $log.Contains($expected)) { throw "The actual backend did not reach the expected model failure: $log" }
    if ($log.Contains('SUCCESS:') -or $log.Contains('Created shortcut:') -or
        (Test-Path -LiteralPath (Join-Path $destination 'install-state.json'))) {
        throw 'The failed model installation was marked successful or created shortcuts.'
    }
    if ($case -eq 'archive_hash' -and (Test-Path -LiteralPath $destination)) { throw 'The corrupt model bundle modified the install destination.' }
    $results += [pscustomobject]@{ test = "backend_$case"; passed = $true; expected_failure = $expected }
    Write-Host "PASS: backend_$case leaves no completion state or shortcuts"
}
[IO.File]::WriteAllText((Join-Path $root 'results.json'), ($results | ConvertTo-Json -Depth 5), $encoding)
Write-Host "Model package fixtures retained at $root"
