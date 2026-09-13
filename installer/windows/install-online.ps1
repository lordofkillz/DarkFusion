[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InstallDirectory,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [switch]$DesktopShortcut,
    [switch]$StartMenuShortcut
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$exitCode = 1
$cacheLock = $null
$client = $null
$cache = $null

function Write-DownloadLog([string]$Message) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    [IO.File]::AppendAllText($LogPath, $line + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
}

function Assert-Unlinked([string]$Path) {
    $current = $Path
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Choose an ordinary local folder instead of a linked folder: $current"
            }
        }
        $current = [IO.Path]::GetDirectoryName($current)
    }
}

function Test-Download([string]$Path, [long]$Size, [string]$Hash) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    Assert-Unlinked $Path
    if ((Get-Item -LiteralPath $Path).Length -ne $Size) { return $false }
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        $actual = [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '')
        return $actual -ieq $Hash
    }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}

function Receive-Part($Part, [int]$Number, [int]$Total) {
    $target = Join-Path $cache ([string]$Part.name)
    if (Test-Download $target ([long]$Part.size_bytes) ([string]$Part.sha256)) {
        Write-DownloadLog "STAGE: Download $Number of $Total is ready"
        return $target
    }
    $partial = $target + '.partial'
    Assert-Unlinked $partial
    if (Test-Download $partial ([long]$Part.size_bytes) ([string]$Part.sha256)) {
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Force }
        Move-Item -LiteralPath $partial -Destination $target
        Write-DownloadLog "STAGE: Download $Number of $Total is ready"
        return $target
    }
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $response = $null
        $request = $null
        $input = $null
        $output = $null
        try {
            [long]$offset = 0
            if (Test-Path -LiteralPath $partial -PathType Leaf) {
                $offset = (Get-Item -LiteralPath $partial).Length
                if ($offset -ge [long]$Part.size_bytes) { $offset = 0 }
            }
            Write-DownloadLog "STAGE: Downloading application files ($Number of $Total)"
            $request = New-Object Net.Http.HttpRequestMessage([Net.Http.HttpMethod]::Get, [string]$Part.url)
            if ($offset -gt 0) { $request.Headers.Range = New-Object Net.Http.Headers.RangeHeaderValue($offset, $null) }
            $response = $client.SendAsync($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
            [void]$response.EnsureSuccessStatusCode()
            if ([int]$response.StatusCode -eq 206) {
                if ($null -eq $response.Content.Headers.ContentRange -or $response.Content.Headers.ContentRange.From -ne $offset) {
                    throw 'The download server returned an unexpected file range.'
                }
            }
            else { $offset = 0 }
            $mode = if ($offset -gt 0) { [IO.FileMode]::Append } else { [IO.FileMode]::Create }
            $output = [IO.File]::Open($partial, $mode, [IO.FileAccess]::Write, [IO.FileShare]::None)
            $input = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
            $buffer = New-Object byte[] (1024 * 1024)
            [long]$received = $offset
            $nextUpdate = [DateTime]::UtcNow
            while ($true) {
                $pendingRead = $input.ReadAsync($buffer, 0, $buffer.Length)
                if (-not $pendingRead.Wait(60000)) { throw 'The download connection timed out. Retry setup to resume.' }
                $count = $pendingRead.Result
                if ($count -eq 0) { break }
                if ($received + $count -gt [long]$Part.size_bytes) { throw 'The download exceeded its expected size.' }
                $output.Write($buffer, 0, $count)
                $received += $count
                if ([DateTime]::UtcNow -ge $nextUpdate) {
                    $percent = [int][Math]::Floor(100.0 * $received / [long]$Part.size_bytes)
                    Write-DownloadLog "STAGE: Downloading application files ($Number of $Total, $percent%)"
                    $nextUpdate = [DateTime]::UtcNow.AddSeconds(2)
                }
            }
            $output.Dispose(); $output = $null
            $input.Dispose(); $input = $null
            if (-not (Test-Download $partial ([long]$Part.size_bytes) ([string]$Part.sha256))) {
                # This is this installer's partial download, never a user document.
                Remove-Item -LiteralPath $partial -Force
                throw 'The downloaded file was incomplete or damaged. Retrying the download.'
            }
            if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Force }
            Move-Item -LiteralPath $partial -Destination $target
            return $target
        }
        catch {
            Write-DownloadLog "Download attempt $attempt failed: $($_.Exception.Message)"
            if ($attempt -eq 3) { throw 'The download could not finish. Check your internet connection and run setup again; completed downloads will be reused.' }
        }
        finally {
            if ($output) { $output.Dispose() }
            if ($input) { $input.Dispose() }
            if ($response) { $response.Dispose() }
            if ($request) { $request.Dispose() }
        }
    }
}

try {
    if ($InstallDirectory -notmatch '^[A-Za-z]:[\\/]' -or $LogPath -notmatch '^[A-Za-z]:[\\/]') {
        throw 'Choose a full folder path on a local drive.'
    }
    $InstallDirectory = [IO.Path]::GetFullPath($InstallDirectory).TrimEnd('\')
    Assert-Unlinked $InstallDirectory
    if (Test-Path -LiteralPath $InstallDirectory) {
        if (-not (Test-Path -LiteralPath $InstallDirectory -PathType Container) -or
            @(Get-ChildItem -LiteralPath $InstallDirectory -Force | Select-Object -First 1).Count -gt 0) {
            throw 'Choose a new or empty installation folder. Existing files will be preserved.'
        }
    }
    if ($InstallDirectory.TrimEnd('\') -eq [IO.Path]::GetPathRoot($InstallDirectory).TrimEnd('\')) {
        throw 'Choose a folder inside the drive, not the drive itself.'
    }
    foreach ($protected in @($env:SystemRoot, $env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:ProgramData)) {
        if ($protected -and ($InstallDirectory -ieq $protected -or $InstallDirectory.StartsWith($protected.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase))) {
            throw 'Choose a writable personal folder outside Windows and Program Files.'
        }
    }
    Assert-Unlinked $LogPath
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($LogPath))
    Write-DownloadLog 'STAGE: Preparing your installation'
    $manifest = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'download.json')) | ConvertFrom-Json
    if ($manifest.schema_version -ne 1 -or $manifest.product -cne 'DarkFusion' -or
        $manifest.payload.sha256 -notmatch '^[a-fA-F0-9]{64}$' -or @($manifest.parts).Count -lt 1) {
        throw 'The installer contains an invalid download manifest. Download the installer again.'
    }
    [long]$partTotal = 0
    $names = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($part in $manifest.parts) {
        $uri = [Uri]([string]$part.url)
        if ($part.name -notmatch '^DarkFusion-runtime\.[0-9]{3}\.bin$' -or
            -not $names.Add([string]$part.name) -or $part.sha256 -notmatch '^[a-fA-F0-9]{64}$' -or
            [long]$part.size_bytes -le 0 -or [long]$part.size_bytes -ge 2GB -or
            $uri.Scheme -cne 'https' -or $uri.Host -cne 'github.com' -or
            -not $uri.AbsolutePath.StartsWith('/lordofkillz/DarkFusion/releases/download/', [StringComparison]::Ordinal)) {
            throw 'The installer contains invalid download information. Download the installer again.'
        }
        $partTotal += [long]$part.size_bytes
    }
    if ($partTotal -ne [long]$manifest.payload.archive_size_bytes) { throw 'The download package size is invalid.' }
    $parent = [IO.Path]::GetDirectoryName($InstallDirectory)
    $cacheBase = Join-Path $parent '.DarkFusion-Setup-Cache'
    $cache = Join-Path $cacheBase ([string]$manifest.payload.sha256).ToLowerInvariant()
    Assert-Unlinked $cache
    [void][IO.Directory]::CreateDirectory($cache)
    [IO.File]::SetAttributes($cacheBase, ([IO.File]::GetAttributes($cacheBase) -bor [IO.FileAttributes]::Hidden))
    $cacheLock = [IO.File]::Open((Join-Path $cache 'download.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $drive = New-Object IO.DriveInfo([IO.Path]::GetPathRoot($InstallDirectory))
    if ($drive.DriveType -eq [IO.DriveType]::Network) { throw 'Choose a local drive.' }
    $needed = [decimal]$manifest.payload.unpacked_size_bytes + 2 * [decimal]$partTotal + 1GB
    $cachedBytes = (Get-ChildItem -LiteralPath $cache -File | Measure-Object Length -Sum).Sum
    if ($cachedBytes) { $needed -= [Math]::Min([decimal]$cachedBytes, 2 * [decimal]$partTotal) }
    if ([decimal]$drive.AvailableFreeSpace -lt $needed) { throw 'Allow at least 25 GB free on the selected drive while setup downloads and installs DarkFusion.' }
    $payload = Join-Path $cache 'payload.zip'
    if (-not (Test-Download $payload ([long]$manifest.payload.archive_size_bytes) ([string]$manifest.payload.sha256))) {
        Add-Type -AssemblyName System.Net.Http
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $client = New-Object Net.Http.HttpClient
        $client.Timeout = [TimeSpan]::FromMinutes(5)
        $client.DefaultRequestHeaders.UserAgent.ParseAdd('DarkFusionSetup/5.2')
        $downloaded = @()
        $index = 0
        foreach ($part in $manifest.parts) {
            $index++
            $downloaded += Receive-Part $part $index @($manifest.parts).Count
        }
        Write-DownloadLog 'STAGE: Preparing downloaded application files'
        $combined = $payload + '.partial'
        Assert-Unlinked $combined
        $output = [IO.File]::Open($combined, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try {
            foreach ($partFile in $downloaded) {
                $input = [IO.File]::OpenRead($partFile)
                try { $input.CopyTo($output) } finally { $input.Dispose() }
            }
        }
        finally { $output.Dispose() }
        if (-not (Test-Download $combined ([long]$manifest.payload.archive_size_bytes) ([string]$manifest.payload.sha256))) {
            throw 'The complete download could not be verified. Run setup again.'
        }
        if (Test-Path -LiteralPath $payload) { Remove-Item -LiteralPath $payload -Force }
        Move-Item -LiteralPath $combined -Destination $payload
    }
    # Free the separate chunks after verifying the assembled archive.
    foreach ($part in $manifest.parts) {
        $path = Join-Path $cache ([string]$part.name)
        if (Test-Path -LiteralPath $path -PathType Leaf) { Assert-Unlinked $path; Remove-Item -LiteralPath $path -Force }
    }
    $payloadManifest = Join-Path $cache 'payload.json'
    [IO.File]::WriteAllText($payloadManifest, ($manifest.payload | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding($false)))
    $installParameters = @{PackagePath=$payload; ManifestPath=$payloadManifest; InstallDirectory=$InstallDirectory; LogPath=$LogPath}
    if ($DesktopShortcut) { $installParameters.DesktopShortcut = $true }
    if ($StartMenuShortcut) { $installParameters.StartMenuShortcut = $true }
    & (Join-Path $PSScriptRoot 'install-standalone.ps1') @installParameters
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        Remove-Item -LiteralPath $payload -Force
        Remove-Item -LiteralPath $payloadManifest -Force
    }
}
catch {
    try { Write-DownloadLog "ERROR: $($_.Exception.Message)" } catch { [Console]::Error.WriteLine('Setup could not start. Choose a writable local folder.') }
}
finally {
    if ($client) { $client.Dispose() }
    if ($cacheLock) { $cacheLock.Dispose() }
}
exit $exitCode
