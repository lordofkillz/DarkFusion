[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PackagePath,
    [Parameter(Mandatory = $true)][string]$ManifestPath,
    [Parameter(Mandatory = $true)][string]$InstallDirectory,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [switch]$DesktopShortcut,
    [switch]$StartMenuShortcut
)

# This backend is invoked by DarkFusionSetup.exe. It never installs into, or
# activates, an existing Python/Conda environment and never deletes a target.
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$script:InstallLog = $null
$script:ExtractionStarted = $false
$script:Destination = $null
$script:SavedEnvironment = @{}
$script:EnvironmentChanged = $false
$script:ExitCode = 1
$packageStream = $null
$archive = $null

function Write-InstallLog([string]$Message) {
    $line = ('[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message)
    [Console]::WriteLine($line)
    if ($script:InstallLog) {
        [IO.File]::AppendAllText($script:InstallLog, $line + [Environment]::NewLine,
            (New-Object Text.UTF8Encoding($false)))
    }
}

function Get-LocalFullPath([string]$Path, [string]$Label) {
    if ([string]::IsNullOrWhiteSpace($Path) -or $Path -notmatch '^[A-Za-z]:[\\/]') {
        throw "$Label must be an absolute path on a local drive."
    }
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    if ($full.Length -eq 2) { $full += '\' }
    if ($full.Substring(2).Contains(':')) { throw "$Label cannot contain alternate data streams." }
    return $full
}

function Test-PathWithin([string]$Path, [string]$Directory) {
    return $Path.Equals($Directory, [StringComparison]::OrdinalIgnoreCase) -or
        $Path.StartsWith($Directory.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)
}

function Assert-NoReparseAncestors([string]$Path) {
    $current = $Path
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Linked folders or files are not supported: $current"
            }
        }
        $parent = [IO.Path]::GetDirectoryName($current)
        if ($parent -eq $current) { break }
        $current = $parent
    }
}

function Assert-EmptyDestination([string]$Path) {
    Assert-NoReparseAncestors $Path
    if (Test-Path -LiteralPath $Path) {
        if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
            throw 'The installation destination is a file. Choose a new or empty folder.'
        }
        if (@(Get-ChildItem -LiteralPath $Path -Force | Select-Object -First 1).Count -gt 0) {
            throw 'The installation folder is not empty. Choose a new or empty folder; existing installations and personal files are never overwritten.'
        }
    }
}

function Ensure-ExtractionDirectory([string]$Path) {
    # The destination started empty. Validate each directory when first created
    # instead of repeatedly walking all ancestors for ~100,000 runtime files.
    # FileMode.CreateNew separately prevents replacement of any existing file.
    if ($script:ValidatedDirectories.Contains($Path)) { return }
    $parent = [IO.Path]::GetDirectoryName($Path)
    if (-not $parent -or -not (Test-PathWithin $Path $script:Destination)) {
        throw 'An extraction directory escaped the installation folder.'
    }
    Ensure-ExtractionDirectory $parent
    [void][IO.Directory]::CreateDirectory($Path)
    $attributes = [IO.File]::GetAttributes($Path)
    if (($attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        ($attributes -band [IO.FileAttributes]::Directory) -eq 0) {
        throw "The extraction folder is linked or is not a directory: $Path"
    }
    [void]$script:ValidatedDirectories.Add($Path)
}

function Get-ManifestInteger($Object, [string]$Name) {
    $value = $Object.PSObject.Properties[$Name]
    if ($null -eq $value -or [string]$value.Value -notmatch '^[0-9]+$') {
        throw "Package manifest has an invalid $Name value."
    }
    try { $number = [long]::Parse([string]$value.Value, [Globalization.CultureInfo]::InvariantCulture) }
    catch { throw "Package manifest has an invalid $Name value." }
    if ($number -le 0) { throw "Package manifest has an invalid $Name value." }
    return $number
}

function Invoke-PrivatePython([string]$Python, [string[]]$Arguments) {
    # Native stderr contains dependency warnings as well as failures. Only the
    # process exit code decides success; both output streams remain in the log.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments 2>&1 | ForEach-Object { Write-InstallLog ([string]$_) }
        $nativeExit = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previousPreference }
    if ($nativeExit -ne 0) { throw "Application verification or runtime setup failed (exit code $nativeExit). See the installation log for details." }
}

function New-LauncherShortcut([string]$Folder, [string]$Destination) {
    Assert-NoReparseAncestors $Folder
    if (-not (Test-Path -LiteralPath $Folder -PathType Container)) {
        [void][IO.Directory]::CreateDirectory($Folder)
    }
    $shortcutPath = Join-Path $Folder 'DarkFusion.lnk'
    # Preserve shortcuts belonging to another installation.
    if (Test-Path -LiteralPath $shortcutPath) {
        $shortcutPath = Join-Path $Folder ('DarkFusion ({0}).lnk' -f (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    }
    Assert-NoReparseAncestors $shortcutPath
    # WScript.Shell converts some Unicode target paths through the system ANSI
    # code page. Use the Unicode shell interface for every path instead.
    if (-not ('DarkFusion.Installer.ShortcutWriter' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;
namespace DarkFusion.Installer {
    [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
    internal class ShellLink { }
    [ComImport, Guid("000214F9-0000-0000-C000-000000000046"),
        InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IShellLinkW {
        void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder path, int size, IntPtr data, uint flags);
        void GetIDList(out IntPtr list);
        void SetIDList(IntPtr list);
        void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder text, int size);
        void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string text);
        void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder path, int size);
        void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string path);
        void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder text, int size);
        void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string text);
        void GetHotkey(out short key);
        void SetHotkey(short key);
        void GetShowCmd(out int command);
        void SetShowCmd(int command);
        void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder path, int size, out int index);
        void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string path, int index);
        void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string path, uint reserved);
        void Resolve(IntPtr window, uint flags);
        void SetPath([MarshalAs(UnmanagedType.LPWStr)] string path);
    }
    public static class ShortcutWriter {
        public static void Save(string shortcut, string target, string directory) {
            var link = (IShellLinkW)new ShellLink();
            try {
                link.SetPath(target);
                link.SetWorkingDirectory(directory);
                link.SetIconLocation(target, 0);
                link.SetDescription("DarkFusion");
                ((IPersistFile)link).Save(shortcut, true);
            }
            finally { Marshal.FinalReleaseComObject(link); }
        }
    }
}
'@
    }
    [DarkFusion.Installer.ShortcutWriter]::Save($shortcutPath, (Join-Path $Destination 'DarkFusion.exe'), $Destination)
    Write-InstallLog "Created shortcut: $shortcutPath"
}

try {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT -or
        -not [Environment]::Is64BitOperatingSystem -or -not [Environment]::Is64BitProcess) {
        throw 'DarkFusion requires 64-bit Windows and the 64-bit installer.'
    }
    $windowsVersion = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
    if ([int]$windowsVersion.CurrentBuildNumber -lt 18362) {
        throw 'DarkFusion requires Windows 10 version 1903 or later, or Windows 11.'
    }

    $script:Destination = Get-LocalFullPath $InstallDirectory 'Installation folder'
    $packageFile = Get-LocalFullPath $PackagePath 'Package path'
    $manifestFile = Get-LocalFullPath $ManifestPath 'Manifest path'
    $logFile = Get-LocalFullPath $LogPath 'Log path'
    if (Test-PathWithin $logFile $script:Destination) {
        throw 'Choose an installation folder separate from the installation log.'
    }
    Assert-NoReparseAncestors $logFile
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($logFile))
    $script:InstallLog = $logFile
    Write-InstallLog 'STAGE: Checking installation package'
    $driveRoot = [IO.Path]::GetPathRoot($script:Destination)
    if ($script:Destination.TrimEnd('\') -eq $driveRoot.TrimEnd('\')) {
        throw 'Choose a folder inside the drive, not the drive root.'
    }
    foreach ($protectedFolder in @($env:SystemRoot, $env:ProgramFiles,
            ${env:ProgramFiles(x86)}, $env:ProgramData)) {
        if ($protectedFolder -and (Test-PathWithin $script:Destination ([IO.Path]::GetFullPath($protectedFolder)))) {
            throw 'Choose a personal folder outside Windows, Program Files, and ProgramData.'
        }
    }
    if ($env:USERPROFILE -and $script:Destination.Equals($env:USERPROFILE, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Choose a folder inside your profile, not the profile root.'
    }
    foreach ($inputFile in @($packageFile, $manifestFile, $PSCommandPath, $logFile)) {
        if (Test-PathWithin $inputFile $script:Destination) {
            throw 'Choose an installation folder separate from the setup package and installation log.'
        }
    }
    # When executed from the source tree, prevent an accidental nested install.
    $sourceCandidate = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
    if ((Test-Path -LiteralPath (Join-Path $sourceCandidate 'UltraDarkFusion') -PathType Container) -and
        (Test-PathWithin $script:Destination $sourceCandidate)) {
        throw 'Choose an installation folder outside the DarkFusion source checkout.'
    }
    Assert-EmptyDestination $script:Destination
    foreach ($inputFile in @($packageFile, $manifestFile)) {
        Assert-NoReparseAncestors $inputFile
        if (-not (Test-Path -LiteralPath $inputFile -PathType Leaf)) { throw "Setup file is missing: $inputFile" }
    }
    $manifest = [IO.File]::ReadAllText($manifestFile) | ConvertFrom-Json
    if ($null -eq $manifest -or $manifest.schema_version -ne 1 -or
        $manifest.product -cne 'DarkFusion' -or
        [string]$manifest.version -notmatch '^[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:[-+][A-Za-z0-9.-]+)?$' -or
        [string]$manifest.source_commit -notmatch '^[0-9a-fA-F]{40}$' -or
        [string]$manifest.sha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw 'Package manifest is invalid or uses an unsupported format.'
    }
    $archiveSize = Get-ManifestInteger $manifest 'archive_size_bytes'
    $unpackedSize = Get-ManifestInteger $manifest 'unpacked_size_bytes'
    # Hold the same read-only file handle for hashing, validation and extraction.
    $packageStream = [IO.File]::Open($packageFile, [IO.FileMode]::Open,
        [IO.FileAccess]::Read, [IO.FileShare]::Read)
    if ($packageStream.Length -ne $archiveSize) { throw 'The installation package size does not match its manifest. Download the complete package again.' }
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { $actualHash = [BitConverter]::ToString($hasher.ComputeHash($packageStream)).Replace('-', '').ToLowerInvariant() }
    finally { $hasher.Dispose() }
    if ($actualHash -cne ([string]$manifest.sha256).ToLowerInvariant()) {
        throw 'The installation package checksum does not match. Download the package again.'
    }
    $packageStream.Position = 0
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = New-Object IO.Compression.ZipArchive($packageStream, [IO.Compression.ZipArchiveMode]::Read, $true)
    $entries = New-Object 'System.Collections.Generic.List[object]'
    $names = New-Object 'System.Collections.Generic.Dictionary[string,bool]' ([StringComparer]::OrdinalIgnoreCase)
    [long]$actualUnpackedSize = 0
    foreach ($entry in $archive.Entries) {
        $name = $entry.FullName.Replace('\', '/')
        $isDirectory = $name.EndsWith('/')
        $relative = $name.TrimEnd('/')
        if (-not $relative -or $relative.StartsWith('/') -or $relative -match '[<>:"|?*\x00-\x1f]') {
            throw "The package contains an unsafe entry: $name"
        }
        foreach ($component in $relative.Split('/')) {
            if (-not $component -or $component -in @('.', '..') -or
                $component.EndsWith('.') -or $component.EndsWith(' ') -or
                $component -match '^(?i:CON|PRN|AUX|NUL|CLOCK\$|COM[1-9\u00b9\u00b2\u00b3]|LPT[1-9\u00b9\u00b2\u00b3])(?:\.|$)') {
                throw "The package contains an unsafe entry: $name"
            }
        }
        $unixKind = ($entry.ExternalAttributes -shr 16) -band 0xf000
        if ($unixKind -notin @(0, 0x4000, 0x8000) -or ($entry.ExternalAttributes -band 0x400) -ne 0) {
            throw "Linked or special archive entries are not supported: $name"
        }
        if ($names.ContainsKey($relative)) { throw "The package contains conflicting paths: $name" }
        $names.Add($relative, $isDirectory)
        $target = [IO.Path]::GetFullPath((Join-Path $script:Destination $relative.Replace('/', '\')))
        if (-not (Test-PathWithin $target $script:Destination) -or $target -eq $script:Destination) {
            throw "The package contains an unsafe entry: $name"
        }
        if ($isDirectory -and $entry.Length -ne 0) { throw "The package has an invalid directory entry: $name" }
        if ($entry.Length -gt ($unpackedSize - $actualUnpackedSize)) { throw 'Unpacked package size exceeds the manifest.' }
        $actualUnpackedSize += $entry.Length
        $entries.Add([pscustomobject]@{ Entry = $entry; Relative = $relative; Target = $target; Directory = $isDirectory })
    }
    if ($actualUnpackedSize -ne $unpackedSize) { throw 'Unpacked package size does not match the manifest.' }
    foreach ($item in $entries) {
        $parts = $item.Relative.Split('/')
        for ($index = 1; $index -lt $parts.Length; $index++) {
            $parentName = [string]::Join('/', $parts[0..($index - 1)])
            if ($names.ContainsKey($parentName) -and -not $names[$parentName]) {
                throw "The package contains conflicting file and folder paths: $parentName"
            }
        }
    }
    foreach ($requiredFile in @('runtime/python.exe', 'app/scripts/verify_install.py', 'DarkFusion.exe')) {
        if (-not $names.ContainsKey($requiredFile) -or $names[$requiredFile]) { throw "The package is incomplete: $requiredFile is missing." }
    }
    $unpackRelative = @('runtime/Scripts/conda-unpack-script.py', 'runtime/Scripts/conda-unpack') |
        Where-Object { $names.ContainsKey($_) -and -not $names[$_] } | Select-Object -First 1
    if (-not $unpackRelative) { throw 'The private runtime relocation script is missing from the package.' }
    $drive = New-Object IO.DriveInfo($driveRoot)
    if ($drive.DriveType -eq [IO.DriveType]::Network) { throw 'Choose a local drive for the application runtime.' }
    $spaceRequired = [decimal]$unpackedSize + [Math]::Max([decimal]536870912, [decimal]$unpackedSize * [decimal]0.05)
    if ([decimal]$drive.AvailableFreeSpace -lt $spaceRequired) {
        throw ('Not enough free space. Allow at least {0:N1} GB for this installation.' -f ($spaceRequired / 1GB))
    }

    Write-InstallLog 'STAGE: Installing DarkFusion files'
    Assert-EmptyDestination $script:Destination
    [void][IO.Directory]::CreateDirectory($script:Destination)
    Assert-NoReparseAncestors $script:Destination
    $script:ValidatedDirectories = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    [void]$script:ValidatedDirectories.Add($script:Destination)
    $script:ExtractionStarted = $true
    [long]$extractedBytes = 0
    $lastProgress = -1
    foreach ($item in $entries) {
        if ($item.Directory) {
            Ensure-ExtractionDirectory $item.Target
        }
        else {
            $parentDirectory = [IO.Path]::GetDirectoryName($item.Target)
            Ensure-ExtractionDirectory $parentDirectory
            $entryInput = $item.Entry.Open()
            $output = $null
            try {
                $output = [IO.File]::Open($item.Target, [IO.FileMode]::CreateNew,
                    [IO.FileAccess]::Write, [IO.FileShare]::None)
                $entryInput.CopyTo($output)
                if ($output.Length -ne $item.Entry.Length) { throw "A package file could not be extracted completely: $($item.Relative)" }
            }
            finally {
                if ($output) { $output.Dispose() }
                $entryInput.Dispose()
            }
        }
        $extractedBytes += $item.Entry.Length
        $progress = [int][Math]::Floor(100.0 * $extractedBytes / $unpackedSize / 5) * 5
        if ($progress -gt $lastProgress) {
            Write-InstallLog "STAGE: Installing DarkFusion files ($progress%)"
            $lastProgress = $progress
        }
    }
    $archive.Dispose()
    $archive = $null
    $packageStream.Dispose()
    $packageStream = $null

    foreach ($variable in @('PATH', 'PYTHONNOUSERSITE', 'PYTHONHOME', 'PYTHONPATH',
            'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'CONDA_SHLVL', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_PLATFORM')) {
        $script:SavedEnvironment[$variable] = [Environment]::GetEnvironmentVariable($variable, 'Process')
    }
    $script:EnvironmentChanged = $true
    $runtime = Join-Path $script:Destination 'runtime'
    $env:PATH = (@($runtime, (Join-Path $runtime 'Library\mingw-w64\bin'),
        (Join-Path $runtime 'Library\usr\bin'), (Join-Path $runtime 'Library\bin'),
        (Join-Path $runtime 'Scripts'), (Join-Path $runtime 'bin'),
        (Join-Path $env:SystemRoot 'System32'), $env:SystemRoot,
        (Join-Path $env:SystemRoot 'System32\Wbem')) -join ';')
    $env:PYTHONNOUSERSITE = '1'
    foreach ($variable in @('PYTHONHOME', 'PYTHONPATH', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV',
            'CONDA_SHLVL', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH')) {
        [Environment]::SetEnvironmentVariable($variable, $null, 'Process')
    }
    $env:CONDA_PREFIX = $runtime
    $env:QT_PLUGIN_PATH = Join-Path $runtime 'Lib\site-packages\PyQt5\Qt5\plugins'
    $env:QT_QPA_PLATFORM_PLUGIN_PATH = Join-Path $env:QT_PLUGIN_PATH 'platforms'
    $python = Join-Path $runtime 'python.exe'
    Write-InstallLog 'STAGE: Preparing the private application runtime'
    Invoke-PrivatePython $python @((Join-Path $script:Destination $unpackRelative.Replace('/', '\')))
    Write-InstallLog 'STAGE: Verifying DarkFusion'
    Invoke-PrivatePython $python @((Join-Path $script:Destination 'app\scripts\verify_install.py'))
    Write-InstallLog 'STAGE: Checking application startup'
    $env:QT_QPA_PLATFORM = 'offscreen'
    Invoke-PrivatePython $python @('-c', "from PyQt5.QtWidgets import QApplication; app = QApplication([]); import mediapipe as mp; segment = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1); segment.close(); print('Qt and image-processing startup verified.')")

    $state = [ordered]@{
        schema_version = 1
        product = 'DarkFusion'
        version = [string]$manifest.version
        source_commit = [string]$manifest.source_commit
        package_sha256 = $actualHash
        installed_utc = [DateTime]::UtcNow.ToString('o')
        install_directory = $script:Destination
        status = 'complete'
    }
    $statePath = Join-Path $script:Destination 'install-state.json'
    Assert-NoReparseAncestors $statePath
    if (Test-Path -LiteralPath $statePath) { throw 'The package unexpectedly supplied installation state.' }
    [IO.File]::WriteAllText($statePath, ($state | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
    Write-InstallLog 'STAGE: Finishing installation'
    if ($DesktopShortcut) {
        try { New-LauncherShortcut ([Environment]::GetFolderPath('DesktopDirectory')) $script:Destination }
        catch { Write-InstallLog "WARNING: Could not create the Desktop shortcut: $($_.Exception.Message). Launch DarkFusion.exe from the installation folder." }
    }
    if ($StartMenuShortcut) {
        try { New-LauncherShortcut (Join-Path ([Environment]::GetFolderPath('Programs')) 'DarkFusion') $script:Destination }
        catch { Write-InstallLog "WARNING: Could not create the Start menu shortcut: $($_.Exception.Message). Launch DarkFusion.exe from the installation folder." }
    }
    Write-InstallLog 'SUCCESS: DarkFusion installation completed.'
    $script:ExitCode = 0
}
catch {
    Write-InstallLog "ERROR: $($_.Exception.Message)"
    if ($script:ExtractionStarted) {
        Write-InstallLog "The incomplete installation was preserved at: $script:Destination"
        Write-InstallLog 'Choose a different empty folder when retrying. After reviewing its contents, you may remove the incomplete folder yourself. No shortcuts were created before verification.'
    }
}
finally {
    if ($archive) { $archive.Dispose() }
    if ($packageStream) { $packageStream.Dispose() }
    if ($script:EnvironmentChanged) {
        foreach ($variable in $script:SavedEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable($variable, $script:SavedEnvironment[$variable], 'Process')
        }
    }
}
exit $script:ExitCode
