# Run with Windows PowerShell 5.1. No third-party packages are required.
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$tree = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'install-standalone.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'The installation backend does not parse.' }
foreach ($function in $tree.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst]}, $false)) {
    . ([ScriptBlock]::Create($function.Extent.Text))
}
$script:InstallLog = $null
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('DarkFusion-shortcuts-' + [Guid]::NewGuid().ToString('N'))
$destination = Join-Path $fixture ('App ' + [char]0x00e9 + [char]0x03a9 + ' & spaces')
$folder = Join-Path $fixture ('Shortcuts ' + [char]0x4e2d)
[void][IO.Directory]::CreateDirectory($destination)
$executable = Join-Path $destination 'DarkFusion.exe'
[IO.File]::WriteAllText($executable, 'A disposable shortcut target; never executed.')
New-LauncherShortcut $folder $destination
$first = Join-Path $folder 'DarkFusion.lnk'
$original = [Convert]::ToBase64String([IO.File]::ReadAllBytes($first))
New-LauncherShortcut $folder $destination
$links = @(Get-ChildItem -LiteralPath $folder -Filter '*.lnk')
if ($links.Count -ne 2 -or $original -cne [Convert]::ToBase64String([IO.File]::ReadAllBytes($first))) {
    throw 'Creating another shortcut did not preserve the existing shortcut.'
}
$shell = New-Object -ComObject Shell.Application
try {
    $namespace = $shell.Namespace($folder)
    foreach ($link in $links) {
        $item = $namespace.ParseName($link.Name)
        $shortcut = $item.GetLink
        if ($shortcut.Path -cne $executable -or $shortcut.WorkingDirectory -cne $destination) {
            throw 'The Windows Shell did not read back the exact Unicode shortcut paths.'
        }
    }
}
finally { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($shell) }
Write-Host 'PASS: Unicode target, working directory and shortcut folder; existing shortcuts preserved.'
Write-Host "Disposable fixture retained at $fixture"
