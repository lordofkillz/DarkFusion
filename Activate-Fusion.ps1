# Dot-source this script to activate fusion in the current PowerShell terminal.
$fusionPrefix = Join-Path $env:USERPROFILE 'miniconda3\envs\fusion'
$fusionHook = Join-Path $env:USERPROFILE 'miniconda3\shell\condabin\conda-hook.ps1'
if (-not (Test-Path -LiteralPath (Join-Path $fusionPrefix 'python.exe'))) {
    throw "The DarkFusion fusion environment is missing: $fusionPrefix"
}
. $fusionHook
conda activate $fusionPrefix
if ($LASTEXITCODE -ne 0) { throw 'Could not activate the fusion environment.' }
$env:PYTHONNOUSERSITE = '1'
foreach ($fusionVariable in @('PYTHONHOME', 'PYTHONPATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_PLATFORM')) {
    [Environment]::SetEnvironmentVariable($fusionVariable, $null, 'Process')
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PATH = "$fusionPrefix;$fusionPrefix\Scripts;$fusionPrefix\Library\bin;$env:PATH"
$env:HF_HOME = Join-Path $PSScriptRoot 'UltraDarkFusion\hf_cache'
$env:TORCH_HOME = Join-Path $PSScriptRoot 'UltraDarkFusion\hf_cache\torch'
$env:ULTRALYTICS_CONFIG_DIR = Join-Path $PSScriptRoot 'UltraDarkFusion\.darkfusion\ultralytics'
Write-Host "DarkFusion environment: $fusionPrefix"
