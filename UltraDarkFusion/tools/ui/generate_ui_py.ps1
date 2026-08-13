# Regenerate the Python UI module after editing UltraDarkFusion_v5.2.ui.
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
python -m PyQt5.uic.pyuic `
    (Join-Path $projectRoot "UltraDarkFusion_v5.2.ui") `
    -o (Join-Path $projectRoot "ui_ultradarkfusion_v5_2.py")
