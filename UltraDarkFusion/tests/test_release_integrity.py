"""Keep direct application modules in source and installer verification lists."""

import ast
import importlib.util
from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
MAIN_SOURCE = APP_ROOT / "UltraDarkFusion_v5.2.py"


def direct_application_modules():
    tree = ast.parse(MAIN_SOURCE.read_text(encoding="utf-8"))
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    named_modules = {
        "dinov5_2",
        "prediction_size_filter",
        "sahi_predict_wrapperv5",
        "splash_utils",
        "training_eta",
        "ui_ultradarkfusion_v5_2",
    }
    return sorted(
        module
        for module in imported
        if module.startswith("darkfusion_") or module in named_modules
    )


class ReleaseIntegrityTests(unittest.TestCase):
    def test_direct_application_modules_exist_and_are_verified(self):
        verify_path = REPO_ROOT / "scripts" / "verify_install.py"
        spec = importlib.util.spec_from_file_location("darkfusion_verify_install_test", verify_path)
        verify_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verify_module)
        required = set(verify_module.REQUIRED_FILES)

        for module in direct_application_modules():
            filename = f"{module}.py"
            self.assertTrue((APP_ROOT / filename).is_file(), f"Missing local import: {filename}")
            self.assertIn(filename, required, f"Installer verification omits {filename}")

    def test_direct_application_modules_are_in_upload_manifest(self):
        manifest = (REPO_ROOT / "UPLOAD_MANIFEST.md").read_text(encoding="utf-8")
        for module in direct_application_modules():
            path = f"UltraDarkFusion/{module}.py"
            self.assertIn(f"`{path}`", manifest, f"Upload manifest omits {path}")


if __name__ == "__main__":
    unittest.main()
