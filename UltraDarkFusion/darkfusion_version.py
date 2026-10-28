"""Read DarkFusion's product version from the repository/install VERSION file."""
from pathlib import Path
import re

VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
APP_VERSION = VERSION_FILE.read_text(encoding="utf-8-sig").strip()
if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", APP_VERSION):
    raise RuntimeError(f"Invalid DarkFusion product version in {VERSION_FILE}")

__version__ = APP_VERSION
