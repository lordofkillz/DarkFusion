"""Run the Ultralytics CLI with the active DarkFusion Python environment."""

from __future__ import annotations

import sys


def main() -> None:
    from ultralytics.cfg import entrypoint

    sys.argv = ["yolo", *sys.argv[1:]]
    entrypoint()


if __name__ == "__main__":
    main()
