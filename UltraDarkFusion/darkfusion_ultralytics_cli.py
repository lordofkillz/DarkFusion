"""Run the Ultralytics CLI with the active DarkFusion Python environment."""

from __future__ import annotations

import sys


def prepare_validation_args(args):
    """Expand Entire Dataset metrics in the worker, without rewriting a dataset.

    Ultralytics accepts train/val/test, not all. Build one deduplicated validation
    list under the run output so mAP is computed on the union, not an average of
    separate split scores. Keep image discovery off the Qt UI thread.
    """
    import os
    from pathlib import Path
    import tempfile

    import yaml

    args = list(args)
    values = dict(str(arg).split("=", 1) for arg in args if "=" in str(arg))
    if "val" not in args and values.get("mode") != "val":
        return args
    split = values.get("split", "val")
    if split not in {"train", "val", "test", "all"}:
        raise ValueError(f"Invalid metrics split: {split}")
    if split != "all":
        return args
    if "classify" in args or values.get("task") == "classify":
        raise ValueError("Entire Dataset metrics requires an annotation dataset YAML. Select Validation or Test for classification.")
    data_path = Path(values.get("data", "")).resolve(strict=True)
    with data_path.open("r", encoding="utf-8-sig") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_path}")
    root = Path(payload.get("path") or data_path.parent)
    if not root.is_absolute():
        root = data_path.parent / root
    root = root.resolve()
    suffixes = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}
    images = {}

    def resolve(value, list_dir=None):
        path = Path(str(value).strip().strip('"').strip("'"))
        if path.is_absolute():
            return path.resolve()
        for base in (list_dir, root, data_path.parent):
            if base is not None and (base / path).exists():
                return (base / path).resolve()
        return (root / path).resolve()

    def add_image(path):
        if path.is_file() and path.suffix.lower() in suffixes:
            images.setdefault(os.path.normcase(str(path.resolve())), path.resolve().as_posix())
        else:
            raise ValueError(f"Dataset image is missing or unsupported: {path}")

    for key in ("train", "val", "validation", "test"):
        entries = payload.get(key) or []
        for value in entries if isinstance(entries, list) else [entries]:
            path = resolve(value)
            if path.is_dir():
                for image in sorted(path.rglob("*")):
                    if image.is_file() and image.suffix.lower() in suffixes:
                        add_image(image)
            elif path.is_file() and path.suffix.lower() == ".txt":
                with path.open("r", encoding="utf-8-sig") as handle:
                    for line in handle:
                        if line.strip():
                            add_image(resolve(line.strip(), path.parent))
            else:
                add_image(path)
    if not images:
        raise ValueError("Dataset YAML has no train, validation, or test images.")
    if not values.get("project") or not values.get("name"):
        raise ValueError("Entire Dataset metrics requires an explicit run output folder.")
    run_dir = Path(values["project"]) / values["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="metrics_inputs_", dir=run_dir))
    image_list = output / "all_images.txt"
    image_list.write_text("".join(path + "\n" for path in sorted(images.values())), encoding="utf-8")
    payload["path"] = root.as_posix()
    # Only val is read during this score-only pass; avoid requiring a train
    # split or permitting an automatic dataset download from a copied YAML.
    payload["train"] = image_list.as_posix()
    payload["val"] = image_list.as_posix()
    payload.pop("download", None)
    generated_yaml = output / "all_metrics.yaml"
    generated_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"DarkFusion Metrics Only: Entire Dataset ({len(images)} unique images); source YAML unchanged.", flush=True)
    args = [arg for arg in args if not str(arg).startswith(("data=", "split="))]
    return [*args, f"data={generated_yaml.as_posix()}", "split=val"]


def run_ultralytics_entrypoint(args):
    """Run the official CLI without its optional VS Code promotion.

    Its extension-path lookup can fail after successful work with a shallow or
    relative YOLO_CONFIG_DIR. Suppress only that message in this worker process;
    never change persisted Ultralytics settings or swallow real command errors.
    """
    from ultralytics import cfg

    was_vscode = cfg.IS_VSCODE
    cfg.IS_VSCODE = False
    try:
        sys.argv = ["yolo", *args]
        return cfg.entrypoint()
    finally:
        cfg.IS_VSCODE = was_vscode


def main() -> None:
    run_ultralytics_entrypoint(prepare_validation_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
