"""Persist dataset tuning choices without changing dataset paths or class metadata."""

from collections.abc import Mapping
import math
import os
from pathlib import Path
import tempfile

import yaml


# These are training settings without duplicate controls in Train Setup.
# Filter against the installed Ultralytics schema when opening the editor.
TRAINING_HYPERPARAMETER_GROUPS = {
    "Learning": ("optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
                 "warmup_momentum", "warmup_bias_lr", "cos_lr", "nbs"),
    "Augmentations": ("multi_scale", "mosaic", "close_mosaic", "mixup", "cutmix", "hsv_h",
                      "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
                      "flipud", "fliplr", "bgr", "copy_paste", "copy_paste_mode"),
    "Precision and performance": ("amp", "quantize", "compile", "channels_last"),
    "Loss and task": ("box", "cls", "cls_pw", "dfl", "pose", "kobj", "rle", "angle",
                      "mask_ratio", "overlap_mask", "dropout", "auto_augment", "erasing"),
    "Run controls": ("seed", "deterministic", "fraction", "save_period", "time"),
}
_TASK_PARAMETERS = {
    "pose": {"pose"}, "kobj": {"pose"}, "rle": {"pose"}, "angle": {"obb"},
    "mask_ratio": {"segment"}, "overlap_mask": {"segment"},
    "copy_paste": {"segment", "obb"}, "copy_paste_mode": {"segment", "obb"},
    "dropout": {"classify"}, "auto_augment": {"classify"}, "erasing": {"classify"},
    "box": {"detect", "segment", "pose", "obb"},
    "dfl": {"detect", "segment", "pose", "obb"},
    **{key: {"detect", "segment", "pose", "obb"} for key in
       ("mosaic", "close_mosaic", "mixup", "cutmix", "multi_scale", "degrees", "translate")},
}


def training_hyperparameter_groups(task="detect"):
    from ultralytics.utils import DEFAULT_CFG_DICT
    return {name: tuple(key for key in keys if key in DEFAULT_CFG_DICT
                       and (key not in _TASK_PARAMETERS or task in _TASK_PARAMETERS[key]))
            for name, keys in TRAINING_HYPERPARAMETER_GROUPS.items()}


def training_augmentation_preset(name, task="detect"):
    """Explicit user choices, limited to native training augmentation settings."""
    from ultralytics.utils import DEFAULT_CFG_DICT
    allowed = set(training_hyperparameter_groups(task)["Augmentations"])
    if name == "lighter":
        values = dict(mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, multi_scale=0.0)
    elif name == "defaults":
        values = {key: DEFAULT_CFG_DICT[key] for key in allowed}
    elif name == "mixing":
        values = dict(mosaic=1.0, mixup=0.1, cutmix=0.1, multi_scale=0.25)
    else:
        raise ValueError("Choose an augmentation preset.")
    return {key: value for key, value in values.items() if key in allowed}


def read_training_configuration(path):
    """Read the selected cfg, without generating files or changing its values."""
    from ultralytics.utils import YAML
    payload = YAML.load(path)
    if not isinstance(payload, dict):
        raise ValueError("Training configuration YAML must contain a mapping.")
    return payload


def training_hyperparameter_values(payload):
    from ultralytics.utils import DEFAULT_CFG_DICT
    values = dict(DEFAULT_CFG_DICT)
    values.update({key: value for key, value in payload.items() if key in values})
    return values


def save_training_hyperparameters(path, changes, write_atomic, *, data_yaml="", baseline=None):
    """Merge only edited native settings into the exact selected training cfg."""
    if data_yaml and Path(path).resolve() == Path(data_yaml).resolve():
        raise ValueError("Select the training configuration YAML, not the dataset YAML.")
    payload = read_training_configuration(path)
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG_DICT
    allowed = {key for keys in TRAINING_HYPERPARAMETER_GROUPS.values() for key in keys}
    if set(changes) - allowed:
        raise ValueError("Change basic training settings using Train Setup.")
    values = training_hyperparameter_values(payload)
    if baseline is not None:
        conflicts = [key for key in changes if values.get(key) != baseline.get(key)]
        if conflicts:
            raise ValueError("The YAML changed while editing. Reload before changing: " + ", ".join(conflicts))
    values.update(changes)
    for key, value in changes.items():
        if key not in DEFAULT_CFG_DICT:
            raise ValueError(f"This Ultralytics version does not support {key}.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite.")
    config = get_cfg(overrides=values)
    if config.quantize == 8 and config.compile:
        raise ValueError("INT8 QAT requires compile=False. Turn compilation off before saving.")
    if changes.get("quantize") == 8:
        import importlib.util
        if importlib.util.find_spec("modelopt") is None:
            raise ValueError("INT8 QAT requires NVIDIA ModelOpt (nvidia-modelopt>=0.44).")
    # Keep unknown/custom keys and all non-edited settings in the original file.
    # get_cfg canonicalizes precision aliases, including quantize='int8'.
    edited = {key: getattr(config, key) for key in changes}
    if any(payload.get(key) != value or key not in payload for key, value in edited.items()):
        payload.update(edited)
        write_atomic(str(path), yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return payload


def tune_target_path(data_yaml, use_recommended):
    """Choose the recommendations file or the exact selected dataset YAML."""
    if not data_yaml:
        return ""
    path = os.fspath(data_yaml)
    return str(Path(path).with_name("train_recommendations.yaml")) if use_recommended else path


def _load_dataset(path):
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid dataset YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return payload


def _darkfusion_settings(payload):
    settings = payload.get("darkfusion", {})
    if not isinstance(settings, dict):
        raise ValueError("Dataset YAML 'darkfusion' must be a mapping.")
    return settings


def _validated_hyperparameters(hyperparams):
    if not isinstance(hyperparams, Mapping):
        raise ValueError("Tuned hyperparameters must be a mapping.")
    result = dict(hyperparams)
    for key, value in result.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Tuned hyperparameter names must be nonempty strings.")
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise ValueError(f"Tuned hyperparameter {key!r} must be a scalar.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Tuned hyperparameter {key!r} must be finite.")
    return result


def read_dataset_tuned_hyperparameters(data_yaml):
    """Read the namespaced parameters; a missing file or absent block yields {}."""
    if not data_yaml:
        return {}
    path = Path(data_yaml)
    try:
        payload = _load_dataset(path)
    except FileNotFoundError:
        return {}
    settings = _darkfusion_settings(payload)
    return _validated_hyperparameters(settings.get("tuned_hyperparameters", {}))


def write_dataset_tuned_hyperparameters(data_yaml, hyperparams, source_path=""):
    """Atomically update DarkFusion tuning metadata in an existing dataset YAML.

    Existing dataset keys and other DarkFusion settings retain their values.
    Missing or malformed datasets are refused before creating any output.
    """
    if not data_yaml:
        raise ValueError("A dataset YAML path is required.")
    path = Path(data_yaml)
    payload = _load_dataset(path)
    settings = dict(_darkfusion_settings(payload))
    settings["tuned_hyperparameters"] = _validated_hyperparameters(hyperparams)
    settings["tune_best_path"] = os.fspath(source_path) if source_path else ""
    payload["darkfusion"] = settings
    serialized = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=str(path.parent),
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
    return os.fspath(data_yaml)
