"""Persist dataset tuning choices without changing dataset paths or class metadata."""

from collections.abc import Mapping
import math
import os
from pathlib import Path
import tempfile

import yaml


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
