"""Read live tuning metadata without importing Ray, torch, or Qt.

The experiment owns process/completion state; this module only selects a trial
whose ordinary Ultralytics CSV can be displayed. Epoch counters stay per trial.
"""

import json
import os
from pathlib import Path
import re


_MAX_ENTRIES = 2000
_MAX_JSON_BYTES = 16 * 1024 * 1024
_TRIAL_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _children(path):
    try:
        with os.scandir(path) as entries:
            result = []
            for index, entry in enumerate(entries):
                if index >= _MAX_ENTRIES:
                    break
                result.append(Path(entry.path))
            return result
    except OSError:
        return []


def _mtime(path):
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _json(path):
    try:
        with path.open("rb") as handle:
            data = handle.read(_MAX_JSON_BYTES + 1)
        if len(data) > _MAX_JSON_BYTES:
            return {}
        value = json.loads(data)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _object(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, UnicodeError):
            return {}
    return value if isinstance(value, dict) else {}


def _last_result(path):
    """Ray appends JSON lines; ignore a final row still being written."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 128 * 1024))
            lines = handle.read().splitlines()
        for line in reversed(lines):
            try:
                result = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if isinstance(result, dict):
                return result
    except OSError:
        pass
    return {}


def _ray_files(root):
    children = _children(root)
    states = sorted(
        (p for p in children if p.name.startswith("experiment_state") and p.suffix == ".json"),
        key=_mtime, reverse=True,
    )
    trials = [p for p in children if p.name.startswith(("_tune_", "tune_")) and p.is_dir()]
    return states, trials


def _experiment_root(root):
    """Ray can increment the name because the console logger creates root first."""
    candidates = [root]
    candidates.extend(
        path for path in _children(root.parent)
        if path.name.startswith(root.name)
        and path.name[len(root.name):].isdigit() and path.is_dir()
    )
    recognized = []
    for path in candidates:
        states, trials = _ray_files(path)
        if states or trials or (path / "tuner.pkl").is_file():
            modified = max((_mtime(p) for p in states + trials), default=_mtime(path))
            recognized.append((modified, path, states, trials))
    if recognized:
        _, path, states, trials = max(recognized, key=lambda value: value[0])
        return path, states, trials, True
    return root, [], [], False


def _nested_metrics(trial_dir):
    """Support older Ray output layouts with a bounded three-level scan."""
    level = [trial_dir]
    found = []
    visited = 0
    for _ in range(4):
        next_level = []
        for directory in level:
            visited += 1
            if visited > 80:
                return max(found, key=lambda p: _mtime(p / "results.csv")) if found else None
            if (directory / "results.csv").is_file():
                found.append(directory)
            next_level.extend(
                p for p in _children(directory)
                if p.is_dir() and not p.is_symlink()
                and p.name not in {"weights", "__pycache__"} and not p.name.startswith(".")
            )
        level = next_level
    return max(found, key=lambda p: _mtime(p / "results.csv")) if found else None


def _metrics_dir(root, experiment, trial):
    suffix = trial["id"].split("_")[-1]
    expected = root.parent / f"{root.name}_{suffix}"
    candidates = [expected, experiment.parent / f"{experiment.name}_{suffix}"]
    for path in candidates:
        if (path / "results.csv").is_file() or (path / "args.yaml").is_file():
            return path
    nested = _nested_metrics(trial["directory"])
    return nested or expected


def _snapshot(trials, total):
    active = [trial for trial in trials if trial["status"] == "RUNNING"]
    selected = max(active or trials, key=lambda trial: trial["updated"], default={})
    trial_id = selected.get("id", "")
    number = selected.get("number", 0)
    total = max(total, len(trials))
    label = f"Trial {number}/{total} ({trial_id})" if trial_id else "Waiting for first trial"
    if len(active) > 1:
        label += f" · {len(active)} active"
    return {
        "metrics_dir": str(selected["metrics_dir"]) if selected else "",
        "trial_label": label,
        "trial_id": trial_id,
        "total_trials": total,
        "completed_trials": sum(t["status"] == "TERMINATED" for t in trials),
        "failed_trials": sum(t["status"] == "ERROR" for t in trials),
        "trials_started": sum(t["status"] != "PENDING" for t in trials),
        "trial_status": selected.get("status", ""),
        "active_trials": len(active),
    }


def tuning_snapshot(run_dir, log_text="", iterations=0):
    """Return a trial CSV location and counts, never overall completion state.

    ``iterations`` also identifies a just-launched tune before Ray writes its
    metadata. A selected trial's metrics directory may not exist yet: callers
    should clear the previous chart and wait for that trial's first epoch.
    """
    if not run_dir:
        return {}
    root = Path(run_dir)
    try:
        total = max(0, int(iterations or 0))
    except (TypeError, ValueError, OverflowError):
        total = 0
    text = _ANSI.sub("", str(log_text or ""))
    if not total:
        announced = re.search(r"(?:Iterations:\s*|--iterations(?:=|\s+))(\d+)", text, re.I)
        if announced:
            total = int(announced.group(1))
    experiment, states, folders, is_ray = _experiment_root(root)
    trials = {}
    for state_path in states[:3]:
        state = _json(state_path)
        entries = state.get("trial_data", state.get("checkpoints", []))
        if not isinstance(entries, list) or not entries:
            continue
        for entry in entries[:_MAX_ENTRIES]:
            record = _object(entry[0] if isinstance(entry, (list, tuple)) and entry else entry)
            trial_id = str(record.get("trial_id", ""))
            if not _TRIAL_ID.fullmatch(trial_id):
                continue
            folder = str(record.get("relative_logdir") or f"_tune_{trial_id}")
            if Path(folder).name != folder:
                folder = f"_tune_{trial_id}"
            trials[trial_id] = {
                "id": trial_id,
                "status": str(record.get("status", "")).upper(),
                "directory": experiment / folder,
                "number": len(trials) + 1,
            }
        if trials:
            break
    for folder in folders:
        trial_id = re.sub(r"^_?tune_", "", folder.name)
        if _TRIAL_ID.fullmatch(trial_id) and trial_id not in trials:
            trials[trial_id] = {"id": trial_id, "status": "", "directory": folder,
                                "number": len(trials) + 1}
    for trial in trials.values():
        directory = trial["directory"]
        result = _last_result(directory / "result.json")
        if (directory / "error.txt").is_file() or (directory / "error.pkl").is_file():
            trial["status"] = "ERROR"
        elif result.get("done") is True:
            trial["status"] = "TERMINATED"
        trial["metrics_dir"] = _metrics_dir(root, experiment, trial)
        csv_time = _mtime(trial["metrics_dir"] / "results.csv")
        args_time = _mtime(trial["metrics_dir"] / "args.yaml")
        if not trial["status"] and (csv_time or args_time):
            trial["status"] = "RUNNING"
        trial["updated"] = csv_time or args_time or _mtime(directory)
    if is_ray:
        return _snapshot(list(trials.values()), total)

    starts = list(re.finditer(r"Starting iteration\s+(\d+)\s*/\s*(\d+)", text, re.I))
    builtin = (root / "tune_results.csv").is_file() or (root / "tune_results.ndjson").is_file()
    if not starts and not builtin and not total:
        return {}
    snapshot = _snapshot([], total)
    if starts:
        start = starts[-1]
        number, announced_total = map(int, start.groups())
        snapshot.update(total_trials=max(total, announced_total), trials_started=number,
                        trial_id=str(number), trial_status="RUNNING",
                        trial_label=f"Trial {number}/{max(total, announced_total)}")
        # Only use paths logged after this trial started, never the prior trial.
        paths = re.findall(r"Logging results to\s+([^\r\n]+)", text[start.end():], re.I)
        if paths:
            path = Path(paths[-1].strip().strip("\"'"))
            if not path.is_absolute():
                path = Path.cwd() / path
            snapshot["metrics_dir"] = str(path)
    return snapshot
