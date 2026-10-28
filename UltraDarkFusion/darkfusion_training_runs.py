"""Identify DarkFusion training processes without matching unrelated jobs."""

import ntpath
import os


def path_identity(path):
    """Compare Windows paths consistently, including slash/case differences."""
    value = str(path or "").strip().strip('"').strip("'")
    if not value:
        return ""
    if ntpath.isabs(value) and (ntpath.splitdrive(value)[0] or "\\" in value):
        return ntpath.normcase(ntpath.normpath(value.replace("/", "\\")))
    return os.path.normcase(os.path.realpath(os.path.abspath(value)))


def is_training_command(command):
    tokens = [str(part).strip().lower() for part in (command or [])]
    # A tuning/calibration helper may contain training arguments, but is not
    # the training run controlled by Stop/Resume.
    helpers = {ntpath.basename(part) for part in tokens}
    if helpers & {"darkfusion_ultralytics_tune.py", "darkfusion_ultralytics_batch.py"}:
        return False
    return "train" in tokens or "mode=train" in tokens


def _option(command, key):
    for index, part in enumerate(command):
        part = str(part)
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
        if part == key and index + 1 < len(command):
            return str(command[index + 1])
    return ""


def process_matches_training_record(command, record):
    command = list(command or [])
    scripts = {ntpath.basename(str(part)).lower() for part in command}
    if not is_training_command(command):
        return False
    if "darkfusion_command_runner.py" in scripts:
        # A runner belongs to exactly one captured log. Never fall back to a
        # basename such as 'train', or even a run-dir match, if its log differs.
        expected = path_identity((record or {}).get("log_path"))
        return bool(expected and path_identity(_option(command, "--log")) == expected)
    if "darkfusion_ultralytics_train.py" not in scripts:
        return False
    project, name = _option(command, "project"), _option(command, "name")
    expected = path_identity((record or {}).get("run_dir"))
    if not expected:
        return False
    if not project or not name:
        # Resume inherits project/name from the checkpoint, not the CLI.
        model = _option(command, "model")
        parent = ntpath.dirname(model)
        return (_option(command, "resume").lower() == "true"
                and ntpath.basename(parent).lower() == "weights"
                and path_identity(ntpath.dirname(parent)) == expected)
    run = ntpath.join(project, name) if ntpath.splitdrive(project)[0] else os.path.join(project, name)
    return path_identity(run) == expected


def training_processes_for_record(record, process_api):
    """Find verified runner roots (or direct trainers), never trust a PID alone."""
    if process_api is None:
        return []
    matches = {}

    def consider(process):
        try:
            if not process.is_running():
                return
            command = process.cmdline()
            if not process_matches_training_record(command, record):
                return
            if process.pid == (record or {}).get("pid") and record.get("process_created_at") is not None:
                if abs(process.create_time() - float(record["process_created_at"])) > 0.01:
                    return
            matches[process.pid] = (process, command)
        except Exception:
            # Processes can exit or become inaccessible during enumeration.
            return

    if (record or {}).get("pid"):
        try:
            consider(process_api.Process(int(record["pid"])))
        except Exception:
            pass
    try:
        for process in process_api.process_iter(["pid", "name", "cmdline"]):
            consider(process)
    except Exception:
        pass
    runners = [process for process, command in matches.values()
               if any(ntpath.basename(str(part)).lower() == "darkfusion_command_runner.py" for part in command)]
    return runners or [process for process, _command in matches.values()]
