from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

try:
    import psutil
except Exception:
    psutil = None


def _configure_text_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _format_command(command: list[str]) -> str:
    try:
        return subprocess.list2cmdline(command) if sys.platform.startswith("win") else " ".join(command)
    except Exception:
        return " ".join(str(part) for part in command)


def main() -> int:
    _configure_text_streams()

    parser = argparse.ArgumentParser(description="Run a command and mirror output to a DarkFusion log file.")
    parser.add_argument("--log", required=True, help="Path to the log file.")
    parser.add_argument("--cwd", default="", help="Working directory for the command.")
    parser.add_argument("--max-commit-gb", type=float, default=0.0, help="Terminate the process tree above this committed virtual memory.")
    parser.add_argument("--min-system-available-gb", type=float, default=0.0, help="Terminate when available system RAM drops below this value.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --.")
    args = parser.parse_args()

    command = list(args.command or [])
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("No command was supplied.", file=sys.stderr)
        return 2

    log_path = os.path.abspath(args.log)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cwd = os.path.abspath(args.cwd) if args.cwd else None

    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log_lock = threading.Lock()

        def emit(text: str = "", end: str = "\n") -> None:
            with log_lock:
                print(text, end=end, flush=True)
                log.write(text + end)
                log.flush()

        header = [
            "",
            "=" * 80,
            f"DarkFusion command started: {datetime.now().isoformat(timespec='seconds')}",
            f"Working directory: {cwd or os.getcwd()}",
            f"Command: {_format_command(command)}",
            "=" * 80,
            "",
        ]
        for line in header:
            emit(line)

        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        stop_monitor = threading.Event()
        termination_reason: list[str] = []

        def terminate_process_tree(reason: str) -> None:
            termination_reason.append(reason)
            emit("")
            emit(f"DarkFusion memory guard: {reason}")
            try:
                parent = psutil.Process(process.pid) if psutil is not None else None
                children = parent.children(recursive=True) if parent is not None else []
                for child in children:
                    try:
                        child.terminate()
                    except Exception:
                        pass
                process.terminate()
                gone, alive = psutil.wait_procs(children, timeout=8) if psutil is not None else ([], [])
                for child in alive:
                    try:
                        child.kill()
                    except Exception:
                        pass
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
            except Exception as exc:
                emit(f"DarkFusion memory guard could not terminate cleanly: {exc}")
                try:
                    process.kill()
                except Exception:
                    pass

        def process_tree_vms_bytes() -> int:
            if psutil is None:
                return 0
            total = 0
            try:
                parent = psutil.Process(process.pid)
                processes = [parent, *parent.children(recursive=True)]
            except Exception:
                return 0
            for proc in processes:
                try:
                    total += int(proc.memory_info().vms)
                except Exception:
                    continue
            return total

        def memory_monitor() -> None:
            if psutil is None:
                return
            max_commit_bytes = int(max(0.0, float(args.max_commit_gb or 0.0)) * (1024 ** 3))
            min_available_bytes = int(max(0.0, float(args.min_system_available_gb or 0.0)) * (1024 ** 3))
            while not stop_monitor.wait(2.0):
                if process.poll() is not None:
                    return
                total_vms = process_tree_vms_bytes()
                available = int(psutil.virtual_memory().available)
                if max_commit_bytes and total_vms > max_commit_bytes:
                    terminate_process_tree(
                        f"process tree committed {total_vms / (1024 ** 3):.1f} GB, above the {args.max_commit_gb:.1f} GB limit"
                    )
                    return
                if min_available_bytes and available < min_available_bytes:
                    terminate_process_tree(
                        f"system available RAM dropped to {available / (1024 ** 3):.1f} GB, below the {args.min_system_available_gb:.1f} GB limit"
                    )
                    return

        monitor_thread = None
        if psutil is not None and (args.max_commit_gb > 0 or args.min_system_available_gb > 0):
            monitor_thread = threading.Thread(target=memory_monitor, name="darkfusion-memory-guard", daemon=True)
            monitor_thread.start()

        assert process.stdout is not None
        for line in process.stdout:
            emit(line.rstrip("\n"), end="\n")

        exit_code = process.wait()
        stop_monitor.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=3)
        if termination_reason and exit_code == 0:
            exit_code = 137
        footer = [
            "",
            "=" * 80,
            f"DarkFusion command finished: {datetime.now().isoformat(timespec='seconds')}",
            f"Exit code: {exit_code}",
            "=" * 80,
        ]
        for line in footer:
            emit(line)
        return int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
