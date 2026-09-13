"""Run a command in its own Windows console without interpreting it as shell code."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def launch_console_command(command, cwd=None, keep_open=True):
    """Preserve argument boundaries and optionally keep the finished console open."""
    if not command:
        raise ValueError("No command to launch.")
    arguments = [sys.executable, str(Path(__file__).resolve())]
    if keep_open:
        arguments.append("--keep-open")
    arguments.extend(["--", *(str(argument) for argument in command)])
    return subprocess.Popen(arguments, cwd=cwd, creationflags=subprocess.CREATE_NEW_CONSOLE)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    try:
        # This string is only displayed; execution always uses the argument list.
        print(subprocess.list2cmdline(command), flush=True)
        code = subprocess.call(command)
    except FileNotFoundError as error:
        print(f"Could not find the command: {error}", file=sys.stderr, flush=True)
        code = 127
    except OSError as error:
        print(f"Could not start the command: {error}", file=sys.stderr, flush=True)
        code = 1
    except KeyboardInterrupt:
        code = 130

    print(f"Command finished with exit code {code}.", flush=True)
    if args.keep_open and sys.stdin is not None and sys.stdin.isatty():
        try:
            input("Press Enter to close this window...")
        except (EOFError, KeyboardInterrupt):
            pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
