"""Training completion estimates from Ultralytics' cumulative epoch timings."""

import math
from datetime import datetime


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _duration(seconds):
    minutes = max(1, math.ceil(seconds / 60))
    days, minutes = divmod(minutes, 1440)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def training_eta(rows, target_epochs, *, status="running", now,
                 last_epoch_at=None, starting_epoch=None, mode="train"):
    """Return display text and timing data; never infer completion from a timer.

    ``time`` in results.csv is cumulative within one trainer process, and resets
    on resume. ``starting_epoch`` excludes old rows during a known resumed run.
    ``last_epoch_at`` is the CSV modification time, not when the UI was opened.
    """
    target = max(0, int(_number(target_epochs) or 0))
    points = []
    for row in rows or []:
        epoch = _number(row.get("epoch"))
        elapsed = _number(row.get("time"))
        if epoch is not None and epoch >= 1 and epoch.is_integer():
            points.append((int(epoch), elapsed))
    completed = points[-1][0] if points else 0
    progress = f"Epoch {completed}/{target}" if target else f"Epoch {completed}"

    def result(text, detail=progress, remaining=None, average=None):
        return {"text": text, "detail": detail, "remaining_seconds": remaining,
                "seconds_per_epoch": average, "completed_epochs": completed}

    if mode != "train":
        return result("Estimated completion: unavailable", "Epoch estimates are shown during training.")
    if status == "complete":
        return result("Training complete", progress, 0)
    if status == "failed":
        return result("Training failed", f"{progress} · no active estimate")
    if status != "running":
        return result("Training stopped" if completed else "Estimated completion: waiting for training",
                      f"{progress} · no active estimate")
    if target and completed >= target:
        return result("Finishing training…", f"{progress} · final validation and saving")
    if not target:
        return result("Estimating completion…", "Waiting for the run's epoch target.")

    # Drop the historical segment when the cumulative timer restarts, even
    # when the user reopens a running session without a stored launch context.
    durations = []
    previous_epoch, previous_time = starting_epoch or 0, 0.0
    for epoch, elapsed in points:
        if starting_epoch is not None and epoch <= starting_epoch:
            continue
        if elapsed is None or elapsed <= 0:
            continue
        if epoch <= previous_epoch or elapsed <= previous_time:
            durations = []
            previous_epoch, previous_time = epoch - 1, 0.0
        epoch_delta = epoch - previous_epoch
        if epoch_delta > 0:
            # Without the start epoch, an isolated resumed row cannot tell us
            # how many epochs its cumulative timestamp covers.
            if previous_time > 0 or epoch_delta == 1 or starting_epoch is not None:
                durations.append((elapsed - previous_time) / epoch_delta)
        previous_epoch, previous_time = epoch, elapsed
    durations = [value for value in durations[-5:] if value > 0]
    if not durations:
        detail = "Waiting for a completed epoch with timing."
        if starting_epoch:
            detail = "Waiting for the first completed epoch after resume."
        return result("Estimating completion…", f"{progress} · {detail}")

    average = sum(durations) / len(durations)
    since_epoch = max(0.0, now - last_epoch_at) if last_epoch_at is not None else 0.0
    remaining_epochs = target - completed
    if remaining_epochs == 1 and since_epoch >= average:
        return result("Finishing last epoch…", f"{progress} · estimate updating as this epoch finishes",
                      average=average)
    remaining = max(1.0, average * remaining_epochs - min(since_epoch, average))
    finish = datetime.fromtimestamp(now + remaining).strftime("%a %I:%M %p").replace(" 0", " ")
    detail = f"{progress} · recent average {average:.0f}s/epoch · finish ≈ {finish}"
    return result(f"Estimated remaining: {_duration(remaining)}", detail, remaining, average)
