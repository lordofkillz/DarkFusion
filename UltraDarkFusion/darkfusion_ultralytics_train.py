"""Launch Ultralytics training with DarkFusion training-time patches."""

from __future__ import annotations

import os
import sys
import json
import math
import shutil
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps


def verify_image_label_keep_duplicates(args: tuple) -> list:
    """Ultralytics label verifier with exact duplicate-row removal disabled."""
    from ultralytics.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, exif_size
    from ultralytics.utils.ops import segments2boxes

    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls = args
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        shape = (shape[1], shape[0])
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: corrupt JPEG restored and saved"

        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]

                assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
                assert lb.min() >= -0.01, f"negative class labels or coordinate {lb[lb < -0.01]}"
                max_cls = 0 if single_cls else lb[:, 0].max()
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
            else:
                ne = 1
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)

        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)

        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def _raise_cache_miss(*_args, **_kwargs):
    raise FileNotFoundError("DarkFusion forced an in-memory Ultralytics label recache.")


def _skip_cache_save(prefix, path, x, version):
    if isinstance(x, dict):
        x["version"] = version
    return None


def install_keep_duplicate_label_patch() -> None:
    """Disable Ultralytics exact duplicate label row removal for this process."""
    import ultralytics.data.dataset as dataset
    import ultralytics.data.utils as data_utils

    data_utils.verify_image_label = verify_image_label_keep_duplicates
    dataset.verify_image_label = verify_image_label_keep_duplicates
    dataset.load_dataset_cache_file = _raise_cache_miss
    dataset.save_dataset_cache_file = _skip_cache_save


def _finite_number(value) -> bool:
    """Return True for finite scalar-like values without raising on tensors."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value.detach()).all().item())
    except Exception:
        pass
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _trainer_values_are_finite(trainer) -> bool:
    values = []
    for name in ("loss", "fitness"):
        value = getattr(trainer, name, None)
        if value is not None:
            values.append(value)
    tloss = getattr(trainer, "tloss", None)
    if isinstance(tloss, dict):
        values.extend(tloss.values())
    metrics = getattr(trainer, "metrics", None)
    if isinstance(metrics, dict):
        values.extend(value for value in metrics.values() if isinstance(value, (int, float, np.number)))
    return all(_finite_number(value) for value in values) if values else True


def _write_training_health(trainer, state: str, detail: str = "") -> None:
    payload = {
        "state": str(state),
        "detail": str(detail or ""),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": int(getattr(trainer, "epoch", -1)) + 1,
        "fitness": float(getattr(trainer, "fitness", 0.0) or 0.0)
        if _finite_number(getattr(trainer, "fitness", None))
        else None,
    }
    try:
        path = os.path.join(str(trainer.save_dir), ".darkfusion_training_health.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception:
        pass


def _preserve_healthy_checkpoint(trainer) -> None:
    """Copy the latest completed checkpoint after the epoch passes health checks."""
    try:
        last_path = str(trainer.last)
        healthy_path = os.path.join(str(trainer.wdir), "darkfusion_healthy.pt")
        if os.path.isfile(last_path):
            shutil.copy2(last_path, healthy_path)
            _write_training_health(
                trainer,
                "healthy",
                "Latest healthy checkpoint preserved as weights/darkfusion_healthy.pt.",
            )
    except Exception as error:
        try:
            from ultralytics.utils import LOGGER

            LOGGER.warning(f"DarkFusion could not preserve the healthy checkpoint: {error}")
        except Exception:
            pass


def install_darkfusion_training_callbacks() -> None:
    """Install safe-stop, frozen-BN, and abnormal-run protection callbacks."""
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils import LOGGER

    if getattr(BaseTrainer, "_darkfusion_stop_controls_installed", False):
        return

    original_run_callbacks = BaseTrainer.run_callbacks

    def marker_path(trainer, name: str) -> str:
        return os.path.join(str(trainer.save_dir), name)

    def acknowledge(trainer, request: str, point: str) -> None:
        request_path = marker_path(trainer, request)
        try:
            os.remove(request_path)
        except OSError:
            pass
        payload = {
            "request": request,
            "acknowledged_at": datetime.now().isoformat(timespec="seconds"),
            "epoch": int(getattr(trainer, "epoch", -1)) + 1,
            "point": point,
        }
        try:
            with open(marker_path(trainer, ".darkfusion_stop_ack.json"), "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            pass

    def run_callbacks_with_stop_control(trainer, event: str):
        health = getattr(trainer, "_darkfusion_health_state", None)
        if not isinstance(health, dict):
            health = {
                "best_fitness": None,
                "collapse_epochs": 0,
                "nonfinite_epochs": 0,
                "frozen_bn_count": None,
                "frozen_bn_epoch": None,
            }
            trainer._darkfusion_health_state = health

        if event == "on_train_start":
            _write_training_health(trainer, "running", "Training started normally.")

        if event == "on_train_batch_start":
            # Parameters marked requires_grad=False are frozen, but BatchNorm
            # running statistics can otherwise continue changing. Ultralytics
            # puts the full model in train mode after on_train_epoch_start, so
            # enforce this at the batch boundary where it remains effective.
            current_epoch = int(getattr(trainer, "epoch", -1))
            if health.get("frozen_bn_epoch") == current_epoch:
                original_run_callbacks(trainer, event)
                return
            health["frozen_bn_epoch"] = current_epoch
            try:
                import torch

                frozen_bn_count = 0
                for module in trainer.model.modules():
                    if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        continue
                    parameters = list(module.parameters(recurse=False))
                    if parameters and all(not parameter.requires_grad for parameter in parameters):
                        module.eval()
                        frozen_bn_count += 1
                if health["frozen_bn_count"] is None:
                    health["frozen_bn_count"] = frozen_bn_count
                    if frozen_bn_count:
                        LOGGER.info(
                            f"DarkFusion protected {frozen_bn_count} frozen BatchNorm layer(s) from statistics updates."
                        )
            except Exception as error:
                LOGGER.warning(f"DarkFusion frozen BatchNorm protection could not be applied: {error}")

        original_run_callbacks(trainer, event)
        stop_now = marker_path(trainer, ".darkfusion_stop_now")
        stop_after_epoch = marker_path(trainer, ".darkfusion_stop_after_epoch")

        if event == "on_train_batch_end" and os.path.exists(stop_now):
            trainer.stop = True
            acknowledge(trainer, ".darkfusion_stop_now", "after current batch")
            LOGGER.warning(
                "DarkFusion Stop Now requested. Finishing this epoch's validation and saving a clean checkpoint."
            )
        elif event == "on_fit_epoch_end":
            requested = None
            if os.path.exists(stop_now):
                requested = ".darkfusion_stop_now"
            elif os.path.exists(stop_after_epoch):
                requested = ".darkfusion_stop_after_epoch"
            if requested:
                trainer.stop = True
                acknowledge(trainer, requested, "after checkpoint save")
                LOGGER.warning("DarkFusion stop requested. Current epoch and checkpoint are complete.")

        if event == "on_fit_epoch_end":
            # Ultralytics emits this event once more during final best-model
            # validation after the training epoch loop has completed.
            if int(getattr(trainer, "epoch", -1)) >= int(getattr(trainer, "epochs", 0)):
                return
            finite = _trainer_values_are_finite(trainer)
            if not finite:
                health["nonfinite_epochs"] += 1
                health["collapse_epochs"] = 0
                detail = (
                    "Non-finite loss or validation metrics remained after Ultralytics recovery "
                    f"for {health['nonfinite_epochs']} completed epoch(s)."
                )
                _write_training_health(trainer, "abnormal", detail)
                LOGGER.warning(f"DarkFusion training watchdog: {detail}")
                if health["nonfinite_epochs"] >= 2:
                    trainer.stop = True
                    _write_training_health(
                        trainer,
                        "stopped_abnormal",
                        detail + " Training stopped; use darkfusion_healthy.pt or best.pt.",
                    )
            else:
                health["nonfinite_epochs"] = 0
                fitness = float(getattr(trainer, "fitness", 0.0) or 0.0)
                prior_best = health.get("best_fitness")
                if prior_best is None or fitness > prior_best:
                    health["best_fitness"] = fitness
                    health["collapse_epochs"] = 0
                    _preserve_healthy_checkpoint(trainer)
                else:
                    epoch_number = int(getattr(trainer, "epoch", -1)) + 1
                    catastrophic = prior_best >= 0.05 and fitness < prior_best * 0.20 and epoch_number >= 5
                    health["collapse_epochs"] = health["collapse_epochs"] + 1 if catastrophic else 0
                    if not catastrophic:
                        _preserve_healthy_checkpoint(trainer)
                    if catastrophic:
                        LOGGER.warning(
                            "DarkFusion training watchdog: validation fitness is below 20%% of the "
                            "best observed value (%s/%s). Collapse count %s/3.",
                            f"{fitness:.5g}",
                            f"{prior_best:.5g}",
                            health["collapse_epochs"],
                        )
                    if health["collapse_epochs"] >= 3:
                        trainer.stop = True
                        detail = (
                            "Validation fitness stayed below 20% of its best value for three "
                            "consecutive completed epochs. This is separate from patience."
                        )
                        _write_training_health(trainer, "stopped_collapse", detail)
                        LOGGER.warning(
                            "DarkFusion stopped a persistently collapsed run. "
                            "Use weights/darkfusion_healthy.pt or weights/best.pt."
                        )

    BaseTrainer.run_callbacks = run_callbacks_with_stop_control
    BaseTrainer._darkfusion_stop_controls_installed = True


def main() -> None:
    keep_duplicates_flag = "--darkfusion-keep-duplicates"
    keep_duplicates = keep_duplicates_flag in sys.argv
    sys.argv = [argument for argument in sys.argv if argument != keep_duplicates_flag]
    if keep_duplicates:
        install_keep_duplicate_label_patch()
    install_darkfusion_training_callbacks()
    from ultralytics.cfg import entrypoint

    sys.argv = ["yolo", *sys.argv[1:]]
    entrypoint()


if __name__ == "__main__":
    main()
