"""Strict offline DINOv3 loading with a metadata-keyed checksum memo.

The package and this loader remain lightweight until ``load_model`` succeeds at
validating a local checkpoint. No remote code or model download is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .config import MODEL_CONFIGS


class CheckpointError(RuntimeError):
    """A required official checkpoint is absent, invalid, or changing."""


def _canonical_path(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def checkpoint_fingerprint(path):
    canonical = _canonical_path(path)
    stat = os.stat(canonical)
    return (canonical, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _read_memo(path, model_key, checkpoint):
    if path is None:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        fingerprint = value.get("fingerprint")
        if (value.get("model_key") == model_key
                and value.get("sha256") == MODEL_CONFIGS[model_key]["sha256"]
                and isinstance(fingerprint, list) and len(fingerprint) == 4
                and fingerprint[0] == _canonical_path(checkpoint)
                and all(isinstance(item, int) for item in fingerprint[1:])):
            return tuple(fingerprint)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def cache_fingerprint(checkpoint, model_key, memo_path=None):
    """Use current metadata, or the last verified metadata if weights moved.

    Reusing descriptors does not require the checkpoint. Actual inference still
    requires the original checkpoint and a successful integrity check.
    """
    try:
        return checkpoint_fingerprint(checkpoint)
    except FileNotFoundError:
        verified = _read_memo(memo_path, model_key, checkpoint)
        return verified or (_canonical_path(checkpoint), "missing")
    except OSError:
        return (_canonical_path(checkpoint), "unavailable")


def _write_memo(path, model_key, fingerprint):
    if path is None:
        return
    temporary = None
    try:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = dict(model_key=model_key, sha256=MODEL_CONFIGS[model_key]["sha256"],
                    fingerprint=fingerprint)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=destination.name + ".",
                                         suffix=".tmp", dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream)
        os.replace(temporary, destination)
    except OSError:
        # A read-only cache must not prevent verified local inference.
        pass
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def verify_checkpoint(model_key, checkpoint, *, memo_path=None, status=None,
                      check_cancelled=None, expected_fingerprint=None):
    """Hash new/replaced files once, and reuse verification for unchanged files."""
    spec = MODEL_CONFIGS[model_key]
    check_cancelled = check_cancelled or (lambda: None)
    status = status or (lambda message: None)
    check_cancelled()
    try:
        before = checkpoint_fingerprint(checkpoint)
    except OSError as exc:
        raise CheckpointError(f"Missing or unreadable {spec['label']} weights: {checkpoint}") from exc
    if expected_fingerprint is not None and before != tuple(expected_fingerprint):
        raise CheckpointError("The DINOv3 checkpoint changed during this scan. Start the scan again.")
    if before[1] != spec["size"]:
        raise CheckpointError(f"{spec['label']} checkpoint has the wrong size. Restore the official file: {checkpoint}")
    if _read_memo(memo_path, model_key, checkpoint) != before:
        status(f"Verifying {spec['label']} checkpoint (once for this file)...")
        digest = hashlib.sha256()
        with Path(checkpoint).open("rb") as stream:
            while True:
                check_cancelled()
                block = stream.read(4 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        check_cancelled()
        if checkpoint_fingerprint(checkpoint) != before:
            raise CheckpointError("The DINOv3 checkpoint changed while being verified. Start the scan again.")
        if digest.hexdigest() != spec["sha256"]:
            raise CheckpointError(f"{spec['label']} checkpoint checksum failed. Restore the official file: {checkpoint}")
        _write_memo(memo_path, model_key, before)
    return before


def load_model(model_key, checkpoint, *, memo_path=None, status=None,
               check_cancelled=None, expected_fingerprint=None):
    """Return the native CPU eval model; callers choose GPU/CPU placement."""
    check_cancelled = check_cancelled or (lambda: None)
    fingerprint = verify_checkpoint(model_key, checkpoint, memo_path=memo_path,
                                    status=status, check_cancelled=check_cancelled,
                                    expected_fingerprint=expected_fingerprint)
    check_cancelled()
    import torch
    from .vision_transformer import DinoVisionTransformer

    spec = MODEL_CONFIGS[model_key]
    # Matches Meta's LVD1689M dinov3_vitb16/dinov3_vitl16 constructors. All
    # parameters and persistent buffers are supplied by a strict state dict.
    model = DinoVisionTransformer(
        img_size=224, patch_size=16, in_chans=3,
        pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32",
        embed_dim=spec["embedding_size"], depth=spec["depth"], num_heads=spec["heads"],
        ffn_ratio=4, qkv_bias=True, drop_path_rate=0.0, layerscale_init=1.0e-5,
        norm_layer="layernormbf16", ffn_layer="mlp", ffn_bias=True, proj_bias=True,
        n_storage_tokens=4, mask_k_bias=True, untie_global_and_local_cls_norm=False,
    )
    check_cancelled()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    check_cancelled()
    model.load_state_dict(state, strict=True)
    del state
    if checkpoint_fingerprint(checkpoint) != fingerprint:
        raise CheckpointError("The DINOv3 checkpoint changed while loading. Start the scan again.")
    return model.eval().requires_grad_(False)
