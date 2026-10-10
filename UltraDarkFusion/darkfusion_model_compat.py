"""Model API compatibility without editing installed ML packages.

Transformers 5 returns pooled feature outputs instead of bare CLIP tensors and
removed the private BERT helpers used by groundingdino-py 0.4. Keep the original
checkpoint keys and sentence-specific attention mask in both runtimes.
"""

from __future__ import annotations

import threading


_GROUNDING_BUILD_LOCK = threading.RLock()


def pooled_features(output):
    """Return projected embeddings from either Transformers 4 or 5."""
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(output, dict) and "pooler_output" in output:
        return output["pooler_output"]
    return output


def _grounding_bert_inputs(_module, args, kwargs):
    """Preserve DINO's per-sentence 3D mask for the public BERT forward API."""
    import torch

    mask = kwargs.get("attention_mask")
    if mask is not None and mask.ndim == 3:
        # Transformers 5 accepts a prepared [batch, heads, query, key] mask.
        # Its eager/SDPA backends expect additive values, not 0/1 floats.
        dtype = _module.dtype
        allowed = mask.to(dtype=torch.bool).unsqueeze(1)
        kwargs = dict(kwargs)
        kwargs["attention_mask"] = torch.zeros(
            allowed.shape, device=allowed.device, dtype=dtype
        ).masked_fill(~allowed, torch.finfo(dtype).min)
    return args, kwargs


def grounding_bert_wrapper(bert_model, legacy_wrapper):
    """Use the original wrapper on v4; the public BERT model on v5."""
    if hasattr(bert_model, "get_extended_attention_mask"):
        return legacy_wrapper(bert_model)
    if not getattr(bert_model, "_darkfusion_sentence_mask_hook", False):
        bert_model.register_forward_pre_hook(_grounding_bert_inputs, with_kwargs=True)
        bert_model._darkfusion_sentence_mask_hook = True
    # Both models expose embeddings/encoder/pooler directly. Do not wrap it in
    # an extra registered child: that would change every checkpoint key.
    return bert_model


def load_groundingdino_model(config_path, checkpoint_path, device="cuda"):
    """Load GroundingDINO safely with the original checkpoint/state layout.

    Model construction mirrors groundingdino.util.inference.load_model (Apache
    2.0), but uses an explicit weights-only load and a scoped BERT adapter.
    No installed package files or global torch/Transformers methods are changed.
    """
    import importlib
    from darkfusion_model_security import install_checkpoint_guard
    install_checkpoint_guard()
    import torch
    from groundingdino.models import build_model
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig

    module = importlib.import_module("groundingdino.models.GroundingDINO.groundingdino")
    config = SLConfig.fromfile(str(config_path))
    config.device = device
    with _GROUNDING_BUILD_LOCK:
        original = module.BertModelWarper
        module.BertModelWarper = lambda bert_model: grounding_bert_wrapper(bert_model, original)
        try:
            model = build_model(config)
        finally:
            module.BertModelWarper = original
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    return model.eval()
