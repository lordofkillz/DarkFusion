"""Model choices and per-model similarity thresholds, without GUI/ML imports."""

DEFAULT_METHOD = "dinov3_base"
SETTINGS_VERSION = 1
METHOD_OPTIONS = (
    ("dinov3_base", "DINOv3 Base (recommended)"),
    ("dinov3_large", "DINOv3 Large"),
    ("visual", "DINOv2"),
    ("appearance", "Appearance and shape (CPU)"),
)
RECOMMENDED_THRESHOLDS = {"dinov3_base": 82, "dinov3_large": 88, "visual": 90, "appearance": 90}


def _threshold(value, default):
    try:
        return max(50, min(99, int(value)))
    except (ValueError, TypeError, OverflowError):
        return default


def review_method(settings):
    method = settings.get("reviewSimilarityMethod", DEFAULT_METHOD)
    return method if isinstance(method, str) and method in RECOMMENDED_THRESHOLDS else DEFAULT_METHOD


def review_threshold(settings, method=None):
    method = method if method in RECOMMENDED_THRESHOLDS else review_method(settings)
    saved = settings.get("reviewSimilarityThresholds")
    saved = saved if isinstance(saved, dict) else {}
    fallback = (settings.get("reviewSimilarityThreshold", RECOMMENDED_THRESHOLDS[method])
                if method == review_method(settings) else RECOMMENDED_THRESHOLDS[method])
    return _threshold(saved.get(method, fallback), RECOMMENDED_THRESHOLDS[method])


def migrate_review_settings(settings):
    """Upgrade the previous AI default once, preserving the old method's cutoff.

    Operates on the provided dictionary only; normal application saving persists
    it. Explicit CPU choices and subsequent model choices remain respected.
    """
    stored = settings.get("reviewSimilarityThresholds")
    thresholds = {key: _threshold(value, RECOMMENDED_THRESHOLDS[key])
                  for key, value in (stored.items() if isinstance(stored, dict) else [])
                  if key in RECOMMENDED_THRESHOLDS}
    old_method = review_method(settings)
    old_value = _threshold(settings.get("reviewSimilarityThreshold", RECOMMENDED_THRESHOLDS[old_method]),
                           RECOMMENDED_THRESHOLDS[old_method])
    thresholds.setdefault(old_method, old_value)
    if settings.get("reviewSimilaritySettingsVersion") != SETTINGS_VERSION and old_method == "visual":
        method = DEFAULT_METHOD
        thresholds.setdefault(method, RECOMMENDED_THRESHOLDS[method])
    else:
        method = old_method
    settings["reviewSimilarityMethod"] = method
    settings["reviewSimilarityThresholds"] = thresholds
    settings["reviewSimilarityThreshold"] = thresholds[method]
    settings["reviewSimilaritySettingsVersion"] = SETTINGS_VERSION
    return settings


def select_review_method(settings, method):
    if method not in RECOMMENDED_THRESHOLDS:
        raise ValueError(f"Unknown review matching method: {method}")
    current = review_method(settings)
    stored = settings.get("reviewSimilarityThresholds")
    thresholds = dict(stored) if isinstance(stored, dict) else {}
    thresholds[current] = review_threshold(settings)
    settings["reviewSimilarityMethod"] = method
    settings["reviewSimilarityThreshold"] = _threshold(thresholds.get(method), RECOMMENDED_THRESHOLDS[method])
    thresholds[method] = settings["reviewSimilarityThreshold"]
    settings["reviewSimilarityThresholds"] = thresholds
    settings["reviewSimilaritySettingsVersion"] = SETTINGS_VERSION
    return settings["reviewSimilarityThreshold"]


def set_review_threshold(settings, value):
    method = review_method(settings)
    value = _threshold(value, RECOMMENDED_THRESHOLDS[method])
    stored = settings.get("reviewSimilarityThresholds")
    thresholds = dict(stored) if isinstance(stored, dict) else {}
    thresholds[method] = value
    settings["reviewSimilarityThresholds"] = thresholds
    settings["reviewSimilarityThreshold"] = value
    return value
