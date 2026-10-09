"""Stable annotation identities for Dataset Analysis review feedback."""

import math


def canonical_label_text(line):
    """Ignore numeric serialization noise, preserving class and full geometry."""
    text = " ".join(str(line or "").split())
    try:
        values = [float(token) for token in text.split()]
        if not values or not all(math.isfinite(value) for value in values):
            return text
        return " ".join(format(round(value, 8) or 0.0, ".8f") for value in values)
    except ValueError:
        return text


def find_annotation_row(issue, lines):
    """Find the original annotation; a nearby replacement is never a match."""
    expected = canonical_label_text(issue.get("annotation_label_text"))
    reference = issue.get("ground_truth") or {}
    preferred = issue.get("annotation_line_index", reference.get("label_line"))
    matches = []
    for index, line in enumerate(lines):
        if expected:
            same = canonical_label_text(line) == expected
        else:
            # Legacy reports lack raw text. Require the complete saved geometry,
            # never just its line number or overlap with a newly drawn box.
            same = _matches_legacy_geometry(issue, reference, line)
        if same:
            matches.append(index)
    if preferred in matches:
        return preferred
    return matches[0] if len(matches) == 1 else -1


def _matches_legacy_geometry(issue, reference, line):
    try:
        values = [float(token) for token in str(line).split()]
        if not values or int(values[0]) != reference.get("class_id", issue.get("class_id")):
            return False
        task = issue.get("task", "detect")
        if task in {"segment", "obb"}:
            expected = [value for point in reference.get("points", []) for value in point]
            actual = values[1:]
        else:
            cx, cy, width, height = values[1:5]
            actual = [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]
            expected = list(reference.get("bbox", []))
            if task == "pose":
                expected += [value for point in reference.get("keypoints", []) for value in point]
                actual += values[5:]
        return bool(expected) and len(expected) == len(actual) and all(
            math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-8)
            for a, b in zip(expected, actual)
        )
    except (TypeError, ValueError, OverflowError):
        return False
