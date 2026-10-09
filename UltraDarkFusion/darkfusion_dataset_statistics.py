"""Dataset statistics built from the analysis worker's existing scan records.

No dataset files are opened here. Charts use compact report data; the scatter
plot uses a deterministic reservoir sample so large datasets stay responsive.
"""
from collections import Counter, defaultdict
import math
import random

from PyQt5 import QtCore, QtWidgets

SCATTER_LIMIT = 5000
HISTOGRAM_BINS = 50
QUALITY_DEFAULTS = {"blur": 100.0, "dark": 50.0, "bright": 200.0, "contrast": 10.0}


def annotation_metrics(parsed, image_size):
    """Measure the enclosing rectangle consistently for boxes, pose and polygons."""
    values = parsed.get("values", [])
    kind = parsed.get("annotation_type")
    try:
        width, height = image_size
        if kind in ("bbox", "bbox_keypoints"):
            x, y, w, h = map(float, values[:4])
        elif kind in ("segmentation", "obb") and len(values) >= 6:
            xs, ys = list(map(float, values[::2])), list(map(float, values[1::2]))
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        else:
            return None
        if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h, width, height) <= 0:
            return None
        return {"x": x, "y": y, "width": w * width, "height": h * height,
                "area": w * h * width * height, "relative_area": w * h,
                "type": kind, "class_id": int(parsed["class_id"])}
    except (ValueError, TypeError, KeyError):
        return None


def image_quality_metrics(image):
    """Measure already-decoded pixels; called only by the background worker."""
    import cv2
    import numpy as np
    gray = np.asarray(image.convert("L"))
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return {"blur": float((grad_x.var() + grad_y.var()) / 2),
            "brightness": float(gray.mean()), "contrast": float(gray.std())}


def quality_findings(metrics, thresholds):
    """Return image-level review hints, not definitive quality judgments."""
    found = []
    if metrics["blur"] < thresholds["blur"]:
        found.append(("blurry_image", f"Detail score {metrics['blur']:.1f} is below {thresholds['blur']:g}."))
    if metrics["brightness"] < thresholds["dark"]:
        found.append(("underexposed_image", f"Mean brightness {metrics['brightness']:.1f} is below {thresholds['dark']:g}."))
    if metrics["brightness"] > thresholds["bright"]:
        found.append(("overexposed_image", f"Mean brightness {metrics['brightness']:.1f} is above {thresholds['bright']:g}."))
    if metrics["contrast"] < thresholds["contrast"]:
        found.append(("low_contrast_image", f"Contrast {metrics['contrast']:.1f} is below {thresholds['contrast']:g}."))
    return found


def build_statistics(image_files, image_sizes, records, label_states, class_names,
                     quality_records, *, basic_enabled=True, quality_enabled=False,
                     quality_thresholds=None, should_cancel=lambda: False):
    """Aggregate a completed scan in the worker; return None on cancellation."""
    thresholds = dict(QUALITY_DEFAULTS, **(quality_thresholds or {}))
    class_names = dict(enumerate(class_names)) if not isinstance(class_names, dict) else class_names
    class_counts, class_images = Counter(), defaultdict(set)
    per_image, type_counts, sizes, positions = Counter(), Counter(), Counter(), Counter()
    histogram = [0] * HISTOGRAM_BINS
    scatter, rng = [], random.Random(0)
    smallest = None
    total = 0
    outside = 0
    for index, record in enumerate(records if basic_enabled else ()):
        if index % 256 == 0 and should_cancel():
            return None
        path = record.get("image_path")
        metrics = annotation_metrics(record.get("parsed", {}), image_sizes.get(path, (0, 0)))
        if metrics is None:
            continue
        total += 1
        cid, area = metrics["class_id"], metrics["relative_area"]
        class_counts[cid] += 1
        class_images[cid].add(path)
        per_image[path] += 1
        type_counts[metrics["type"]] += 1
        bucket = "tiny" if area < .01 else "small" if area < .10 else "medium" if area < .30 else "large"
        sizes[bucket] += 1
        x, y = metrics["x"], metrics["y"]
        positions["left" if x < .33 else "right" if x > .66 else "center"] += 1
        positions["top" if y < .33 else "bottom" if y > .66 else "middle"] += 1
        if 0 <= area <= 1:
            histogram[min(HISTOGRAM_BINS - 1, int(area * HISTOGRAM_BINS))] += 1
        else:
            outside += 1
        point = [round(x, 6), round(y, 6), round(area, 8), cid]
        if len(scatter) < SCATTER_LIMIT:
            scatter.append(point)
        else:
            slot = rng.randrange(total)
            if slot < SCATTER_LIMIT:
                scatter[slot] = point
        if smallest is None or metrics["area"] < smallest["area"]:
            smallest = dict(metrics, image=path, line=record.get("line_number"))

    states, resolutions, objects_per_image = Counter(), Counter(), Counter()
    invalid_rows = 0
    for index, path in enumerate(image_files):
        if index % 256 == 0 and should_cancel():
            return None
        if path in image_sizes:
            resolutions["{} × {}".format(*image_sizes[path])] += 1
        if not basic_enabled:
            continue
        info = label_states.get(path, {"state": "missing", "line_count": 0})
        state = info["state"]
        if state == "annotated":
            if path not in image_sizes:
                state = "unavailable"
            else:
                invalid = max(0, info["line_count"] - per_image[path])
                invalid_rows += invalid
                state = ("partial" if invalid else "annotated") if per_image[path] else "invalid"
        states[state] += 1
        # Zero-object distribution includes confirmed blank labels only.
        if state in ("empty", "annotated", "partial"):
            n = per_image[path]
            objects_per_image["0" if n == 0 else "1" if n == 1 else "2–5" if n <= 5 else "6–10" if n <= 10 else ">10"] += 1

    quality_counts = Counter()
    if quality_enabled:
        for index, metrics in enumerate(quality_records.values()):
            if index % 256 == 0 and should_cancel():
                return None
            quality_counts.update(kind for kind, _message in quality_findings(metrics, thresholds))
    classes = [{"id": int(cid), "name": str(name), "labels": class_counts[cid],
                "images": len(class_images[cid])} for cid, name in sorted(class_names.items())]
    return {
        "version": 1, "basic_enabled": bool(basic_enabled),
        "total_images": len(image_files), "readable_images": len(image_sizes),
        "unreadable_images": len(image_files) - len(image_sizes),
        "total_annotations": total, "annotated_images": sum(n > 0 for n in per_image.values()),
        "label_states": dict(states), "excluded_annotation_rows": invalid_rows,
        "classes": classes, "annotation_types": dict(type_counts),
        "object_sizes": dict(sizes), "position_counts": dict(positions),
        "objects_per_image": dict(objects_per_image),
        "image_resolutions": dict(resolutions), "smallest_object": smallest,
        "area_histogram": histogram, "area_histogram_outside_range": outside,
        "scatter_sample": scatter, "scatter_limit": SCATTER_LIMIT,
        "quality": {"enabled": bool(quality_enabled), "thresholds": thresholds,
                    "checked_images": len(quality_records) if quality_enabled else 0,
                    "skipped_images": len(image_files) - len(quality_records) if quality_enabled else 0,
                    "counts": dict(quality_counts)},
    }


class DatasetStatisticsWidget(QtWidgets.QWidget):
    """A read-only view of one report. Changing views never starts a scan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("datasetAnalysisStatistics")
        self.data = None
        self._chart_dirty = True
        self._canvas = None
        layout = QtWidgets.QVBoxLayout(self)
        self.notice = QtWidgets.QLabel()
        self.notice.setWordWrap(True)
        self.notice.setObjectName("datasetStatisticsStatus")
        layout.addWidget(self.notice)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("datasetStatisticsViews")
        layout.addWidget(self.tabs, 1)
        self.overview = self._table(["Statistic", "Value"])
        self.classes = self._table(["Class ID", "Class", "Annotations", "Images", "Share of annotations"])
        self.classes.setObjectName("datasetStatisticsClasses")
        self.quality = self._table(["Quality check", "Result"])
        self.tabs.addTab(self.overview, "Overview")
        self.tabs.addTab(self.classes, "Classes")
        self.chart_page = QtWidgets.QWidget()
        self.chart_layout = QtWidgets.QVBoxLayout(self.chart_page)
        self.chart_choice = QtWidgets.QComboBox()
        self.chart_choice.setObjectName("datasetStatisticsChart")
        for text, key in (("Class distribution", "classes"), ("Object area histogram", "histogram"),
                          ("Object centers", "scatter"), ("Objects per image", "objects")):
            self.chart_choice.addItem(text, key)
        self.chart_layout.addWidget(self.chart_choice)
        self.chart_note = QtWidgets.QLabel()
        self.chart_note.setWordWrap(True)
        self.chart_layout.addWidget(self.chart_note)
        self.tabs.addTab(self.chart_page, "Charts")
        self.tabs.addTab(self.quality, "Image Quality")
        self.tabs.currentChanged.connect(self._draw_chart)
        self.chart_choice.currentIndexChanged.connect(self._invalidate_chart)
        self.set_report({})

    @staticmethod
    def _table(headers):
        table = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        return table

    @staticmethod
    def _rows(table, rows):
        table.setUpdatesEnabled(False)
        try:
            table.setRowCount(len(rows))
            for row, values in enumerate(rows):
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem()
                    item.setData(QtCore.Qt.DisplayRole, value)
                    table.setItem(row, column, item)
        finally:
            table.setUpdatesEnabled(True)

    def set_report(self, report):
        self.data = report.get("statistics")
        self._chart_dirty = True
        if not self.data:
            self.notice.setText(
                "Statistics have not been collected. Select Collect statistics in Scan Settings, then press Start Scan."
            )
            for table in (self.overview, self.classes, self.quality):
                table.setRowCount(0)
            self.tabs.setEnabled(False)
            if self._canvas:
                self._canvas.figure.clear()
                self._canvas.draw_idle()
            return
        self.tabs.setEnabled(True)
        data = self.data
        basic = data.get("basic_enabled", False)
        self.notice.setText(
            f"Snapshot from {report.get('generated_at', 'the last completed scan')}. "
            "Tabs and charts reuse this scan. After editing labels, press Start Scan to refresh."
        )
        self.tabs.setTabEnabled(0, basic)
        self.tabs.setTabEnabled(1, basic)
        self.tabs.setTabEnabled(2, basic)
        if not basic:
            self.tabs.setCurrentIndex(3)
        states = data.get("label_states", {})
        annotated = data.get("annotated_images", 0)
        total = data.get("total_annotations", 0)
        rows = [
            ("Total images", data["total_images"]),
            ("Readable image headers", data["readable_images"]),
            ("Unreadable images", data["unreadable_images"]),
            ("Images with measurable annotations", annotated),
            ("Blank label files (possible intentional negatives)", states.get("empty", 0)),
            ("Missing label files", states.get("missing", 0)),
            ("Invalid label files (no measurable annotations)", states.get("invalid", 0)),
            ("Partially valid label files", states.get("partial", 0)),
            ("Unreadable label files", states.get("unreadable", 0)),
            ("Labels unavailable due to unreadable images", states.get("unavailable", 0)),
            ("Measurable annotations", total),
            ("Excluded annotation rows", data.get("excluded_annotation_rows", 0)),
            ("Annotations per annotated image", round(total / annotated, 2) if annotated else 0),
        ]
        for title, key in (("Annotation types", "annotation_types"), ("Object size groups", "object_sizes"),
                           ("Objects per image (known labels)", "objects_per_image"),
                           ("Object center positions", "position_counts")):
            rows.append((title, ", ".join(f"{k}: {v}" for k, v in data.get(key, {}).items()) or "None"))
        rows.append(("Size groups (enclosing rectangle / image area)", "tiny <1%, small <10%, medium <30%, large ≥30%"))
        resolutions = sorted(data.get("image_resolutions", {}).items(), key=lambda item: (-item[1], item[0]))
        rows.extend((f"Image resolution: {size}", count) for size, count in resolutions[:100])
        if len(resolutions) > 100:
            rows.append(("Other image resolutions", f"{len(resolutions) - 100} more in the saved report"))
        smallest = data.get("smallest_object")
        if smallest:
            rows.append(("Smallest enclosing rectangle", f"{smallest['width']:.2f} × {smallest['height']:.2f} px"))
        rows.append(("Training image size", "Use Trainer → Generate Files + Parameters for a dataset-based recommendation."))
        self._rows(self.overview, rows if basic else [])
        self._rows(self.classes, [
            [row["id"], row["name"], row["labels"], row["images"],
             f"{100 * row['labels'] / total:.1f}%" if total else "0.0%"]
            for row in data.get("classes", []) if basic
        ])
        quality = data.get("quality", {})
        if quality.get("enabled"):
            thresholds = quality["thresholds"]
            counts = quality["counts"]
            quality_rows = [
                ("Images checked", quality["checked_images"]),
                ("Images unavailable for quality checks", quality["skipped_images"]),
                (f"Low detail / possible blur (< {thresholds['blur']:g})", counts.get("blurry_image", 0)),
                (f"Low brightness (< {thresholds['dark']:g})", counts.get("underexposed_image", 0)),
                (f"High brightness (> {thresholds['bright']:g})", counts.get("overexposed_image", 0)),
                (f"Low contrast (< {thresholds['contrast']:g})", counts.get("low_contrast_image", 0)),
                ("Interpretation", "Review hints; image content and resolution affect these scores. See Findings."),
            ]
        else:
            quality_rows = [("Image quality", "Not scanned. Enable Image quality in Scan Settings.")]
        self._rows(self.quality, quality_rows)
        self._draw_chart()

    def showEvent(self, event):
        super().showEvent(event)
        self._draw_chart()

    def _invalidate_chart(self, *_args):
        self._chart_dirty = True
        self._draw_chart()

    def select_chart(self, key):
        index = self.chart_choice.findData(key)
        self.chart_choice.setCurrentIndex(max(0, index))
        self.tabs.setCurrentWidget(self.chart_page)
        self._draw_chart()

    def _draw_chart(self, *_args):
        if not self.isVisible() or self.tabs.currentWidget() is not self.chart_page:
            return
        if not self._chart_dirty or not self.data or not self.data.get("basic_enabled"):
            return
        if self._canvas is None:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure
            self._canvas = FigureCanvasQTAgg(Figure(figsize=(7, 3), tight_layout=True))
            self.chart_layout.addWidget(NavigationToolbar2QT(self._canvas, self), 0)
            self.chart_layout.addWidget(self._canvas, 1)
        figure = self._canvas.figure
        figure.clear()
        figure.set_facecolor("#15191e")
        ax = figure.add_subplot(111)
        ax.set_facecolor("#15191e")
        ax.tick_params(colors="#dce6f0", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#6b7785")
        ax.xaxis.label.set_color("#dce6f0")
        ax.yaxis.label.set_color("#dce6f0")
        data, key = self.data, self.chart_choice.currentData()
        note = "Based on the completed scan; changing charts does not rescan files."
        if key == "classes":
            rows = sorted(data.get("classes", []), key=lambda row: (-row["labels"], row["id"]))[:40]
            ax.bar(range(len(rows)), [row["labels"] for row in rows], color="#70b9ef")
            ax.set_xticks(range(len(rows)), [f"{row['id']}: {row['name']}" for row in rows], rotation=35, ha="right")
            ax.set_ylabel("Annotations")
            if len(data.get("classes", [])) > 40:
                note += " Showing the 40 largest classes; the Classes table includes every class."
        elif key == "histogram":
            counts = data.get("area_histogram", [])
            ax.bar([(i + .5) * 100 / HISTOGRAM_BINS for i in range(len(counts))],
                   counts, width=100 / HISTOGRAM_BINS, color="#70b9ef")
            ax.set_xlabel("Enclosing rectangle area (% of image)")
            ax.set_ylabel("Annotations")
            if data.get("area_histogram_outside_range"):
                note += f" Excludes {data['area_histogram_outside_range']} rectangles larger than their image."
        elif key == "scatter":
            points = data.get("scatter_sample", [])
            if points:
                ax.scatter([p[0] for p in points], [p[1] for p in points],
                           c=[p[3] for p in points], s=7, alpha=.55, cmap="tab20", rasterized=True)
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_xlabel("Horizontal center (0–1)")
            ax.set_ylabel("Vertical center (0–1)")
            note += f" Showing {len(points):,} of {data['total_annotations']:,} centers; colors identify class IDs."
            if len(points) < data["total_annotations"]:
                note += " A representative sample keeps rendering fast."
        else:
            buckets = ["0", "1", "2–5", "6–10", ">10"]
            ax.bar(buckets, [data.get("objects_per_image", {}).get(key, 0) for key in buckets], color="#70b9ef")
            ax.set_xlabel("Annotations per image")
            ax.set_ylabel("Images")
            note += " Missing, invalid and unreadable labels are excluded."
        self.chart_note.setText(note)
        self._chart_dirty = False
        self._canvas.draw_idle()

