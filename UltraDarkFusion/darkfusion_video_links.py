"""Video link resolution, cached fallbacks and download workers.

Methods retain their existing owner argument and the application's state;
this module does not import the application entry point.
"""

import cv2
import glob
import hashlib
import html
import logging
import os
import random
import re
import tempfile
import time
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QHeaderView, QMessageBox, QProgressBar, QTableWidgetItem,
)
from darkfusion_media_tools import yt_dlp_options
from urllib.parse import urlparse
from yt_dlp import YoutubeDL


APP_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("UltraDarkFusionLogger")


SUPPORTED_VIDEO_URL_EXTS = (
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".wmv", ".m4v",
    ".mpg", ".mpeg", ".3gp", ".ts", ".ogv", ".m3u8",
)


def yt_dlp_runtime_options():
    """Use the same private media tools for playback, caching and downloads."""
    return yt_dlp_options(APP_DIR)


def video_stream_quality_key(value):
    """Website source quality, independent of player scaling and output FPS."""
    key = str(value or "1080").strip().lower()
    return key if key in {"720", "1080", "best"} else "1080"


def video_stream_format_selector(quality):
    quality = video_stream_quality_key(quality)
    limit = "" if quality == "best" else f"[height<={quality}]"
    return (
        f"bestvideo[vcodec^=avc1]{limit}/bestvideo{limit}/"
        "bestvideo[vcodec^=avc1]/bestvideo/best"
    )


class DownloadThread(QThread):
    update_status = pyqtSignal(int, str)
    update_progress = pyqtSignal(int, int)
    download_complete = pyqtSignal(int, str)

    def __init__(self, url, row, directory):
        super().__init__()
        self.url = url
        self.row = row
        self.directory = directory

    def clean_filename(self, filename):
        filename = re.sub(r'[<>:"/\\|?*]', " ", filename)
        filename = re.sub(r"\s+", " ", filename)
        filename = filename.strip()
        return filename or "unknown_video"

    def downloaded_file_path(self, ydl, info):
        requested_downloads = info.get("requested_downloads") or []
        for requested in requested_downloads:
            for key in ("filepath", "_filename", "filename"):
                candidate = requested.get(key)
                if candidate and os.path.exists(candidate):
                    return candidate

        candidates = []
        prepared = ydl.prepare_filename(info)
        if prepared:
            candidates.append(prepared)
            root, ext = os.path.splitext(prepared)
            for candidate_ext in (info.get("ext"), "mp4", "mkv", "webm"):
                if candidate_ext:
                    candidates.append(f"{root}.{candidate_ext.lstrip('.')}")

        existing = [path for path in candidates if path and os.path.exists(path)]
        if existing:
            return max(existing, key=os.path.getmtime)

        return ""

    def yt_dlp_progress_hook(self, d):
        status = d.get("status")

        if status == "downloading":
            total_bytes = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded_bytes = d.get("downloaded_bytes") or 0

            fragment_count = d.get("fragment_count") or 0
            fragment_index = d.get("fragment_index") or 0

            if total_bytes > 0:
                progress = int((downloaded_bytes / total_bytes) * 100)
            elif fragment_count > 0:
                progress = int((fragment_index / fragment_count) * 100)
            else:
                progress = 0

            progress = max(0, min(progress, 99))
            self.update_progress.emit(self.row, progress)

        elif status == "finished":
            self.update_progress.emit(self.row, 100)
            self.update_status.emit(self.row, "Processing...")

    def run(self):
        os.makedirs(self.directory, exist_ok=True)

        retries = 3
        output_template = os.path.join(self.directory, "%(title).200B.%(ext)s")

        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": output_template,
            "progress_hooks": [self.yt_dlp_progress_hook],
            "noplaylist": True,
            "windowsfilenames": True,
            "restrictfilenames": False,
            "overwrites": False,
            "continuedl": True,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": False,
        }
        try:
            ydl_opts.update(yt_dlp_runtime_options())
        except Exception as exc:
            logger.error("Video download media-tool setup failed: %s", exc)
            self.update_status.emit(self.row, f"Failed: {exc}")
            self.update_progress.emit(self.row, 0)
            return

        for attempt in range(retries):
            try:
                self.update_status.emit(self.row, f"Downloading... attempt {attempt + 1}/{retries}")

                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.url, download=True)
                    downloaded_path = self.downloaded_file_path(ydl, info)

                title = self.clean_filename(info.get("title", "unknown_video"))
                self.update_progress.emit(self.row, 100)
                self.update_status.emit(self.row, f"Complete: {title}")
                self.download_complete.emit(self.row, downloaded_path)
                logger.info(f"Downloaded video: {title}")
                return

            except Exception as e:
                if attempt < retries - 1:
                    wait_time = random.uniform(1, 3)
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Download failed with error: {e}")
                    self.update_status.emit(self.row, f"Failed: {str(e)}")
                    self.update_progress.emit(self.row, 0)


class VideoUrlResolverWorker(QThread):
    """Worker thread to resolve video URLs without blocking the UI."""
    resolved = pyqtSignal(str, str)  # Emits: (capture_source, original_source)
    error = pyqtSignal(str)  # Emits: error_message

    def __init__(self, main_window, source, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.source = source

    def run(self):
        try:
            capture_source, original_source = self.main_window._prepare_video_capture_source(self.source)
            if capture_source:
                self.resolved.emit(capture_source, original_source)
            else:
                self.error.emit(self.main_window._video_source_error_message(self.source))
        except Exception as e:
            logger.error(f"URL resolution error: {e}")
            self.error.emit(str(e))


class FrameExtractorUrlResolverWorker(QThread):
    """Worker thread to resolve video URLs for frame extraction without blocking the UI."""
    resolved = pyqtSignal(str, str)  # Emits: (capture_source, original_source)
    error = pyqtSignal(str)  # Emits: error_message

    def __init__(self, main_window, source, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.source = source

    def run(self):
        try:
            capture_source, original_source = self.main_window._prepare_video_capture_source(self.source)
            if capture_source:
                self.resolved.emit(capture_source, original_source)
            else:
                self.error.emit(self.main_window._video_source_error_message(self.source))
        except Exception as e:
            logger.error(f"Frame extraction URL resolution error: {e}")
            self.error.emit(str(e))


def is_video_url(self, source):
    source = str(source or "").strip()
    if not source:
        return False

    parsed = urlparse(source)
    return parsed.scheme.lower() in {"http", "https", "rtsp", "rtmp", "mms"}


def _is_direct_video_url(self, source):
    source = str(source or "").strip()
    if not source or not self.is_video_url(source):
        return False

    parsed = urlparse(source)
    if parsed.scheme.lower() in {"rtsp", "rtmp", "mms"}:
        return True

    path = parsed.path.lower()
    return path.endswith(SUPPORTED_VIDEO_URL_EXTS)


def _clean_video_source_name(self, value, fallback="stream"):
    value = html.unescape(str(value or "")).strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._ ")
    return (value[:120] or fallback)


def _safe_video_name_from_source(self, source, fallback="stream"):
    source = str(source or "").strip()
    cache = getattr(self, "video_source_cache", {}).get(source, {})

    if cache.get("name"):
        return self._clean_video_source_name(cache["name"], fallback)

    if self.is_video_url(source):
        parsed = urlparse(source)
        url_name = os.path.basename(parsed.path.rstrip("/"))
        url_name = os.path.splitext(url_name)[0] if url_name else ""
        return self._clean_video_source_name(url_name or parsed.netloc, fallback)

    path_name = os.path.splitext(os.path.basename(source))[0]
    return self._clean_video_source_name(path_name, fallback)


def _pick_yt_dlp_stream_url(self, info):
    if not info:
        return None

    formats = info.get("formats") or []
    quality = video_stream_quality_key(getattr(self, "settings", {}).get("videoStreamQuality"))

    def height_of(fmt):
        try:
            return max(0, int(fmt.get("height") or 0))
        except (TypeError, ValueError):
            return 0

    def format_score(fmt):
        if not fmt.get("url"):
            return None

        vcodec = (fmt.get("vcodec") or "").lower()
        if not vcodec or vcodec == "none":
            return None

        height = fmt.get("height") or 0
        try:
            height = int(height)
        except Exception:
            height = 0

        fps = fmt.get("fps") or 0
        try:
            fps = int(round(float(fps)))
        except Exception:
            fps = 0

        protocol = (fmt.get("protocol") or "").lower()
        ext = (fmt.get("ext") or "").lower()
        acodec = (fmt.get("acodec") or "").lower()
        is_h264 = vcodec.startswith(("avc1", "h264"))
        is_hls = "m3u8" in protocol
        has_audio = bool(acodec and acodec != "none")

        # Resolve source detail independently of display scaling. Among
        # compatible streams, prefer the requested resolution before HLS
        # versus progressive delivery; a 360p HLS stream must not silently
        # beat a compatible 1080p progressive stream.
        return (
            1 if is_h264 else 0,
            height,
            1 if is_hls else 0,
            1 if ext == "mp4" else 0,
            fps,
            1 if has_audio else 0,
        )

    playable_formats = [fmt for fmt in formats if format_score(fmt) is not None]
    if playable_formats:
        if quality != "best":
            limit = int(quality)
            within_limit = [fmt for fmt in playable_formats if 0 < height_of(fmt) <= limit]
            if within_limit:
                playable_formats = within_limit
            else:
                # Some sources only offer a higher resolution. Prefer the
                # smallest known fallback instead of unexpectedly taking 4K.
                heights = [height_of(fmt) for fmt in playable_formats if height_of(fmt)]
                if heights:
                    smallest = min(heights)
                    playable_formats = [fmt for fmt in playable_formats if height_of(fmt) == smallest]
        selected = max(playable_formats, key=format_score)
        logger.info(
            "Selected yt-dlp stream format %s (%sp, %s, %s).",
            selected.get("format_id", "unknown"),
            selected.get("height") or "unknown",
            selected.get("vcodec") or "unknown codec",
            selected.get("protocol") or "unknown protocol",
        )
        return selected.get("url")

    direct_url = info.get("url")
    if direct_url:
        return direct_url

    return None


def _capture_source_is_readable(self, source):
    """Probe one frame so an unusable signed URL never reaches playback."""
    capture = None
    try:
        capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        return bool(ok and frame is not None and getattr(frame, "size", 0))
    except Exception as error:
        logger.warning("Video URL probe failed: %s", error)
        return False
    finally:
        if capture is not None:
            capture.release()


def _download_video_url_cache(self, source):
    """Download a compatible fallback when direct OpenCV streaming fails."""
    cache_dir = os.path.join(tempfile.gettempdir(), "DarkFusion", "video_url_cache")
    os.makedirs(cache_dir, exist_ok=True)
    quality = video_stream_quality_key(getattr(self, "settings", {}).get("videoStreamQuality"))
    cache_key = hashlib.sha256(f"{source}\n{quality}".encode("utf-8", errors="ignore")).hexdigest()[:20]

    for candidate in glob.glob(os.path.join(cache_dir, f"{cache_key}.*")):
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            return candidate

    output_template = os.path.join(cache_dir, f"{cache_key}.%(ext)s")
    ydl_opts = {
        "format": video_stream_format_selector(quality),
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "overwrites": False,
        "socket_timeout": 30,  # Prevent hanging on slow/dead URLs
    }
    ydl_opts.update(yt_dlp_runtime_options())

    logger.info("Direct stream was unreadable; downloading a compatible cached copy.")
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=True)
        requested = info.get("requested_downloads") or []
        candidates = []
        for item in requested:
            candidates.extend(item.get(key) for key in ("filepath", "_filename", "filename"))
        candidates.append(ydl.prepare_filename(info))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            # Track temp file for cleanup
            if not hasattr(self, "_downloaded_temp_files"):
                self._downloaded_temp_files = set()
            self._downloaded_temp_files.add(candidate)
            return candidate

    matches = glob.glob(os.path.join(cache_dir, f"{cache_key}.*"))
    if matches:
        result = max(matches, key=os.path.getmtime)
        if not hasattr(self, "_downloaded_temp_files"):
            self._downloaded_temp_files = set()
        self._downloaded_temp_files.add(result)
        return result
    raise RuntimeError("yt-dlp completed but no cached video file was created.")


def _resolve_stream_url(self, source):
    source = str(source or "").strip()
    if not self.is_video_url(source):
        return source

    if not hasattr(self, "video_source_cache"):
        self.video_source_cache = {}

    quality = video_stream_quality_key(getattr(self, "settings", {}).get("videoStreamQuality"))
    cache = self.video_source_cache.get(source)
    if (cache and cache.get("stream_quality") == quality
            and time.time() - cache.get("timestamp", 0) < 20 * 60):
        return cache.get("capture_url", source)

    parsed = urlparse(source)
    if parsed.scheme.lower() in {"rtsp", "rtmp", "mms"}:
        self.video_source_cache[source] = {
            "capture_url": source,
            "stream_quality": quality,
            "name": self._safe_video_name_from_source(source),
            "timestamp": time.time(),
        }
        return source

    ydl_opts = {
        "format": video_stream_format_selector(quality),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "skip_download": True,
        "socket_timeout": 20,
    }
    ydl_opts.update(yt_dlp_runtime_options())

    try:
        QApplication.setOverrideCursor(Qt.WaitCursor)

        logger.info("Resolving video URL with yt-dlp.")

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=False)

        if info and info.get("_type") == "playlist":
            entries = [entry for entry in (info.get("entries") or []) if entry]
            info = entries[0] if entries else info

        capture_url = self._pick_yt_dlp_stream_url(info)
        if not capture_url:
            raise RuntimeError("yt-dlp did not return a playable stream URL.")

        if not self._capture_source_is_readable(capture_url):
            capture_url = self._download_video_url_cache(source)

        title = info.get("title") if info else None
        self.video_source_cache[source] = {
            "capture_url": capture_url,
            "stream_quality": quality,
            "is_live": bool(info and (info.get('is_live') or info.get('live_status') == 'is_live')
                            and self.is_video_url(capture_url)),
            "name": self._clean_video_source_name(title or self._safe_video_name_from_source(source)),
            "timestamp": time.time(),
        }
        self.last_video_source_error = ""

        logger.info("Resolved video URL with yt-dlp.")
        return capture_url

    except Exception as e:
        self.last_video_source_error = str(e)
        logger.warning(f"yt-dlp stream resolve failed: {e}")

        if self._is_direct_video_url(source):
            logger.warning("yt-dlp could not resolve the URL; trying direct video URL with OpenCV.")
            self.video_source_cache[source] = {
                "capture_url": source,
                "stream_quality": quality,
                "name": self._safe_video_name_from_source(source),
                "timestamp": time.time(),
            }
            return source

        logger.error(f"Could not resolve playable video URL: {source}")
        return ""

    finally:
        QApplication.restoreOverrideCursor()


def _video_source_error_message(self, source):
    detail = getattr(self, "last_video_source_error", "")

    if "javascript runtime" in detail.lower():
        return (
            "YouTube requires a supported JavaScript runtime.\n\n"
            "Install Deno and restart DarkFusion, then try the URL again."
        )

    if "not a bot" in detail.lower() or "sign in to confirm" in detail.lower():
        return (
            "YouTube blocked the anonymous stream request.\n\n"
            "Try updating yt-dlp, or download the video first and play the downloaded file from the queue."
        )

    if detail:
        return f"Could not resolve a playable stream for:\n{source}\n\n{detail}"

    return f"Could not resolve a playable stream for:\n{source}"


def _prepare_video_capture_source(self, source):
    source = str(source or "").strip()
    if not source:
        return "", ""

    if self.is_video_url(source):
        capture_source = self._resolve_stream_url(source)
        self.video_name = self._safe_video_name_from_source(source)
        self.current_source = source
        return capture_source, source

    self.video_name = self._safe_video_name_from_source(source, "video")
    self.current_source = source
    return source, source


def setup_video_queue_table(self):
    table = getattr(self, "video_table", None)
    if table is None:
        return

    table.setSortingEnabled(False)
    table.setColumnCount(3)
    table.setHorizontalHeaderLabels(["Video File or URL", "Status", "Progress"])
    table.setRowCount(max(table.rowCount(), 5))
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.SingleSelection)
    table.setEditTriggers(
        QAbstractItemView.DoubleClicked
        | QAbstractItemView.EditKeyPressed
        | QAbstractItemView.AnyKeyPressed
    )
    table.setTextElideMode(Qt.ElideMiddle)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(30)
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(0, QHeaderView.Stretch)
    header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(2, QHeaderView.Fixed)
    table.setColumnWidth(2, 140)

    for row in range(table.rowCount()):
        if table.item(row, 0) is None:
            table.setItem(row, 0, QTableWidgetItem(""))
        self._set_video_queue_status(row, "Ready")
        self._set_video_queue_progress(row, 0)

    self.video_queue_paste_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.Paste, table)
    self.video_queue_paste_shortcut.setContext(Qt.WidgetShortcut)
    self.video_queue_paste_shortcut.activated.connect(self.paste_video_queue_sources)


def _set_video_queue_status(self, row, status):
    table = getattr(self, "video_table", None)
    if table is None or not (0 <= row < table.rowCount()):
        return

    status_text = str(status)
    display_text = status_text

    if status_text.lower().startswith("failed"):
        display_text = "Failed"
    elif len(status_text) > 42:
        display_text = f"{status_text[:39]}..."

    item = QTableWidgetItem(display_text)
    item.setToolTip(status_text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    table.setItem(row, 1, item)


def _set_video_queue_progress(self, row, value):
    table = getattr(self, "video_table", None)
    if table is None or not (0 <= row < table.rowCount()):
        return

    progress_bar = table.cellWidget(row, 2)
    if not isinstance(progress_bar, QProgressBar):
        progress_bar = QProgressBar(table)
        progress_bar.setRange(0, 100)
        progress_bar.setTextVisible(True)
        table.setCellWidget(row, 2, progress_bar)

    progress_bar.setValue(max(0, min(100, int(value))))


def paste_video_queue_sources(self):
    table = getattr(self, "video_table", None)
    if table is None:
        return

    clipboard_text = QApplication.clipboard().text().strip()
    if not clipboard_text:
        return

    sources = [
        value.strip()
        for value in re.split(r"[\r\n\t]+", clipboard_text)
        if value.strip()
    ]

    for source in sources:
        self.add_video_to_table(source)


def _video_queue_item_text(self, row):
    table = getattr(self, "video_table", None)
    if table is None or not (0 <= row < table.rowCount()):
        return ""

    item = table.item(row, 0)
    return item.text().strip() if item is not None else ""


def _first_empty_video_queue_row(self, preferred_row=None):
    table = getattr(self, "video_table", None)
    if table is None:
        return -1

    candidate_rows = []
    if preferred_row is not None and 0 <= preferred_row < table.rowCount():
        candidate_rows.append(preferred_row)

    candidate_rows.extend(
        row for row in range(table.rowCount())
        if row not in candidate_rows
    )

    for row in candidate_rows:
        if not self._video_queue_item_text(row):
            return row

    table.insertRow(table.rowCount())
    new_row = table.rowCount() - 1
    table.setItem(new_row, 0, QTableWidgetItem(""))
    self._set_video_queue_status(new_row, "Ready")
    self._set_video_queue_progress(new_row, 0)
    return new_row


def setup_video_download_table(self):
    if self.video_download is None:
        self.progress_bars = []
        return

    for widget in (self.video_download, self.download_video, self.remove_video):
        if widget is not None:
            widget.setStyleSheet("")

    table = self.video_download
    table.setSortingEnabled(False)
    table.setColumnCount(3)
    table.setHorizontalHeaderLabels(["Video URL", "Status", "Progress"])
    table.setRowCount(5)
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.SingleSelection)
    table.setEditTriggers(
        QAbstractItemView.DoubleClicked
        | QAbstractItemView.EditKeyPressed
        | QAbstractItemView.AnyKeyPressed
    )
    table.setTextElideMode(Qt.ElideMiddle)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(30)
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(0, QHeaderView.Stretch)
    header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(2, QHeaderView.Fixed)
    table.setColumnWidth(2, 140)

    self.progress_bars = []
    for row in range(table.rowCount()):
        if table.item(row, 0) is None:
            table.setItem(row, 0, QTableWidgetItem(""))
        table.setItem(row, 1, QTableWidgetItem("Ready"))

        progress_bar = QProgressBar(table)
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        table.setCellWidget(row, 2, progress_bar)
        self.progress_bars.append(progress_bar)


def remove_selected_video(self):
    self.on_remove_video_clicked()


def download_videos(self):
    logger.debug("download_videos is called")

    directory = QtWidgets.QFileDialog.getExistingDirectory(
        self,
        'Select download directory',
        self.dialog_start_directory("video", getattr(self, "output_path", "")),
    )
    if not directory:
        return  # Ensure a directory is selected
    self.remember_dialog_selection("video", directory)

    started_download = False
    table = getattr(self, "video_table", None)
    if table is None:
        return

    for row in range(table.rowCount()):
        url = self._video_queue_item_text(row)
        if not url:
            logger.debug(f"No source found in video queue row {row}")
            continue

        if not self.is_video_url(url):
            logger.debug(f"Skipping local file row during download: {url}")
            continue

        logger.debug(f"Processing URL from queue row {row}: {url}")
        self.update_download_status(row, "Queued")
        self.update_download_progress(row, 0)

        download_thread = DownloadThread(url, row, directory)
        download_thread.update_status.connect(self.update_download_status)
        download_thread.update_progress.connect(self.update_download_progress)
        download_thread.download_complete.connect(self.on_video_queue_download_complete)
        download_thread.finished.connect(
            lambda thread=download_thread: self._cleanup_download_thread(thread)
        )
        self.download_threads.append(download_thread)
        download_thread.start()
        started_download = True

    if not started_download:
        QMessageBox.information(self, "Video Download", "Add at least one URL to the Video Queue before downloading.")


def update_download_progress(self, row, progress):
    self._set_video_queue_progress(row, progress)


def update_download_status(self, row, status):
    table = getattr(self, "video_table", None)
    if table is None or not (0 <= row < table.rowCount()):
        return

    self._set_video_queue_status(row, status)


def on_video_queue_download_complete(self, row, file_path):
    table = getattr(self, "video_table", None)
    if table is None or not (0 <= row < table.rowCount()):
        return

    if file_path and os.path.exists(file_path):
        table.setItem(row, 0, QTableWidgetItem(file_path))
        self._set_video_queue_status(row, "Ready")
        self._set_video_queue_progress(row, 100)
        table.selectRow(row)
        logger.info(f"Video queue row {row} now points to downloaded file: {file_path}")
    else:
        self._set_video_queue_status(row, "Downloaded")
        logger.warning("Download finished but the output file path was not detected.")


def _on_url_resolved(self, capture_source, original_source, generation=None):
    """Called when URL resolution completes successfully."""
    if not self.is_current_video_reader_generation(generation):
        return
    self._start_video_playback(capture_source, original_source)


def _on_url_resolution_error(self, error_message, generation=None):
    """Called when URL resolution fails."""
    if not self.is_current_video_reader_generation(generation):
        return
    QApplication.restoreOverrideCursor()
    self.play_video_button.setEnabled(True)
    logger.error(f"Could not resolve video source for playback: {error_message}")
    QMessageBox.warning(self, "Video Playback", f"Could not resolve video URL:\n\n{error_message}")


def _on_extraction_url_resolved(self, capture_source, original_source, generation=None):
    """Called when extraction URL resolution completes successfully."""
    if not self.is_current_extraction_generation(generation):
        if not self.extraction_is_active():
            self.set_extraction_ui_state("idle")
        return

    if not self.prepare_runtime_for_frame_extraction():
        QMessageBox.warning(
            self,
            "Extraction",
            "Live playback/inference is still shutting down. Try Extract again in a moment.",
        )
        self.set_extraction_ui_state("idle")
        return

    video_path = getattr(self, "_extraction_video_path", "")
    self.output_subfolder = self.get_video_output_dir(original_source)
    model_loaded = self.ensure_live_inference_model()
    config = self._snapshot_extraction_settings(
        "video",
        source=video_path,
        capture_source=capture_source,
        original_source=original_source,
        output_dir=self.output_subfolder,
        model_loaded=model_loaded,
    )
    if not self.start_extraction_thread(config) and not self.extraction_is_active():
        self.set_extraction_ui_state("idle")


def _on_extraction_url_resolution_error(self, error_message, generation=None):
    """Called when extraction URL resolution fails."""
    if not self.is_current_extraction_generation(generation):
        if not self.extraction_is_active():
            self.set_extraction_ui_state("idle")
        return

    QApplication.restoreOverrideCursor()
    self.set_extraction_ui_state("idle", "Extraction URL resolve failed.")
    logger.error(f"Could not resolve video source for extraction: {error_message}")
    QMessageBox.warning(self, "Frame Extraction", f"Could not resolve video URL:\n\n{error_message}")


def _on_direct_extraction_url_error(self, error_message, generation=None):
    """Called when direct video extraction URL resolution fails."""
    if not self.is_current_extraction_generation(generation):
        if not self.extraction_is_active():
            self.set_extraction_ui_state("idle")
        return

    logger.error(f"Could not resolve video source for direct extraction: {error_message}")
    QMessageBox.warning(self, "Frame Extraction", f"Could not resolve video URL:\n\n{error_message}")
    self.extracting_frames = False
    self.set_extraction_ui_state("idle", "Extraction URL resolve failed.")
