"""Dataset split and revert dialog using DarkFusion's shared filesystem helpers."""
import logging
import math
import os
from pathlib import Path
import shutil

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFileDialog, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QVBoxLayout,
)

from darkfusion_dataset_operations import (
    PROJECT_SETTINGS_DIR, _files_have_same_content, dataset_metadata_directory,
    dataset_metadata_path, migrate_darkfusion_dataset_metadata, resolve_dataset_classes_path,
)
from darkfusion_image_loading import OUTPUT_IMAGE_SUFFIXES

logger = logging.getLogger("UltraDarkFusionLogger")


class DatasetSplitDialog(QDialog):
    """Native DarkFusion dataset splitter that replaces the old Tk window."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window

        self.setWindowTitle("Split Dataset")
        self.setMinimumSize(780, 560)
        self.resize(860, 620)
        parent_style = main_window.styleSheet() if main_window is not None else ""
        if parent_style:
            self.setStyleSheet(parent_style)

        self._build_ui()
        self.dataset_edit.setText(self._initial_dataset_dir())
        self.chunk_spinbox.setValue(int(getattr(main_window, "settings", {}).get("splitChunkSize", 10000)))
        self.prefix_edit.setText(str(getattr(main_window, "settings", {}).get("splitFolderPrefix", "images_")))
        self.refresh_summary()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header_row = QtWidgets.QHBoxLayout()
        title_label = QLabel("Split Dataset")
        title_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        header_row.addWidget(title_label)
        header_row.addStretch()
        layout.addLayout(header_row)

        dataset_group = QtWidgets.QGroupBox("Dataset")
        dataset_layout = QtWidgets.QGridLayout(dataset_group)
        dataset_layout.setHorizontalSpacing(8)
        dataset_layout.setVerticalSpacing(8)

        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText("Select a dataset folder")
        self.dataset_edit.textChanged.connect(self.refresh_summary)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_dataset)

        dataset_layout.addWidget(QLabel("Folder:"), 0, 0)
        dataset_layout.addWidget(self.dataset_edit, 0, 1)
        dataset_layout.addWidget(browse_btn, 0, 2)
        dataset_layout.setColumnStretch(1, 1)
        layout.addWidget(dataset_group)

        settings_group = QtWidgets.QGroupBox("Split Settings")
        settings_layout = QtWidgets.QFormLayout(settings_group)
        settings_layout.setLabelAlignment(Qt.AlignRight)
        settings_layout.setHorizontalSpacing(12)
        settings_layout.setVerticalSpacing(8)

        self.chunk_spinbox = QSpinBox()
        self.chunk_spinbox.setRange(1, 999999999)
        self.chunk_spinbox.setSingleStep(100)
        self.chunk_spinbox.setValue(10000)
        self.chunk_spinbox.setToolTip("Number of images per output folder.")
        self.chunk_spinbox.valueChanged.connect(self.refresh_summary)

        self.prefix_edit = QLineEdit("images_")
        self.prefix_edit.setToolTip("Split folders will be named with this prefix, like images_1.")
        self.prefix_edit.textChanged.connect(self.refresh_summary)

        settings_layout.addRow("Images per folder:", self.chunk_spinbox)
        settings_layout.addRow("Folder prefix:", self.prefix_edit)
        layout.addWidget(settings_group)

        summary_group = QtWidgets.QGroupBox("Summary")
        summary_layout = QtWidgets.QGridLayout(summary_group)
        summary_layout.setHorizontalSpacing(8)
        summary_layout.setVerticalSpacing(8)

        self.metric_labels = {}
        metrics = [
            ("root_images", "Root Images"),
            ("labels", "Matching Labels"),
            ("classes", "classes.txt"),
            ("existing_split", "Split Folders"),
            ("output_folders", "Folders Needed"),
        ]

        for index, (key, title) in enumerate(metrics):
            summary_layout.addWidget(self._metric_card(key, title), index // 3, index % 3)

        layout.addWidget(summary_group, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #9aa4af;")
        layout.addWidget(self.status_label)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_summary)

        split_btn = QPushButton("Split Dataset")
        split_btn.setObjectName("primarySplitButton")
        split_btn.clicked.connect(self.split_dataset)

        revert_btn = QPushButton("Revert Split")
        revert_btn.clicked.connect(self.revert_split)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)

        button_row.addWidget(refresh_btn)
        button_row.addWidget(split_btn)
        button_row.addWidget(revert_btn)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

    def _metric_card(self, key, title):
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        frame.setStyleSheet(
            "QFrame { border: 1px solid #30363d; border-radius: 6px; padding: 8px; }"
            "QLabel { border: none; }"
        )

        card_layout = QVBoxLayout(frame)
        card_layout.setContentsMargins(8, 6, 8, 6)

        value_label = QLabel("0")
        value_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #7cc7ff;")

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #9aa4af;")

        card_layout.addWidget(value_label)
        card_layout.addWidget(title_label)
        self.metric_labels[key] = value_label
        return frame

    def _initial_dataset_dir(self):
        candidates = [
            getattr(self.main_window, "image_directory", ""),
            getattr(self.main_window, "output_path", ""),
        ]

        settings = getattr(self.main_window, "settings", {})
        if isinstance(settings, dict):
            candidates.append(settings.get("last_dir", ""))

        candidates.append(os.getcwd())

        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return self.main_window.normalize_path(candidate)

        return ""

    def dataset_path(self):
        text = self.dataset_edit.text().strip()
        return Path(text).expanduser() if text else None

    def folder_prefix(self):
        prefix = self.prefix_edit.text().strip()
        return prefix or "images_"

    def _image_files(self, dataset_path):
        try:
            return sorted(
                path
                for path in dataset_path.iterdir()
                if path.is_file() and path.suffix.lower() in OUTPUT_IMAGE_SUFFIXES
            )
        except Exception:
            return []

    def _split_folders(self, dataset_path):
        prefix = self.folder_prefix()
        try:
            return sorted(
                path
                for path in dataset_path.iterdir()
                if path.is_dir() and path.name.startswith(prefix)
            )
        except Exception:
            return []

    def _unique_destination(self, destination):
        destination = Path(destination)
        if not destination.exists():
            return destination

        for counter in range(1, 100000):
            candidate = destination.with_name(f"{destination.stem}_{counter}{destination.suffix}")
            if not candidate.exists():
                return candidate

        raise FileExistsError(f"Could not create a unique name for {destination}")

    def _unique_image_destination(self, destination):
        destination = Path(destination)
        if not destination.exists() and not destination.with_suffix(".txt").exists():
            return destination

        for counter in range(1, 100000):
            candidate = destination.with_name(f"{destination.stem}_{counter}{destination.suffix}")
            if not candidate.exists() and not candidate.with_suffix(".txt").exists():
                return candidate

        raise FileExistsError(f"Could not create a unique image name for {destination}")

    def _set_metric(self, key, value, color=None):
        label = self.metric_labels.get(key)
        if label is None:
            return

        label.setText(str(value))
        if color:
            label.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {color};")

    def _set_status(self, text, color="#9aa4af"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")

    def browse_dataset(self):
        start_dir = self.main_window.dialog_start_directory(
            "dataset", self.dataset_edit.text().strip() or self._initial_dataset_dir()
        )
        selected = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", start_dir)
        if selected:
            self.main_window.remember_dialog_selection("dataset", selected)
            self.dataset_edit.setText(self.main_window.normalize_path(selected))

    def refresh_summary(self):
        dataset_path = self.dataset_path()
        if not dataset_path or not dataset_path.is_dir():
            self._set_metric("root_images", 0)
            self._set_metric("labels", 0)
            self._set_metric("classes", "Missing", "#f85149")
            self._set_metric("existing_split", 0)
            self._set_metric("output_folders", 0)
            self._set_status("Choose a dataset folder to see split details.")
            return

        image_files = self._image_files(dataset_path)
        label_count = sum(1 for image_file in image_files if image_file.with_suffix(".txt").exists())
        classes_path = resolve_dataset_classes_path(str(dataset_path), migrate=True)
        classes_file = Path(classes_path) if classes_path else Path()
        split_folders = self._split_folders(dataset_path)
        chunk_size = max(1, int(self.chunk_spinbox.value()))
        output_count = math.ceil(len(image_files) / chunk_size) if image_files else 0

        self._set_metric("root_images", len(image_files), "#7cc7ff")
        self._set_metric("labels", label_count, "#9aa4af")
        classes_found = bool(classes_path and classes_file.is_file())
        self._set_metric("classes", "Found" if classes_found else "Missing", "#7cc7ff" if classes_found else "#f85149")
        self._set_metric("existing_split", len(split_folders), "#f0b45f" if split_folders else "#9aa4af")
        self._set_metric("output_folders", output_count, "#7cc7ff")

        if not classes_found:
            self._set_status("classes.txt is required before splitting.", "#f0b45f")
        elif not image_files and split_folders:
            self._set_status("No root images found. This dataset may already be split.", "#9aa4af")
        elif not image_files:
            self._set_status("No root images found in this folder.", "#9aa4af")
        else:
            self._set_status(f"Ready to split {len(image_files)} images into {output_count} folder(s).", "#7cc7ff")

    def _save_split_settings(self):
        settings = getattr(self.main_window, "settings", None)
        if not isinstance(settings, dict):
            return

        settings["splitChunkSize"] = int(self.chunk_spinbox.value())
        settings["splitFolderPrefix"] = self.folder_prefix()
        if hasattr(self.main_window, "saveSettings"):
            self.main_window.saveSettings()

    def _refresh_parent_view_if_current_dataset(self, dataset_path):
        active_dir = self.main_window.normalize_path(getattr(self.main_window, "image_directory", ""))
        target_dir = self.main_window.normalize_path(str(dataset_path))
        if active_dir != target_dir:
            return

        image_files = [self.main_window.normalize_path(str(path)) for path in self._image_files(dataset_path)]
        remaining = set(image_files)
        removed = [path for path in getattr(self.main_window, "image_files", [])
                   if self.main_window.normalize_path(path) not in remaining]
        remove_from_review = getattr(self.main_window, "_remove_moved_images_from_review", None)
        if removed and callable(remove_from_review):
            # Use the same cache/scene invalidation as other dataset moves.
            # A moved image must not remain editable through a stale crop.
            remove_from_review(removed)
            if list(self.main_window.image_files) == image_files:
                return
        self.main_window.image_files = image_files
        self.main_window.filtered_image_files = list(image_files)
        if hasattr(self.main_window, "update_list_view"):
            self.main_window.update_list_view(image_files)
        if (image_files and getattr(self.main_window, "current_file", None) not in remaining
                and hasattr(self.main_window, "on_list_view_clicked")):
            # Reverting a split can repopulate a formerly empty dataset.
            model = self.main_window.List_view.model()
            self.main_window.on_list_view_clicked(model.index(0, 0))

    def split_dataset(self):
        dataset_path = self.dataset_path()
        if not dataset_path or not dataset_path.is_dir():
            QMessageBox.warning(self, "Split Dataset", "Select a valid dataset folder first.")
            return

        classes_path = resolve_dataset_classes_path(str(dataset_path), migrate=True)
        classes_file = Path(classes_path) if classes_path else Path()
        if not classes_path or not classes_file.is_file():
            QMessageBox.warning(self, "Split Dataset", ".darkfusion/classes.txt was not found for this dataset.")
            return

        image_files = self._image_files(dataset_path)
        if not image_files:
            QMessageBox.information(self, "Split Dataset", "No root images were found to split.")
            return

        split_folders = self._split_folders(dataset_path)
        if split_folders:
            reply = QMessageBox.question(
                self,
                "Split Dataset",
                "Split folders already exist. Continue and add root images into those folders?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._save_split_settings()
        chunk_size = max(1, int(self.chunk_spinbox.value()))
        prefix = self.folder_prefix()
        metadata_paths = migrate_darkfusion_dataset_metadata(str(dataset_path))
        if "classes.txt" not in metadata_paths:
            canonical_classes = dataset_metadata_path(
                str(dataset_path), "classes.txt", create_parent=True
            )
            shutil.copy2(str(classes_file), canonical_classes)
            metadata_paths["classes.txt"] = canonical_classes

        progress = QtWidgets.QProgressDialog("Splitting dataset...", "", 0, len(image_files), self)
        progress.setWindowTitle("Split Dataset")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(250)
        progress.setCancelButton(None)

        moved_count = 0
        try:
            for chunk_index, start in enumerate(range(0, len(image_files), chunk_size), start=1):
                folder_path = dataset_path / f"{prefix}{chunk_index}"
                folder_path.mkdir(parents=True, exist_ok=True)
                for metadata_name, metadata_source in metadata_paths.items():
                    metadata_target = dataset_metadata_path(
                        str(folder_path), metadata_name, create_parent=True
                    )
                    shutil.copy2(metadata_source, metadata_target)

                for image_file in image_files[start:start + chunk_size]:
                    destination_image = self._unique_image_destination(folder_path / image_file.name)
                    shutil.move(str(image_file), str(destination_image))

                    label_file = dataset_path / f"{image_file.stem}.txt"
                    if label_file.exists():
                        destination_label = destination_image.with_suffix(".txt")
                        shutil.move(str(label_file), str(destination_label))

                    moved_count += 1
                    progress.setValue(moved_count)
                    QApplication.processEvents()

            progress.setValue(len(image_files))
            self.refresh_summary()
            self._refresh_parent_view_if_current_dataset(dataset_path)
            self._set_status(f"Split complete. Moved {moved_count} image(s).", "#7cc7ff")
            QMessageBox.information(self, "Split Dataset", f"Split complete. Moved {moved_count} image(s).")

        except Exception as e:
            logger.error(f"Dataset split failed: {e}")
            QMessageBox.critical(self, "Split Dataset", f"Dataset split failed:\n{e}")
            self.refresh_summary()

    def _move_file_unique(self, source, destination):
        destination = self._unique_destination(destination)
        shutil.move(str(source), str(destination))
        return destination

    def _move_directory_contents(self, source_dir, destination_dir):
        destination_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(source_dir.iterdir()):
            if item.is_file():
                self._move_file_unique(item, destination_dir / item.name)
            elif item.is_dir():
                self._move_directory_contents(item, destination_dir / item.name)
                if not any(item.iterdir()):
                    item.rmdir()

    def _merge_split_metadata(self, source_dir, dataset_path):
        destination_dir = Path(dataset_metadata_directory(str(dataset_path), create=True))
        for item in sorted(source_dir.iterdir()):
            destination = destination_dir / item.name
            if item.is_file():
                if not destination.exists():
                    shutil.move(str(item), str(destination))
                elif _files_have_same_content(str(item), str(destination)):
                    item.unlink()
                else:
                    self._move_file_unique(item, destination)
            elif item.is_dir():
                self._move_directory_contents(item, destination)
                if not any(item.iterdir()):
                    item.rmdir()
        if source_dir.exists() and not any(source_dir.iterdir()):
            source_dir.rmdir()

    def revert_split(self):
        dataset_path = self.dataset_path()
        if not dataset_path or not dataset_path.is_dir():
            QMessageBox.warning(self, "Revert Split", "Select a valid dataset folder first.")
            return

        split_folders = self._split_folders(dataset_path)
        if not split_folders:
            QMessageBox.information(self, "Revert Split", "No split folders were found.")
            return

        reply = QMessageBox.question(
            self,
            "Revert Split",
            f"Move files back from {len(split_folders)} split folder(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        total_files = sum(1 for folder in split_folders for item in folder.rglob("*") if item.is_file())
        progress = QtWidgets.QProgressDialog("Reverting split...", "", 0, max(1, total_files), self)
        progress.setWindowTitle("Revert Split")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(250)
        progress.setCancelButton(None)

        moved_count = 0
        try:
            for folder in split_folders:
                for image_file in sorted(
                    item
                    for item in folder.iterdir()
                    if item.is_file() and item.suffix.lower() in OUTPUT_IMAGE_SUFFIXES
                ):
                    destination_image = self._unique_image_destination(dataset_path / image_file.name)
                    shutil.move(str(image_file), str(destination_image))

                    label_file = image_file.with_suffix(".txt")
                    if label_file.exists():
                        shutil.move(str(label_file), str(destination_image.with_suffix(".txt")))

                    moved_count += 1
                    progress.setValue(min(progress.maximum(), moved_count))
                    QApplication.processEvents()

                for item in sorted(folder.iterdir()):
                    if item.is_file() and item.name.lower() == "classes.txt":
                        destination_classes = dataset_path / "classes.txt"
                        if destination_classes.exists():
                            item.unlink()
                        else:
                            shutil.move(str(item), str(destination_classes))
                    elif item.is_file():
                        self._move_file_unique(item, dataset_path / item.name)
                    elif item.is_dir() and item.name.lower() == PROJECT_SETTINGS_DIR:
                        self._merge_split_metadata(item, dataset_path)
                    elif item.is_dir():
                        self._move_directory_contents(item, dataset_path / item.name)
                        if not any(item.iterdir()):
                            item.rmdir()

                    progress.setValue(min(progress.maximum(), progress.value() + 1))
                    QApplication.processEvents()

                if not any(folder.iterdir()):
                    folder.rmdir()

            progress.setValue(progress.maximum())
            self.refresh_summary()
            self._refresh_parent_view_if_current_dataset(dataset_path)
            self._set_status(f"Revert complete. Restored {moved_count} image(s).", "#7cc7ff")
            QMessageBox.information(self, "Revert Split", f"Revert complete. Restored {moved_count} image(s).")

        except Exception as e:
            logger.error(f"Dataset split revert failed: {e}")
            QMessageBox.critical(self, "Revert Split", f"Revert failed:\n{e}")
            self.refresh_summary()

