"""Darknet training preparation, CFG/YAML editing, and anchor tools.

These are the existing main-window callbacks, bound by the application in their
original order and using its shared project and trainer state.
"""

import glob
import json
import logging
import numpy as np
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import yaml
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem
from darkfusion_dataset_operations import resolve_dataset_pose_schema_path
from darkfusion_gpu_memory import gpu_work
from darkfusion_ui_controls import blocked_signals


logger = logging.getLogger("UltraDarkFusionLogger")


def suggested_cfg_path_for_weights(self, weights_path):
    weights_path = str(weights_path or "")
    if not weights_path:
        return os.getcwd()

    cfg_candidate = os.path.splitext(weights_path)[0] + ".cfg"
    if os.path.exists(cfg_candidate):
        return cfg_candidate

    model_dir = os.path.dirname(weights_path)
    return model_dir if model_dir else os.getcwd()


def prompt_for_matching_cfg(self, weights_path):
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    start_path = self.suggested_cfg_path_for_weights(weights_path)

    cfg_name, _ = QFileDialog.getOpenFileName(
        self,
        "Select Matching Darknet CFG",
        start_path,
        "Config Files (*.cfg)",
        options=options
    )

    if not cfg_name:
        logger.info("Darknet CFG selection skipped after loading weights.")
        return False

    self.remember_dialog_selection("model", cfg_name)
    self.cfg_file_path = cfg_name
    self.settings["lastCfgPath"] = self.normalize_path(cfg_name)
    self.saveSettings()
    logger.info(f"Loaded Darknet CFG: {cfg_name}")
    return self.initialize_yolo()


def open_cfg(self):
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    start_path = (
        self.suggested_cfg_path_for_weights(self.weights_file_path)
        if getattr(self, "weights_file_path", None)
        else self.dialog_start_directory("model")
    )
    file_name, _ = QFileDialog.getOpenFileName(
        self, "Open CFG", start_path, "Config Files (*.cfg)", options=options)
    if file_name:
        self.remember_dialog_selection("model", file_name)
        self.cfg_file_path = file_name  # Store the cfg file path
        self.settings["lastCfgPath"] = self.normalize_path(file_name)
        self.saveSettings()
        initialized = False
        has_darknet_weights = (
            getattr(self, "weights_file_path", None)
            and self.determine_model_type(self.weights_file_path) == "weights"
        )
        if has_darknet_weights:
            initialized = self.initialize_yolo()

        QMessageBox.information(
            self,
            "CFG File Loaded",
            "Darknet weights and CFG are ready."
            if initialized
            else (
                "CFG file loaded. Select matching Darknet weights before auto-labeling."
                if not has_darknet_weights
                else "CFG file loaded, but the model could not be initialized. Check the weights and classes.txt."
            )
        )


def import_images(self):
    """
        Imports images from the selected directory and loads class names.
        """
    self.stop_live_collect_runtime(reset_input=True)
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    directory = QFileDialog.getExistingDirectory(
        self,
        "Select Import Directory",
        self.dialog_start_directory("dataset"),
        options=options,
    )

    if directory:
        self.remember_dialog_selection("dataset", directory)
        self._graphics_dataset_explicitly_opened = True
        self.image_directory = directory

        self.images = self.get_image_files(directory)

        if not self.images:
            QMessageBox.warning(self, 'Warning', 'No images found in the selected directory.')
        else:
            logger.info(f"Found {len(self.images)} images in {directory}")

        self.classes = self.load_classes(data_directory=directory, default_classes=["person"])
        self.autoload_training_data_yaml_for_dataset(directory)
        self.queue_settings_save()


def output_paths(self):
    """
        Handles creation of train.txt and valid.txt.
        """
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly

    selected_option = self.dropdown.currentText()
    default_filename = selected_option if selected_option in ["valid.txt", "train.txt"] else ""
    default_dir = getattr(self, "train_valid_output_directory", "") or self.settings.get("trainValidOutputDirectory", "")
    start_dir = self.dialog_start_directory("training", default_dir)
    default_path = os.path.join(start_dir, default_filename) if start_dir else default_filename

    save_file, _ = QFileDialog.getSaveFileName(
        self, "Select Output File", default_path, "Text Files (*.txt);;All Files (*)", options=options
    )

    if save_file:
        self.remember_dialog_selection("training", save_file)
        output_dir = os.path.dirname(save_file).replace("\\", "/")
        self.train_valid_output_directory = output_dir
        self.queue_settings_save()

        if selected_option == "train.txt":
            train_txt_path = os.path.join(output_dir, "train.txt")
            with open(train_txt_path, "w", encoding="utf-8") as f:
                for image in self.images:
                    abs_path = os.path.abspath(image).replace("\\", "/")
                    f.write(abs_path + "\n")
            QMessageBox.information(self, 'Information', 'train.txt file has been created!')

        elif selected_option == "valid.txt":
            valid_percent = int(self.valid_percent.currentText().replace('%', ''))

            images_copy = self.images[:]
            random.shuffle(images_copy)
            num_valid_images = int(len(images_copy) * (valid_percent / 100.0))

            # Keep validation entries tied to real label files.
            valid_images = []
            for image in images_copy:
                txt_file = os.path.splitext(image)[0] + ".txt"
                if os.path.exists(txt_file) and os.path.getsize(txt_file) > 0:
                    valid_images.append(image)
                if len(valid_images) >= num_valid_images:
                    break

            with open(save_file, "w", encoding="utf-8") as f:
                for image in valid_images:
                    abs_path = os.path.abspath(image).replace("\\", "/")
                    f.write(abs_path + "\n")

            QMessageBox.information(self, 'Information', 'valid.txt file has been created!')

        try:
            self.create_and_update_yaml_and_data_files(output_dir, selected_option)
        except Exception as error:
            QMessageBox.critical(self, "Training File Generation", str(error))


def create_and_update_yaml_and_data_files(self, output_dir, selected_option, source_yaml_path="", data_yaml_path=""):
    """
        Creates obj.data, obj.yaml, and obj.names with correct class count and paths.
        Ensures obj.names matches classes.txt.
        """
    output_dir = output_dir.replace("\\", "/")
    source_yaml_path = self.normalize_path(source_yaml_path or "")

    dataset_dir = self.image_directory

    if not dataset_dir:
        logger.warning("No dataset directory found. Import images before exporting train/valid files.")
        return

    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Dataset directory is not an existing folder: {dataset_dir}")

    self.classes = self.load_classes(data_directory=dataset_dir)

    if not self.classes:
        raise ValueError("No valid classes found. Check classes.txt in the dataset's .darkfusion folder.")

    class_numb = len(self.classes)

    # Training metadata paths.
    train_txt_path = os.path.join(output_dir, "train.txt").replace("\\", "/")
    valid_txt_path = os.path.join(output_dir, "valid.txt").replace("\\", "/") if selected_option == "valid.txt" else train_txt_path
    obj_names_file = os.path.join(output_dir, "obj.names").replace("\\", "/")
    data_file_path = os.path.join(output_dir, "obj.data").replace("\\", "/")
    obj_yaml_file = self.normalize_path(data_yaml_path or "") or os.path.join(output_dir, "obj.yaml").replace("\\", "/")
    backup_dir = os.path.join(dataset_dir, "runs").replace("\\", "/")
    preserved_yaml = {}
    for yaml_candidate in (source_yaml_path, obj_yaml_file):
        yaml_candidate = self.normalize_path(yaml_candidate or "")
        if not yaml_candidate or not os.path.exists(yaml_candidate):
            continue
        try:
            from darkfusion_tune_settings import read_training_configuration
            loaded_yaml = read_training_configuration(yaml_candidate)
            if isinstance(loaded_yaml, dict):
                preserved_yaml = dict(loaded_yaml)
                break
        except Exception as e:
            logger.warning(f"Could not preserve metadata from {yaml_candidate}: {e}")

    # Keep obj.names in sync with classes.txt.
    try:
        self._training_eval_write_text_atomic(
            obj_names_file,
            "".join(class_name + "\n" for class_name in self.classes),
        )

        logger.info(f"Updated obj.names to match classes.txt (loaded {len(self.classes)} classes)")

    except Exception as e:
        logger.error(f"Failed to update obj.names: {e}")
        raise

    # Create Darknet obj.data.
    try:
        self._training_eval_write_text_atomic(
            data_file_path,
            f"classes = {class_numb}\n"
                f"train = {train_txt_path}\n"
                f"valid = {valid_txt_path}\n"
                f"names = {obj_names_file}\n"
                f"backup = {backup_dir}\n",
        )

        logger.info(f" Created `obj.data` with {class_numb} classes.")
    except Exception as e:
        logger.error(f"Failed to create obj.data: {e}")
        raise
    # Create Ultralytics obj.yaml.
    try:
        yaml_payload = {
            "path": output_dir,
            "train": train_txt_path,
            "val": valid_txt_path,
            "nc": class_numb,
        }
        from ultralytics.utils import DEFAULT_CFG_DICT
        # Dataset metadata and training cfg have separate owners. Preserve pose,
        # channels and namespaced/custom metadata, not obsolete training defaults.
        training_only_keys = set(DEFAULT_CFG_DICT) - {"task", "channels"}
        for key, value in preserved_yaml.items():
            if key not in {"path", "train", "val", "nc", "names"} and key not in training_only_keys:
                yaml_payload[key] = value

        # points.json is the editable DarkFusion source when present; otherwise keep obj.yaml metadata.
        points_json_path = resolve_dataset_pose_schema_path(dataset_dir, migrate=True)
        if os.path.exists(points_json_path):
            try:
                with open(points_json_path, "r", encoding="utf-8") as json_file:
                    points_data = json.load(json_file)
                for pose_key in ("kpt_shape", "flip_idx", "skeleton", "kpt_names"):
                    yaml_payload.pop(pose_key, None)
                keypoints = points_data.get("keypoints", [])
                num_keypoints = len(keypoints)
                if num_keypoints > 0:
                    configured_shape = points_data.get("kpt_shape")
                    dimensions = 3
                    if isinstance(configured_shape, (list, tuple)) and len(configured_shape) >= 2:
                        try:
                            dimensions = max(2, min(3, int(configured_shape[1])))
                        except Exception:
                            dimensions = 3
                    yaml_payload["kpt_shape"] = [num_keypoints, dimensions]
                    yaml_payload["flip_idx"] = list(points_data.get("flip_idx", []) or [])
                    yaml_payload["skeleton"] = list(points_data.get("skeleton", []) or [])
                    kpt_names = []
                    for idx, kp in enumerate(keypoints):
                        if isinstance(kp, dict):
                            name = kp.get("label") or kp.get("name") or f"keypoint_{idx}"
                        else:
                            name = str(kp) if kp else f"keypoint_{idx}"
                        kpt_names.append(str(name))
                    if kpt_names:
                        yaml_payload["kpt_names"] = {0: kpt_names}
                    logger.info("Injected kpt_shape, flip_idx, skeleton, and kpt_names from points.json")
            except Exception as e:
                logger.warning(f"Failed to parse keypoint config from points.json: {e}")
        elif any(key in yaml_payload for key in ("kpt_shape", "flip_idx", "skeleton", "kpt_names")):
            logger.info("Preserved keypoint metadata from existing obj.yaml.")
        else:
            logger.info("No points.json or preserved keypoint metadata found. Skipping keypoint structure.")

        yaml_payload["names"] = {i: str(name) for i, name in enumerate(self.classes)}
        yaml_content = "# Ultralytics dataset configuration\n\n" + yaml.safe_dump(
            yaml_payload,
            sort_keys=False,
            allow_unicode=True,
        )
        self._training_eval_write_text_atomic(obj_yaml_file, yaml_content)

        logger.info("Updated dataset YAML %s with %s classes.", obj_yaml_file, class_numb)
        self._training_eval_yaml_split_report_cache = {}
        self._set_training_data_yaml_path(obj_yaml_file)
        if isinstance(getattr(self, "file_paths", None), dict):
            self.file_paths["data"] = data_file_path
        if getattr(self, "data_label", None) is not None:
            self.data_label.setText(f"File: {data_file_path}")
        self.queue_settings_save()

    except Exception as e:
        logger.error(f"Failed to create obj.yaml: {e}")
        raise


def browse_file_clicked(self, title, file_types, key, multiple=False):
    file_names = self.open_file_dialog(
        title, file_types, multiple=multiple
    )
    if multiple:
        file_names = [
            file for file in file_names
            if file.endswith(('.weights', '.conv.')) or re.match(r'.*\.conv\.\d+', file)
        ]
        file_label = "Files: " + ", ".join(file_names).rstrip()
    else:
        file_label = f"File: {file_names}"

    self.file_paths[key] = file_names
    getattr(self, f"{key}_label").setText(file_label)
    self.queue_settings_save()


def browse_data_clicked(self):
    self.browse_file_clicked(
        "Select Data File", "Data Files (*.data);;All Files (*)", "data"
    )


def browse_cfg_clicked(self):
    self.browse_file_clicked(
        "Select Config File", "Config Files (*.cfg);;All Files (*)", "cfg"
    )


def browse_weights_clicked(self):
    self.browse_file_clicked(
        "Select Weights Files", "Weights Files (*.weights *.conv.*);;All Files (*)", "weights", multiple=True
    )


def normalize_darknet_gpu_options(self, changed=None):
    if not all(hasattr(self, name) for name in ("gpu1", "gpu2")):
        return

    if changed == "gpu1" and self.gpu1.isChecked():
        with blocked_signals(self.gpu2):
            self.gpu2.setChecked(False)
    elif changed == "gpu2" and self.gpu2.isChecked():
        with blocked_signals(self.gpu1):
            self.gpu1.setChecked(False)
    elif self.gpu1.isChecked() and self.gpu2.isChecked():
        with blocked_signals(self.gpu1):
            self.gpu1.setChecked(False)


def build_darknet_train_command(self):
    data_file = self.file_paths.get("data", "")
    cfg_file = self.file_paths.get("cfg", "")
    weights_files = self.file_paths.get("weights", []) or []
    weights_file = weights_files[-1] if weights_files else ""

    required_files = {
        "data": data_file,
        "cfg": cfg_file,
    }
    if weights_file:
        required_files["weights"] = weights_file

    missing = [label for label, path in required_files.items() if not path or not os.path.isfile(path)]
    if missing:
        raise ValueError(f"Select a valid {', '.join(missing)} file before training.")

    if self.clear_check.isChecked() and not weights_file:
        raise ValueError("Clear Weights only applies when a starting weights file is selected.")

    command = ["darknet", "detector"]

    # Hank AI Darknet documents detector flags before the train command.
    if self.gpu2.isChecked():
        command.extend(["-gpus", "0,1,2"])
    elif self.gpu1.isChecked():
        command.extend(["-gpus", "0,1"])

    if not self.map_check.isChecked():
        command.append("-map")

    if not self.show_chart.isChecked():
        command.append("-dont_show")

    if self.cache.isChecked():
        cache_value = self.cache_input.text().strip()
        if not cache_value.isdigit() or int(cache_value) <= 0:
            raise ValueError("Cache must be a positive number when enabled.")
        command.extend(["-cache", cache_value])

    if self.random.isChecked():
        command.append("-random")

    command.extend(["train", data_file, cfg_file])

    if weights_file:
        if self.clear_check.isChecked():
            command.append("-clear")
        command.append(weights_file)

    return command, os.path.dirname(data_file) or None


def format_command_for_log(self, command):
    if sys.platform == 'win32':
        return subprocess.list2cmdline(command)
    return shlex.join(command)


@gpu_work("Darknet training", keep=(), external=True, exclusive=True)
def darknet_train(self):
    try:
        command, working_dir = self.build_darknet_train_command()
        search_path = os.pathsep.join(filter(None, (working_dir, os.environ.get("PATH", ""))))
        if shutil.which(command[0], path=search_path) is None:
            platform_note = (
                " This Windows launcher needs darknet.exe; a Darknet installation inside WSL "
                "must be launched from WSL."
                if sys.platform == "win32" else ""
            )
            raise FileNotFoundError(
                "Darknet executable was not found. Add Darknet to PATH or place it beside "
                "the selected obj.data file." + platform_note
            )
        command_display = self.format_command_for_log(command)

        if sys.platform == 'win32':
            process = subprocess.Popen(["cmd.exe", "/k", command_display], cwd=working_dir)
        else:
            try:
                process = subprocess.Popen(["gnome-terminal", "--working-directory", working_dir, "--"] + command)
            except (FileNotFoundError, TypeError):
                process = subprocess.Popen(command, cwd=working_dir)

        logger.info(f"Command to be executed: {command_display}")
        return process
    except Exception as e:
        logger.error(f"Failed to start Darknet training: {e}")
        QtWidgets.QMessageBox.critical(self, "Darknet Training", str(e))


def import_data(self):
    """Import text files and class configurations."""
    data_directory = QtWidgets.QFileDialog.getExistingDirectory(
        self, 'Select Data Directory', self.dialog_start_directory("dataset")
    )
    if not data_directory:
        logger.warning("No directory selected.")
        return
    self.remember_dialog_selection("dataset", data_directory)

    data_directory = os.path.normpath(data_directory)
    self._graphics_dataset_explicitly_opened = True
    self.text_files = glob.glob(os.path.join(data_directory, '*.txt'))
    self.image_directory = data_directory

    logger.info(f"Detected text files: {self.text_files}")

    # Cache class mapping once for Darknet CFG updates.
    self.class_mapping = self.load_classes(data_directory)
    if self.class_mapping:
        num_classes = len(self.class_mapping)
        logger.info(f"Loaded {num_classes} classes.")
        self.update_classes_and_batches(num_classes)
    else:
        logger.error("Could not load classes.txt.")
        QtWidgets.QMessageBox.warning(self, "Error", "Could not load classes.txt.")


def update_classes_and_batches(self, num_classes):
    """
        Update Darknet training values that depend on the project class count.
        """
    max_batches = num_classes * 5000
    steps = ",".join([str(int(p * max_batches)) for p in self.STEP_PERCENTAGES])
    scales = self.DEFAULT_SCALES

    updates = {
        'classes': num_classes,
        'max_batches': max_batches,
    }

    # Update global [net] schedule values.
    net_updates = {
        'steps': steps,
        'scales': scales,
    }

    # Update class count and batch settings in their owning sections.
    for param_name, value in updates.items():
        if param_name == 'classes':
            self.update_cfg_param(param_name, value, section="[yolo]")
        else:
            self.update_cfg_param(param_name, value, section="[net]")

    for param_name, value in net_updates.items():
        self.update_cfg_param(param_name, value, section="[net]")

    self.update_yolo_filters(num_classes)
    if hasattr(self, 'filename') and self.filename:
        self.parse_cfg_file(self.filename)


def calculate_anchors(self):
    """Calculate anchors and update CFG file."""
    if not hasattr(self, 'class_mapping') or not self.class_mapping:
        QtWidgets.QMessageBox.warning(self, "Error", "Classes are not loaded.")
        return

    if not self.text_files:
        QtWidgets.QMessageBox.warning(self, "Error", "No annotation files loaded.")
        return

    if not hasattr(self, 'filename') or not self.filename:
        QtWidgets.QMessageBox.warning(self, "Error", "Please open a .cfg file first.")
        return

    num_clusters = self.clusters_spinbox.value()
    width_value = self.width_spinbox.value()
    height_value = self.height_spinbox.value()
    yolo_count = self.count_cfg_sections("yolo")
    if yolo_count <= 0:
        QtWidgets.QMessageBox.warning(self, "Error", "No [yolo] sections found in the CFG file.")
        return

    if num_clusters < yolo_count:
        QtWidgets.QMessageBox.warning(
            self,
            "Anchor Clusters",
            "Use at least one anchor cluster per [yolo] section.",
        )
        return

    annotation_dims = self.get_annotation_dimensions(valid_classes=set(range(len(self.class_mapping))))

    max_samples = 50000
    if len(annotation_dims) > max_samples:
        annotation_dims = random.Random(0).sample(annotation_dims, max_samples)

    if not annotation_dims:
        QtWidgets.QMessageBox.warning(self, "Error", "No valid annotations found.")
        return

    if len(annotation_dims) < num_clusters:
        QtWidgets.QMessageBox.warning(
            self,
            "Anchor Clusters",
            f"Need at least {num_clusters} labeled boxes to calculate {num_clusters} anchors.",
        )
        return

    try:
        centroids = self.calculate_anchor_clusters(annotation_dims, num_clusters)
    except ValueError as exc:
        QtWidgets.QMessageBox.warning(self, "Anchor Clusters", str(exc))
        return

    anchors_pixel = np.maximum(
        1,
        np.rint(np.asarray(centroids, dtype=np.float64) * [width_value, height_value]).astype(int),
    )
    anchors_str = ', '.join(map(str, anchors_pixel.flatten()))
    self.imported_anchors = anchors_pixel.tolist()

    yolo_masks = self.default_yolo_masks(num_clusters, yolo_count)
    if num_clusters % yolo_count != 0:
        QtWidgets.QMessageBox.information(
            self,
            "Anchor Masks",
            "Anchor clusters do not divide evenly across YOLO heads; the first heads will receive one extra mask.",
        )

    self.update_cfg_param('width', width_value, section="[net]")
    self.update_cfg_param('height', height_value, section="[net]")
    self.update_cfg_param('anchors', anchors_str, section="[yolo]", update_all=True)

    # num, mask, and the preceding convolutional filters must agree.
    self.update_cfg_param('num', num_clusters, section="[yolo]", update_all=True)
    self.update_cfg_yolo_masks(yolo_masks)
    self.update_yolo_filters(len(self.class_mapping))
    self.parse_cfg_file(self.filename)

    avg_iou = round(self.calculate_avg_iou(annotation_dims, centroids) * 100, 2)
    self.saveAnchorsSettings(avg_iou)

    if self.show_checkbox.isChecked():
        self.plot_anchors(centroids, avg_iou)


def get_annotation_dimensions(self, valid_classes):
    """Retrieve dimensions, only considering valid classes."""
    annotation_dims = []
    for text_file in self.text_files:
        if text_file.endswith("classes.txt"):
            continue
        with open(text_file, 'r', encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        if class_id in valid_classes:
                            width, height = float(parts[3]), float(parts[4])
                            if 0.0 < width <= 1.0 and 0.0 < height <= 1.0:
                                annotation_dims.append((width, height))
                    except ValueError:
                        logger.warning(f"Invalid numeric values in {text_file}")
    return annotation_dims


def anchor_iou_matrix(self, boxes, centroids):
    boxes = np.asarray(boxes, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)

    box_area = boxes[:, 0:1] * boxes[:, 1:2]
    centroid_area = centroids[:, 0] * centroids[:, 1]
    intersect_w = np.minimum(boxes[:, 0:1], centroids[:, 0])
    intersect_h = np.minimum(boxes[:, 1:2], centroids[:, 1])
    intersection = intersect_w * intersect_h
    union = box_area + centroid_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def initialize_anchor_centroids(self, boxes, num_clusters, rng):
    centroids = [boxes[int(rng.integers(len(boxes)))]]

    while len(centroids) < num_clusters:
        distances = 1.0 - np.max(self.anchor_iou_matrix(boxes, np.asarray(centroids)), axis=1)
        probabilities = distances ** 2
        total = probabilities.sum()

        if total <= 0:
            centroids.append(boxes[int(rng.integers(len(boxes)))])
            continue

        probabilities = probabilities / total
        centroids.append(boxes[int(rng.choice(len(boxes), p=probabilities))])

    return np.asarray(centroids, dtype=np.float64)


def calculate_anchor_clusters(self, annotation_dims, num_clusters, attempts=12, max_iterations=100):
    boxes = np.asarray(annotation_dims, dtype=np.float64)
    valid_mask = (
        (boxes[:, 0] > 0.0)
        & (boxes[:, 1] > 0.0)
        & (boxes[:, 0] <= 1.0)
        & (boxes[:, 1] <= 1.0)
    )
    boxes = boxes[valid_mask]

    if len(boxes) < num_clusters:
        raise ValueError("Not enough valid boxes for requested anchor clusters.")

    rng = np.random.default_rng(0)
    best_centroids = None
    best_iou = -1.0

    for _attempt in range(attempts):
        centroids = self.initialize_anchor_centroids(boxes, num_clusters, rng)
        previous_assignments = None

        for _iteration in range(max_iterations):
            distances = 1.0 - self.anchor_iou_matrix(boxes, centroids)
            assignments = np.argmin(distances, axis=1)

            if previous_assignments is not None and np.array_equal(assignments, previous_assignments):
                break

            new_centroids = centroids.copy()
            for cluster_index in range(num_clusters):
                cluster_boxes = boxes[assignments == cluster_index]
                if len(cluster_boxes) == 0:
                    new_centroids[cluster_index] = boxes[int(rng.integers(len(boxes)))]
                else:
                    new_centroids[cluster_index] = np.median(cluster_boxes, axis=0)

            if np.allclose(centroids, new_centroids, rtol=0.0, atol=1e-9):
                centroids = new_centroids
                break

            centroids = new_centroids
            previous_assignments = assignments

        avg_iou = self.calculate_avg_iou(boxes, centroids)
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_centroids = centroids.copy()

    return sorted(best_centroids.tolist(), key=lambda c: c[0] * c[1])


def cfg_section_ranges(self, lines=None):
    if lines is None:
        if not hasattr(self, 'filename') or not self.filename:
            return []
        with open(self.filename, 'r', encoding="utf-8") as f:
            lines = f.readlines()

    sections = []
    current = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            if current is not None:
                current["end"] = index
                sections.append(current)

            section_name = stripped[1:stripped.find("]")].strip().lower()
            current = {"name": section_name, "start": index, "end": len(lines)}

    if current is not None:
        current["end"] = len(lines)
        sections.append(current)

    return sections


def count_cfg_sections(self, section_name):
    if not hasattr(self, 'filename') or not self.filename:
        return 0

    try:
        with open(self.filename, 'r', encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return 0

    target = str(section_name).strip().lower()
    return sum(1 for section in self.cfg_section_ranges(lines) if section["name"] == target)


def parse_cfg_section_params(self, lines, section):
    params = {}
    for line in lines[section["start"] + 1:section["end"]]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        params[key.strip().lower()] = value.split("#", 1)[0].strip()

    return params


def set_cfg_section_param(self, lines, section, param_name, value):
    key = str(param_name).strip()
    prefix = key.lower()
    replacement = f"{key}={value}\n"

    for index in range(section["start"] + 1, section["end"]):
        stripped = lines[index].strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue

        existing_key = stripped.split("=", 1)[0].strip().lower()
        if existing_key == prefix:
            lines[index] = replacement
            return True

    lines.insert(section["end"], replacement)
    return True


def mask_indices_from_value(self, value):
    if value is None:
        return []

    indices = []
    for part in str(value).replace(" ", "").split(","):
        if not part:
            continue
        try:
            indices.append(int(part))
        except ValueError:
            pass

    return indices


def yolo_mask_count(self, params):
    mask_indices = self.mask_indices_from_value(params.get("mask"))
    if mask_indices:
        return len(mask_indices)

    try:
        return max(1, int(params.get("num", 3)))
    except Exception:
        return 3


def default_yolo_masks(self, num_clusters, yolo_count):
    num_clusters = max(1, int(num_clusters))
    yolo_count = max(1, int(yolo_count))
    base_count = num_clusters // yolo_count
    extra = num_clusters % yolo_count

    groups = []
    start = 0
    for index in range(yolo_count):
        group_size = base_count + (1 if index < extra else 0)
        stop = min(num_clusters, start + max(1, group_size))
        groups.append(list(range(start, stop)))
        start = stop

    if start < num_clusters:
        groups[-1].extend(range(start, num_clusters))

    return list(reversed(groups))


def update_cfg_yolo_masks(self, yolo_masks):
    if not hasattr(self, 'filename') or not self.filename:
        return

    with open(self.filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    sections = self.cfg_section_ranges(lines)
    yolo_sections = [section for section in sections if section["name"] == "yolo"]

    for section, mask_values in zip(yolo_sections, yolo_masks):
        mask_text = ",".join(str(value) for value in mask_values)
        self.set_cfg_section_param(lines, section, "mask", mask_text)

    with open(self.filename, 'w', encoding="utf-8") as f:
        f.writelines(lines)


def update_yolo_filters(self, num_classes):
    if not hasattr(self, 'filename') or not self.filename:
        return

    with open(self.filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    sections = self.cfg_section_ranges(lines)
    updated = 0

    for section_index, section in enumerate(sections):
        if section["name"] != "yolo":
            continue

        params = self.parse_cfg_section_params(lines, section)
        filters = (int(num_classes) + 5) * self.yolo_mask_count(params)

        conv_section = None
        for previous_index in range(section_index - 1, -1, -1):
            if sections[previous_index]["name"] == "convolutional":
                conv_section = sections[previous_index]
                break

        if conv_section is None:
            logger.warning("Could not find [convolutional] section before [yolo].")
            continue

        self.set_cfg_section_param(lines, conv_section, "filters", filters)
        updated += 1

    with open(self.filename, 'w', encoding="utf-8") as f:
        f.writelines(lines)

    logger.info(f"Updated filters before {updated} [yolo] section(s) for {num_classes} classes.")


def update_cfg_param(self, param_name, value, section=None, update_all=False):
    """Update a parameter in the CFG file, optionally updating all occurrences."""
    if not hasattr(self, 'filename') or not self.filename:
        QtWidgets.QMessageBox.warning(self, "Error", "No .cfg file loaded.")
        return

    with open(self.filename, 'r') as f:
        lines = f.readlines()

    new_lines = []
    in_target_section = False
    updated_count = 0

    for line in lines:
        stripped_line = line.strip()

        # Track the active section.
        if stripped_line.startswith('['):
            if section:
                in_target_section = (stripped_line == section)
            else:
                in_target_section = True  # No section specified; global update

        is_param_line = (
            in_target_section
            and stripped_line
            and not stripped_line.startswith("#")
            and "=" in stripped_line
            and stripped_line.split("=", 1)[0].strip() == str(param_name)
        )

        if is_param_line:
            new_lines.append(f"{param_name}={value}\n")
            updated_count += 1
            if not update_all:
                in_target_section = False  # Exit section unless updating all occurrences
        else:
            new_lines.append(line)

    if updated_count == 0 and section:
        # Missing section-scoped value: append it under that section.
        new_lines.append(f"{section}\n{param_name}={value}\n")

    with open(self.filename, 'w') as f:
        f.writelines(new_lines)

    self.update_table_param(param_name, value)
    logger.info(f"Updated '{param_name}' to '{value}' in {updated_count if updated_count else 'new'} occurrence(s).")


def update_table_param(self, param_name, value):
    """Update a parameter in the cfg_table."""
    for row in range(self.cfg_table.rowCount()):
        param_item = self.cfg_table.item(row, 0)
        if param_item and param_name in param_item.text():
            self.cfg_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))
            logger.info(f"Updated '{param_name}' in cfg_table to {value}")


def calculate_avg_iou(self, annotation_dims, centroids):
    """Calculate average IoU for anchors."""
    boxes = np.asarray(annotation_dims, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if len(boxes) == 0 or len(centroids) == 0:
        return 0.0

    return float(np.mean(np.max(self.anchor_iou_matrix(boxes, centroids), axis=1)))


def update_cfg_anchors(self, anchors):
    """
        Update anchors specifically in existing [yolo] sections of the CFG file.
        """
    try:
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        new_config = []
        inside_yolo_section = False
        yolo_section_count = 0

        for line in lines:
            stripped_line = line.strip()

            if stripped_line.startswith("[yolo]"):
                inside_yolo_section = True
                yolo_section_count += 1

            elif stripped_line.startswith('[') and inside_yolo_section:
                inside_yolo_section = False

            if inside_yolo_section and stripped_line.startswith("anchors="):
                new_config.append(f"anchors={anchors}\n")
                continue

            new_config.append(line)

        # Anchor updates require existing [yolo] sections.
        if yolo_section_count == 0:
            raise ValueError("No [yolo] sections found in the CFG file!")

        with open(self.filename, 'w') as f:
            f.writelines(new_config)

        logger.info(f"Updated anchors in all {yolo_section_count} [yolo] sections.")
    except Exception as e:
        QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred while updating anchors: {str(e)}")


def clean_cfg_file(self):
    """
        Remove duplicate entries and maintain correct CFG structure.
        """
    try:
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        seen_params = set()
        clean_config = []
        section_index = -1

        for line in lines:
            stripped_line = line.strip()

            # Track section changes
            if stripped_line.startswith('['):
                section_index += 1
                clean_config.append(line)
                continue

            # Skip duplicate parameters
            if stripped_line and "=" in stripped_line:
                param_name = stripped_line.split('=')[0].strip()
                if (section_index, param_name) in seen_params:
                    continue
                seen_params.add((section_index, param_name))

            clean_config.append(line)

        # Write the cleaned config back to the file
        with open(self.filename, 'w') as f:
            f.writelines(clean_config)

        logger.info("Cleaned duplicates from the CFG file.")
    except Exception as e:
        QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred while cleaning the CFG file: {str(e)}")


def plot_anchors(self, centroids, avg_iou):
    """Simplified anchor plotting."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor('white')

    colors = plt.cm.tab10(np.linspace(0, 1, len(centroids)))

    for i, centroid in enumerate(centroids):
        w, h = centroid
        rect = mpatches.Rectangle((0, 0), w, h, linewidth=2, edgecolor=colors[i], facecolor='none')
        ax.add_patch(rect)

    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title(f'Anchors (Avg IoU: {avg_iou:.2f}%)')
    plt.legend([f'Anchor {i+1}' for i in range(len(centroids))])
    try:
        plt.show(block=True)
    finally:
        plt.close(fig)


def saveAnchorsSettings(self, avg_iou):
    """Save the calculated anchor settings to a JSON file."""
    try:
        anchors = self.imported_anchors if self.imported_anchors else []
        if isinstance(anchors, np.ndarray):
            anchors = anchors.tolist()

        self.settings['anchors'] = anchors
        self.settings['avg_iou'] = avg_iou
        self.saveSettings()

        logger.info(f"Anchor settings saved successfully with avg IoU: {avg_iou}%")

    except Exception as e:
        QtWidgets.QMessageBox.critical(self, "Error", f"Error saving anchor settings: {str(e)}")


def main(self):
    """Main pipeline to import data and calculate anchors."""
    self.import_data()
    class_mapping = self.load_classes(self.image_directory)
    if not class_mapping:
        logger.warning("Classes not loaded. Exiting.")
        return

    if not self.text_files:
        logger.warning("No annotation files detected. Exiting.")
        return

    self.calculate_anchors()


def clear_table(self):
    self.cfg_table.setRowCount(0)


def import_yaml(self):
    """Handles the import of YAML configuration files."""
    if self.file_type != "yaml":
        self.clear_table()

    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly

    initial_directory = self.dialog_start_directory("training", self.default_yaml_path.text())
    file_name, _ = QFileDialog.getOpenFileName(
        None, "Select YAML File", initial_directory, "YAML Files (*.yaml *.yml);;All Files (*)", options=options)

    if file_name:
        self.remember_dialog_selection("training", file_name)
        self.hide_activation_checkbox.setChecked(False)
        self.yaml_filename = file_name
        self.default_yaml_path.setText(file_name)

        # Parse YAML and populate the table
        with open(file_name, 'r', encoding="utf-8") as f:
            self.parsed_yaml = yaml.safe_load(f)

        self.file_type = "yaml"
        self.cfg_table.setColumnCount(2)
        self.cfg_table.setHorizontalHeaderLabels(["Key", "Value"])
        self.cfg_table.setRowCount(len(self.parsed_yaml))

        for row, (key, value) in enumerate(self.parsed_yaml.items()):
            self.cfg_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.cfg_table.setItem(row, 1, QTableWidgetItem(str(value)))

        self.filename = None  # Clear any previous CFG file reference
        self.cfg_open_label.setText("")


def save_table_changes(self, row, column):
    key_item = self.cfg_table.item(row, 0)
    value_item = self.cfg_table.item(row, 1)

    if key_item is not None and value_item is not None:
        key = key_item.text()
        value = value_item.text()

        if self.file_type == "yaml" and self.yaml_filename is not None:
            self.parsed_yaml[key] = self.parse_value(value)
            with open(self.yaml_filename, 'w', encoding="utf-8") as f:
                yaml.dump(self.parsed_yaml, f)


def parse_value(self, value):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'
    return value


def import_anchors(self):
    if not self.filename:
        QtWidgets.QMessageBox.warning(
            self, "Error", "Please open a .cfg file before importing anchors.")
        return


def cfg_open_clicked(self):
    """Handles opening a .cfg file."""
    self.hide_activation_checkbox.setChecked(False)

    if self.file_type != "cfg":
        self.clear_table()

    options = QtWidgets.QFileDialog.Options()
    options |= QtWidgets.QFileDialog.ReadOnly
    initial_directory = self.dialog_start_directory("model", getattr(self, "filename", ""))
    file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
        self, "Select Config File", initial_directory, "Config Files (*.cfg);;All Files (*)", options=options)

    if file_name:
        self.remember_dialog_selection("model", file_name)
        self.filename = file_name
        self.cfg_open_label.setText(f"Cfg: {file_name}")
        self.parse_cfg_file(file_name)
        self.yaml_filename = None
        self.default_yaml_path.setText("")


def toggle_activation_layers(self):
    if not hasattr(self, 'activation_row_count'):
        return

    for row in self.activation_row_count.values():
        if self.hide_activation_checkbox.isChecked():
            self.cfg_table.hideRow(row)
        else:
            self.cfg_table.showRow(row)


def parse_cfg_file(self, file_name=None):
    if file_name is None:
        file_name = self.filename

    try:
        with open(file_name, 'r', encoding="utf-8") as f:
            config = f.read()

        self.activation_row_count = {}
        activation_count = 0
        self.cfg_table.setRowCount(0)
        self.cfg_table.setColumnCount(2)
        self.cfg_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.cfg_table.setAlternatingRowColors(True)

        sections = re.findall(r"\[(.*?)\]\s*([^[]*)", config, re.DOTALL)

        yolo_section_idx = 0

        if self.imported_anchors is not None:
            for idx, (section_type, section_content) in enumerate(sections):
                if section_type == "yolo":
                    section_lines = section_content.strip().split("\n")
                    section_dict = {line.split("=")[0].strip(): line.split("=")[1].strip()
                                    for line in section_lines if "=" in line}

                    section_dict["anchors"] = ', '.join(
                        [f"{x},{y}" for x, y in self.imported_anchors])

                    sections[idx] = (section_type, '\n'.join(
                        [f"{key} = {value}" for key, value in section_dict.items()]))

        for idx, (section_type, section_content) in enumerate(sections):
            section_lines = section_content.strip().split("\n")
            section_dict = {line.split("=")[0].strip(): line.split("=")[1].strip()
                            for line in section_lines if "=" in line}

            # net section.
            if section_type == "net":
                net_items = ["batch", "subdivisions", "width", "height", "saturation", "exposure", "hue",
                            "max_batches", "flip", "mosaic", "letter_box", "cutmix", "mosaic_bound",
                            "mosaic_scale", "mosaic_center", "mosaic_crop", "mosaic_flip", "steps",
                            "scales", "classes"]

                for item in net_items:
                    if item in section_dict:
                        value_without_comment = section_dict[item].split('#')[0].strip()
                        if not value_without_comment:
                            continue

                        row_count = self.cfg_table.rowCount()
                        self.cfg_table.insertRow(row_count)
                        self.cfg_table.setItem(row_count, 0, QtWidgets.QTableWidgetItem(f"{item}_0"))
                        self.cfg_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(value_without_comment))

                        # Highlight common training knobs.
                        if item in {"batch", "width", "height", "subdivisions", "max_batches"}:
                            self.cfg_table.item(row_count, 0).setBackground(QtGui.QColor(255, 255, 144))
                            self.cfg_table.item(row_count, 1).setBackground(QtGui.QColor(255, 255, 144))

                self.net_dict = section_dict

            # convolutional section.
            elif section_type == "convolutional":
                is_before_yolo = idx < len(sections) - 1 and sections[idx + 1][0] == "yolo"
                conv_items = ["filters", "activation"] if is_before_yolo else ["activation"]

                for item in conv_items:
                    if item in section_dict and (is_before_yolo or item != "filters"):
                        row_count = self.cfg_table.rowCount()
                        self.cfg_table.insertRow(row_count)
                        self.cfg_table.setItem(row_count, 0, QtWidgets.QTableWidgetItem(f"{item}_{idx}"))

                        if item == "activation":
                            activation_combo = QtWidgets.QComboBox()
                            activation_combo.addItems(["leaky", "mish", "swish", "linear"])
                            activation_combo.setCurrentText(section_dict[item])
                            self.cfg_table.setCellWidget(row_count, 1, activation_combo)
                            self.activation_row_count[activation_count] = row_count
                            activation_count += 1
                        else:
                            self.cfg_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(section_dict[item]))

                        if item == "filters":
                            self.cfg_table.item(row_count, 0).setBackground(QtGui.QColor(255, 255, 144))
                            self.cfg_table.item(row_count, 1).setBackground(QtGui.QColor(255, 255, 144))

            # yolo section.
            elif section_type == "yolo":
                yolo_items = ["mask", "anchors", "num", "classes", "ignore_thresh", "random"]

                for item in yolo_items:
                    if item in section_dict:
                        if item == "num":
                            num_clusters = int(section_dict[item])
                            self.clusters_spinbox.setValue(num_clusters)

                        row_count = self.cfg_table.rowCount()
                        self.cfg_table.insertRow(row_count)
                        self.cfg_table.setItem(row_count, 0, QtWidgets.QTableWidgetItem(f"{item}_{idx}"))

                        if item == "anchors" and self.imported_anchors is not None:
                            self.cfg_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(
                                ','.join([f"{x},{y}" for x, y in self.imported_anchors])))
                        else:
                            self.cfg_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(section_dict[item]))

                        # Highlight the 'classes' parameter
                        if item == "classes":
                            self.cfg_table.item(row_count, 0).setBackground(QtGui.QColor(255, 255, 144))
                            self.cfg_table.item(row_count, 1).setBackground(QtGui.QColor(255, 255, 144))

                yolo_section_idx += 1

        self.cfg_table.resizeColumnsToContents()
        self.cfg_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

    except Exception as e:
        QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred while parsing the CFG file: {str(e)}")


def cfg_save_clicked(self):
    if self.filename:
        table_data = {}
        for row in range(self.cfg_table.rowCount()):
            param = self.cfg_table.item(row, 0).text()
            cell_widget = self.cfg_table.cellWidget(row, 1)
            value = cell_widget.currentText() if isinstance(cell_widget, QtWidgets.QComboBox) else self.cfg_table.item(row, 1).text()
            table_data[param] = value

        new_config = ""
        section_idx = -1
        previous_line_empty = False

        with open(self.filename, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            stripped_line = line.strip()

            # Start of a new section
            if stripped_line.startswith("["):
                section_idx += 1

                # Ensure exactly one empty line before a new section
                if not previous_line_empty:
                    new_config += "\n"

                new_config += f"{stripped_line}\n"
                previous_line_empty = False
                continue

            # Parameter line
            elif "=" in stripped_line and not stripped_line.startswith("#"):
                param, old_value = stripped_line.split("=", 1)
                param = param.strip()
                new_value = table_data.get(f"{param}_{section_idx}", old_value.strip())
                new_config += f"{param}={new_value}\n"
                previous_line_empty = False
                continue

            # Comment or other lines (preserved as is, except for excess whitespace)
            elif stripped_line.startswith("#"):
                new_config += f"{stripped_line}\n"
                previous_line_empty = False
                continue

            # Empty lines
            if stripped_line == "":
                if not previous_line_empty:  # Allow a single empty line
                    new_config += "\n"
                    previous_line_empty = True
            else:
                previous_line_empty = False

        # Trim leading/trailing whitespace/newlines
        new_config = new_config.strip() + "\n"

        # Save updated config
        options = QtWidgets.QFileDialog.Options()
        save_file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Config File As", self.dialog_start_directory("model", getattr(self, "filename", "")),
            "Config Files (*.cfg);;All Files (*)", options=options)

        if save_file_name:
            self.remember_dialog_selection("model", save_file_name)
            if not save_file_name.endswith('.cfg'):
                save_file_name += '.cfg'

            try:
                with open(save_file_name, 'w', encoding='utf-8') as f:
                    f.write(new_config)

                QtWidgets.QMessageBox.information(self, "Success", "Configuration file saved successfully.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))


def _compact_darknet_options_panel(self):
    if getattr(self, "_darknet_options_panel_compacted", False):
        return

    group = getattr(self, "groupBox_12", None)
    layout = self._prepare_group_grid(group)
    if layout is None:
        return

    names = (
        "show_chart",
        "clear_check",
        "cache",
        "cache_input",
        "label_27",
        "gpu1",
        "gpu2",
        "map_check",
        "random",
    )
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)

    widgets["show_chart"].setText("Chart")
    widgets["clear_check"].setText("Clear Weights")
    widgets["random"].setText("Random Size")
    widgets["gpu1"].setText("2 GPUs")
    widgets["gpu2"].setText("3 GPUs")
    widgets["map_check"].setText("No mAP")
    widgets["cache"].setText("Cache")
    widgets["label_27"].setText("Limit")
    widgets["cache_input"].setMaximumWidth(86)

    row_one = self._row_layout(12)
    for name in ("show_chart", "random"):
        row_one.addWidget(widgets[name])
    row_one.addStretch(1)

    row_two = self._row_layout(12)
    row_two.addWidget(widgets["clear_check"])
    row_two.addStretch(1)

    row_three = self._row_layout(12)
    row_three.addWidget(widgets["cache"])
    row_three.addWidget(widgets["label_27"])
    row_three.addWidget(widgets["cache_input"])
    row_three.addWidget(widgets["map_check"])
    row_three.addStretch(1)

    gpu_row = self._row_layout(12)
    gpu_row.addWidget(widgets["gpu1"])
    gpu_row.addWidget(widgets["gpu2"])
    gpu_row.addStretch(1)

    layout.addLayout(row_one, 0, 0)
    layout.addLayout(row_two, 1, 0)
    layout.addLayout(row_three, 2, 0)
    layout.addLayout(gpu_row, 3, 0)

    self._darknet_options_panel_compacted = True


def _compact_darknet_train_panel(self):
    if getattr(self, "_darknet_train_panel_compacted", False):
        return

    group = getattr(self, "groupBox_11", None)
    layout = self._prepare_group_grid(group)
    if layout is None:
        return

    names = (
        "browse_weights",
        "weights_label",
        "browse_data",
        "data_label",
        "browse_cfg",
        "cfg_label",
        "train_button",
    )
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)

    for button_name in ("browse_weights", "browse_data", "browse_cfg"):
        widgets[button_name].setMinimumWidth(92)
        widgets[button_name].setMaximumWidth(118)

    for row, (button_name, field_name) in enumerate(
        (("browse_weights", "weights_label"), ("browse_data", "data_label"), ("browse_cfg", "cfg_label"))
    ):
        line = self._row_layout()
        line.addWidget(widgets[button_name])
        line.addWidget(widgets[field_name], 1)
        layout.addLayout(line, row, 0)
    layout.addWidget(widgets["train_button"], 3, 0)

    self._darknet_train_panel_compacted = True


def _compact_advanced_panels(self):
    self._compact_cfg_editor_panel()
    self._compact_anchor_panel()
    self._group_cfg_editor_contents()


def _group_cfg_editor_contents(self):
    """Keep the working editor inside its category, not as an empty panel."""
    if getattr(self, "_cfg_editor_contents_grouped", False):
        return
    table = getattr(self, "cfg_table", None)
    group = getattr(self, "groupBox_18", None)
    if table is None or group is None or group.layout() is None:
        return
    for widget in (table, self.hide_activation_checkbox, self.btn_save_file):
        parent = widget.parentWidget()
        self._take_layout_widget(parent.layout() if parent is not None else None, widget)
    table.setMinimumHeight(220)
    table.setMaximumHeight(480)
    table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    table.setFrameShape(QtWidgets.QFrame.NoFrame)
    table.setToolTip("Parameters from the loaded CFG or YAML file. Edit values here.")
    footer = self._row_layout()
    self.hide_activation_checkbox.setText("Hide Activation Layers")
    self.btn_save_file.setText("Save CFG As...")
    footer.addWidget(self.hide_activation_checkbox)
    footer.addStretch(1)
    footer.addWidget(self.btn_save_file)
    layout = group.layout()
    if isinstance(layout, QtWidgets.QFormLayout):
        layout.addRow(table)
        layout.addRow(footer)
    else:
        layout.addWidget(table, layout.rowCount(), 0)
        layout.addLayout(footer, layout.rowCount(), 0)
    model = table.model()
    model.rowsInserted.connect(self._sync_cfg_editor_visibility)
    model.rowsRemoved.connect(self._sync_cfg_editor_visibility)
    model.modelReset.connect(self._sync_cfg_editor_visibility)
    self._sync_cfg_editor_visibility()
    container = getattr(self, "frame_19", None)
    if container is not None:
        container.setFrameShape(QtWidgets.QFrame.NoFrame)
        container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        container.layout().setContentsMargins(0, 0, 0, 0)
        container.layout().setAlignment(Qt.AlignTop)
    page = getattr(self, "tab_2", None)
    if page is not None and page.layout() is not None:
        page.layout().setAlignment(Qt.AlignTop)
    self._cfg_editor_contents_grouped = True


@pyqtSlot()
def _sync_cfg_editor_visibility(self):
    self.cfg_table.setVisible(self.cfg_table.rowCount() > 0)


def _compact_cfg_editor_panel(self):
    if getattr(self, "_cfg_editor_panel_compacted", False):
        return

    group = getattr(self, "groupBox_18", None)
    layout = group.layout() if group is not None else None
    if layout is None:
        return

    names = ("btn_open_file", "cfg_open_label", "import_yaml_button", "default_yaml_path")
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)

    self._reset_layout(layout)
    if isinstance(layout, QtWidgets.QFormLayout):
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)
        cfg_row = self._row_layout()
        cfg_row.addWidget(widgets["btn_open_file"])
        cfg_row.addWidget(widgets["cfg_open_label"], 1)
        yaml_row = self._row_layout()
        yaml_row.addWidget(widgets["import_yaml_button"])
        yaml_row.addWidget(widgets["default_yaml_path"], 1)
        layout.addRow(cfg_row)
        layout.addRow(yaml_row)
    elif isinstance(layout, QtWidgets.QGridLayout):
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)
        cfg_row = self._row_layout()
        cfg_row.addWidget(widgets["btn_open_file"])
        cfg_row.addWidget(widgets["cfg_open_label"], 1)
        yaml_row = self._row_layout()
        yaml_row.addWidget(widgets["import_yaml_button"])
        yaml_row.addWidget(widgets["default_yaml_path"], 1)
        layout.addLayout(cfg_row, 0, 0)
        layout.addLayout(yaml_row, 1, 0)

    self._cfg_editor_panel_compacted = True


def _compact_anchor_panel(self):
    if getattr(self, "_anchor_panel_compacted", False):
        return

    group = getattr(self, "groupBox_13", None)
    layout = group.layout() if group is not None else None
    if layout is None:
        return

    names = (
        "import_data_button",
        "label_23",
        "clusters_spinbox",
        "width_spinbox",
        "height_spinbox",
        "show_checkbox",
        "calculate_anchors_button",
    )
    widgets = {name: getattr(self, name, None) for name in names}
    if any(widget is None for widget in widgets.values()):
        return

    for widget in widgets.values():
        self._take_layout_widget(layout, widget)
        self._compact_control(widget)

    self._reset_layout(layout)

    panel = QtWidgets.QWidget(group)
    panel.setObjectName("anchor_compact_panel")
    panel.setMinimumWidth(0)
    panel.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
    panel_layout = QtWidgets.QVBoxLayout(panel)
    panel_layout.setContentsMargins(12, 16, 12, 12)
    panel_layout.setSpacing(8)

    widgets["label_23"].setText("Clusters")
    widgets["show_checkbox"].setText("Graph")

    source_row = self._row_layout()
    source_row.addWidget(widgets["import_data_button"], 1)

    size_row = self._row_layout()
    size_row.addWidget(widgets["label_23"])
    size_row.addWidget(widgets["clusters_spinbox"])
    size_row.addWidget(self._workflow_label("W", group, 16))
    size_row.addWidget(widgets["width_spinbox"])
    size_row.addWidget(self._workflow_label("H", group, 16))
    size_row.addWidget(widgets["height_spinbox"])
    size_row.addStretch(1)

    option_row = self._row_layout(12)
    option_row.addWidget(widgets["show_checkbox"])
    option_row.addStretch(1)

    action_row = self._row_layout()
    action_row.addWidget(widgets["calculate_anchors_button"], 1)

    panel_layout.addLayout(source_row)
    panel_layout.addLayout(size_row)
    panel_layout.addLayout(option_row)
    panel_layout.addLayout(action_row)

    if isinstance(layout, QtWidgets.QGridLayout):
        layout.addWidget(panel, 0, 0)
    elif isinstance(layout, QtWidgets.QBoxLayout):
        layout.addWidget(panel)
    else:
        layout.addWidget(panel)

    group.setMinimumWidth(0)
    group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    self._anchor_panel_compacted = True

