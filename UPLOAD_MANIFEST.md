# GitHub upload manifest

This manifest defines the files that make up the distributable
UltraDarkFusion v5.2 repository.

## Commit these files

### Repository setup

- `.env.example`
- `.gitattributes`
- `.gitignore`
- `LICENSE.txt`
- `README.md`
- `MODEL_SETUP.md`
- `UPLOAD_MANIFEST.md`
- `requirements.txt`
- `fusion_install.bat`
- `install.ps1`
- `run_darkfusion.bat`
- `scripts/verify_install.py`
- `samples/` (README media)

### Application

- `UltraDarkFusion/UltraDarkFusion_v5.2.py`
- `UltraDarkFusion/UltraDarkFusion_v5.2.ui`
- `UltraDarkFusion/ui_ultradarkfusion_v5_2.py`
- `UltraDarkFusion/__init__.py`
- `UltraDarkFusion/dinov5_2.py`
- `UltraDarkFusion/sahi_predict_wrapperv5.py`
- `UltraDarkFusion/prediction_size_filter.py`
- `UltraDarkFusion/splash_utils.py`
- `UltraDarkFusion/darkfusion_command_runner.py`
- `UltraDarkFusion/darkfusion_ultralytics_batch.py`
- `UltraDarkFusion/darkfusion_ultralytics_cli.py`
- `UltraDarkFusion/darkfusion_ultralytics_train.py`
- `UltraDarkFusion/darkfusion_ultralytics_tune.py`
- `UltraDarkFusion/darkfusion_validation_review.py`
- `UltraDarkFusion/generate_ui_py.ps1`
- `UltraDarkFusion/botsort.yaml`
- `UltraDarkFusion/zlibwapi.dll` (legacy Darknet runtime support)

### Runtime resources

- `UltraDarkFusion/styles/`
- `UltraDarkFusion/sounds/`
- `UltraDarkFusion/translations/`
- `UltraDarkFusion/documents/`
- `UltraDarkFusion/images/`
- `UltraDarkFusion/Sam/README.md`
- `UltraDarkFusion/Sam/FSRCNN_x4.pb`
- Small tokenizer/configuration files under `UltraDarkFusion/Sam/`

## Keep these outside normal Git history

- SAM3, GroundingDINO, YOLO, Darknet, ONNX, and TensorRT model binaries.
- Datasets, labels, `train.txt`, `valid.txt`, and dataset YAML files.
- Training/tuning/validation run folders and generated reports.
- Videos, exported frames, archives, and caches.
- `.env`, `settings.json`, local databases, thresholds, and machine-specific
  configuration.
- Recovery copies, backups, scratch scripts, and generated debug output.

The `.gitignore` enforces these exclusions. See `MODEL_SETUP.md` for the model
files that users install separately after cloning.
