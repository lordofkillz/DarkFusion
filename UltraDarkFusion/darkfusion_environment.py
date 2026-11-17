"""Keep the application and its workers on their installed environment packages."""
import os
import site
import sys
import logging


def configure_opencv_dnn_cache():
    """Prepare persistent OpenCL convolution tuning before the first DNN run.

    OpenCV does not create this directory itself. Respect an explicit path (or
    an empty value disabling the cache), and keep a cache failure non-fatal.
    No OpenCV/model imports or machine-wide environment changes are needed.
    """
    key = 'OPENCV_OCL4DNN_CONFIG_PATH'
    configured = os.environ.get(key)
    if configured is not None and not configured.strip():
        return None
    if configured is None:
        root = (os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CACHE_HOME')
                or os.path.join(os.path.expanduser('~'), '.cache'))
        path = os.path.join(root, 'DarkFusion', 'opencv-dnn')
    else:
        path = os.path.expandvars(os.path.expanduser(configured))
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        logging.getLogger('UltraDarkFusionLogger').warning(
            'OpenCV DNN tuning cache unavailable at %s: %s', path, error)
        return None
    os.environ[key] = path
    return path


def configure_gui_opencv_threads():
    """Give GUI resizing a bounded CPU budget after Ultralytics initialization.

    Ultralytics disables OpenCV parallelism for its loader. The GUI also resizes
    full-resolution frames, which benefits from several workers. Call only in
    the GUI entrypoint, so training and command-line workers keep their settings.
    """
    import cv2

    budget = min(8, max(1, (os.cpu_count() or 1) // 4))
    cv2.setNumThreads(budget)
    return cv2.getNumThreads()


def configure_runtime():
    """Exclude per-user packages even when launched directly from an IDE.

    The normal launcher already uses -s. A direct python app.py launch must not
    silently mix an older user-wide ML package with the reviewed environment.
    This only changes the current process/children, never the user's packages.
    """
    paths = site.getusersitepackages()
    paths = [paths] if isinstance(paths, str) else paths
    roots = [os.path.normcase(os.path.abspath(path)) for path in paths if path]

    def user_package(path):
        if not path:
            return False
        resolved = os.path.normcase(os.path.abspath(path))
        return any(resolved == root or resolved.startswith(root + os.sep) for root in roots)

    sys.path[:] = [path for path in sys.path if not user_package(path)]
    site.ENABLE_USER_SITE = False
    os.environ['PYTHONNOUSERSITE'] = '1'


APP_DIR = os.path.dirname(os.path.abspath(__file__))


APP_SETTINGS_FILE = "settings.json"


APP_SETTINGS_PATH = os.environ.get(
    "DARKFUSION_SETTINGS_PATH", os.path.join(APP_DIR, APP_SETTINGS_FILE))


def model_directory(default=None):
    """Use the persistent pip model store; retain existing source/EXE paths."""
    configured = os.environ.get("DARKFUSION_MODEL_DIR")
    if configured:
        return os.path.abspath(os.path.expandvars(os.path.expanduser(configured)))
    return os.fspath(default) if default is not None else os.path.join(APP_DIR, "Sam")
