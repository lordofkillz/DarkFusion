"""Keep the application and its workers on their installed environment packages."""
import os
import site
import sys


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
