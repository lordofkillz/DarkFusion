"""One-frame decode lookahead for a local FFmpeg capture; no Qt/model objects."""

import cv2
import logging
import math
import numpy as np
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor


class _CameraExposureControl:
    """One DirectShow camera control, used and restored on its capture thread."""

    def __init__(self, device_index):
        import ctypes as C
        import uuid
        self._original = None
        self._initialized = False
        self._camera = C.c_void_p()
        self._C = C
        self._ole = C.OleDLL('ole32')

        class GUID(C.Structure):
            _fields_ = [('a', C.c_uint32), ('b', C.c_uint16), ('c', C.c_uint16), ('d', C.c_ubyte * 8)]
            def __init__(self, value):
                super().__init__()
                C.memmove(C.byref(self), uuid.UUID(value).bytes_le, 16)

        self._ole.CoCreateInstance.argtypes = [C.POINTER(GUID), C.c_void_p, C.c_ulong, C.POINTER(GUID), C.POINTER(C.c_void_p)]
        self._ole.CreateBindCtx.argtypes = [C.c_ulong, C.POINTER(C.c_void_p)]
        self._ole.CoInitializeEx.argtypes = [C.c_void_p, C.c_ulong]
        self._ole.CoInitializeEx.restype = C.c_long
        devices, enumerator, context = C.c_void_p(), C.c_void_p(), C.c_void_p()
        try:
            result = self._ole.CoInitializeEx(None, 2)
            self._initialized = result >= 0
            if result < 0 and (result & 0xffffffff) != 0x80010106:
                raise OSError('Cannot initialize camera controls')
            self._ole.CoCreateInstance(C.byref(GUID('62BE5D10-60EB-11D0-BD3B-00A0C911CE86')), None, 1,
                C.byref(GUID('29840822-5B84-11D0-BD3B-00A0C911CE86')), C.byref(devices))
            self._call(devices, 3, [C.POINTER(GUID), C.POINTER(C.c_void_p), C.c_ulong],
                C.byref(GUID('860BB310-5D01-11D0-BD3B-00A0C911CE86')), C.byref(enumerator), 0)
            if not enumerator:
                raise OSError('No DirectShow cameras')
            self._ole.CreateBindCtx(0, C.byref(context))
            for index in range(int(device_index) + 1):
                moniker, fetched = C.c_void_p(), C.c_ulong()
                if self._call(enumerator, 3, [C.c_ulong, C.POINTER(C.c_void_p), C.POINTER(C.c_ulong)],
                              1, C.byref(moniker), C.byref(fetched)) != 0:
                    raise OSError('Selected camera is no longer connected')
                try:
                    if index != int(device_index):
                        continue
                    filter_ = C.c_void_p()
                    try:
                        self._call(moniker, 8, [C.c_void_p, C.c_void_p, C.POINTER(GUID), C.POINTER(C.c_void_p)],
                            context, None, C.byref(GUID('56A86895-0AD4-11CE-B03A-0020AF0BA770')), C.byref(filter_))
                        self._call(filter_, 0, [C.POINTER(GUID), C.POINTER(C.c_void_p)],
                            C.byref(GUID('C6E13370-30AC-11D0-A18C-00A0C9118956')), C.byref(self._camera))
                    finally:
                        self._release(filter_)
                finally:
                    self._release(moniker)
            if not self._camera:
                raise OSError('Camera has no exposure control')
        except BaseException:
            for interface in (context, enumerator, devices):
                self._release(interface)
            context = enumerator = devices = C.c_void_p()
            self.close()
            raise
        finally:
            for interface in (context, enumerator, devices):
                self._release(interface)

    def _call(self, interface, index, args, *values):
        C = self._C
        if not interface:
            raise OSError('Camera control interface unavailable')
        table = C.cast(interface, C.POINTER(C.POINTER(C.c_void_p))).contents
        result = C.WINFUNCTYPE(C.c_long, C.c_void_p, *args)(table[index])(interface, *values)
        if result < 0:
            raise OSError(f'Camera control error 0x{result & 0xffffffff:08x}')
        return result

    def _release(self, interface):
        if interface:
            self._call(interface, 2, [])

    def limit_exposure(self, fps):
        C = self._C
        limits = [C.c_long() for _ in range(5)]
        self._call(self._camera, 3, [C.c_long, *[C.POINTER(C.c_long)] * 5],
                   4, *[C.byref(value) for value in limits])
        minimum, maximum, step, _, supported = [value.value for value in limits]
        if not supported & 2 or step <= 0 or not math.isfinite(fps) or fps <= 0:
            raise OSError('Camera does not support frame-rate-priority exposure')
        value, flags = C.c_long(), C.c_long()
        self._call(self._camera, 5, [C.c_long, C.POINTER(C.c_long), C.POINTER(C.c_long)],
                   4, C.byref(value), C.byref(flags))
        # DirectShow exposure is log2(seconds). Select a supported step short
        # enough for the requested rate, retaining an already faster manual value.
        target = min(maximum, math.floor(math.log2(1 / fps)))
        target = max(minimum, minimum + math.floor((target - minimum) / step) * step)
        if flags.value == 2:
            target = min(value.value, target)
        self._original = (value.value, flags.value)
        self._call(self._camera, 4, [C.c_long, C.c_long, C.c_long], 4, target, 2)

    def close(self):
        C = self._C
        try:
            if self._camera and self._original is not None:
                self._call(self._camera, 4, [C.c_long, C.c_long, C.c_long], 4, *self._original)
        finally:
            self._original = None
            try:
                self._release(self._camera)
            finally:
                self._camera = C.c_void_p()
                if self._initialized:
                    self._ole.CoUninitialize()
                    self._initialized = False


def camera_exposure_priority(device_index, fps, enabled=False):
    """Opt in only; leave unsupported devices and default auto exposure alone."""
    if not enabled or os.name != 'nt':
        return None
    control = None
    try:
        control = _CameraExposureControl(device_index)
        control.limit_exposure(float(fps))
        return control
    except (OSError, ValueError, TypeError):
        if control is not None:
            try:
                control.close()
            except OSError:
                logging.getLogger('UltraDarkFusionLogger').warning('Could not restore camera exposure after setup failed.')
        logging.getLogger('UltraDarkFusionLogger').warning('Camera FPS priority is unavailable for device %s.', device_index)
        return None


def negotiated_capture_properties(device_index, properties):
    """Select a confirmed MJPEG rate at the existing resolution, off the GUI."""
    if os.name != 'nt' or not properties or properties.get(cv2.CAP_PROP_FOURCC) != cv2.VideoWriter_fourcc(*'MJPG'):
        return properties
    from darkfusion_media_tools import bundle_directory
    bundled = os.path.join(bundle_directory(), 'ffmpeg', 'ffmpeg.exe')
    executable = bundled if os.path.isfile(bundled) else shutil.which('ffmpeg')
    if not executable:
        return properties
    try:
        def query(args):
            result = subprocess.run([executable,'-hide_banner','-f','dshow',*args],
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                timeout=4,creationflags=subprocess.CREATE_NO_WINDOW)
            return result.stdout.decode('utf-8',errors='replace')
        devices = re.findall(r'"([^"\r\n]+)"\s+\((?:video|none)\)',query(['-list_devices','true','-i','dummy']))
        index = int(device_index)
        if not 0 <= index < len(devices):
            return properties
        text = query(['-list_options','true','-i','video='+devices[index]])
        width = int(properties.get(cv2.CAP_PROP_FRAME_WIDTH,0))
        height = int(properties.get(cv2.CAP_PROP_FRAME_HEIGHT,0))
        rates=[]
        for w,h,mw,mh,rate in re.findall(
                r'vcodec=mjpeg\s+min s=(\d+)x(\d+) fps=[\d.]+ max s=(\d+)x(\d+) fps=([\d.]+)',text):
            if (int(w),int(h),int(mw),int(mh)) == (width,height,width,height):
                rates.append(float(rate))
        selected_fourcc=properties[cv2.CAP_PROP_FOURCC]
        if not rates:
            # Capture cards often offer raw YUY2 rather than MJPEG. Keep the
            # requested resolution and use an advertised 8-bit input format.
            profiles=re.findall(r'pixel_format=(\w+)\s+min s=(\d+)x(\d+) fps=[\d.]+ max s=(\d+)x(\d+) fps=([\d.]+)',text)
            for pixel_format,fourcc in (('yuyv422','YUY2'),('uyvy422','UYVY'),('nv12','NV12')):
                rates=[float(rate) for fmt,w,h,mw,mh,rate in profiles
                       if fmt==pixel_format and (int(w),int(h),int(mw),int(mh))==(width,height,width,height)]
                if rates:
                    selected_fourcc=cv2.VideoWriter_fourcc(*fourcc)
                    break
        if not rates:
            return properties
        rate = min(240.,max(rates))
        if abs(rate-round(rate)) < .01:
            rate=float(round(rate))
        result = {prop:value for prop,value in properties.items() if prop != cv2.CAP_PROP_FPS}
        result[cv2.CAP_PROP_FOURCC]=selected_fourcc
        # DirectShow negotiates the format first, then its supported frame rate.
        result[cv2.CAP_PROP_FPS]=rate
        return result
    except (OSError,ValueError,subprocess.TimeoutExpired):
        return properties


def optimize_video_capture(capture, source, *, live_stream=False, capture_properties=None):
    """Use verified hardware decoding for finite FFmpeg files/HTTP video."""
    if live_stream or capture_properties:
        return capture
    try:
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if capture.getBackendName() != 'FFMPEG' or not (0 < width*height*3 <= PrefetchedVideoCapture.MAX_PREFETCH_BYTES):
            return capture
        from darkfusion_video_decode import hardware_capture
        accelerated = hardware_capture(capture, source)
        if accelerated is not capture:
            return PrefetchedVideoCapture(accelerated)
    except (AttributeError, cv2.error):
        pass
    return prefetch_local_capture(capture, source)


def prefetch_local_capture(capture, source, *, live_stream=False, capture_properties=None):
    """Overlap file decoding without changing camera/network/hardware captures."""
    if live_stream or capture_properties or not isinstance(source, (str, os.PathLike)):
        return capture
    if not os.path.isfile(source):
        return capture
    try:
        if capture.getBackendName() != 'FFMPEG':
            return capture
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not (width > 0 and height > 0 and width * height * 3 <= PrefetchedVideoCapture.MAX_PREFETCH_BYTES):
            return capture
    except (AttributeError, cv2.error):
        return capture
    return PrefetchedVideoCapture(capture)


class PrefetchedVideoCapture:
    """Keep logical frame position independent of the decoder's lookahead.

    All reads/seeks/release are serialized on one decoder thread. A seek drains
    and discards an old pending frame only after the seek succeeds. At most one
    decoded frame is held ahead, and large frames use the synchronous schedule.
    """

    MAX_PREFETCH_BYTES = 32 * 1024 * 1024

    def __init__(self, capture):
        self._capture = capture
        self._position = capture.get(cv2.CAP_PROP_POS_FRAMES)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='video-decode')
        self._pending = None
        self._closed = False
        self.last_read_finished = None

    def __getattr__(self, name):
        return getattr(self._capture, name)

    def _read(self):
        ok, image = self._capture.read()
        position = self._capture.get(cv2.CAP_PROP_POS_FRAMES)
        return ok, image, position, time.time()

    def read(self):
        pending = self._pending
        self._pending = None
        if pending is None:
            pending = self._executor.submit(self._read)
        ok, image, self._position, self.last_read_finished = pending.result()
        if ok and image.nbytes <= self.MAX_PREFETCH_BYTES:
            self._pending = self._executor.submit(self._read)
        return ok, image

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self._position
        return self._executor.submit(self._capture.get, prop).result()

    def set(self, prop, value):
        pending = self._pending
        self._pending = None
        def update():
            result = self._capture.set(prop, value)
            return result, self._capture.get(cv2.CAP_PROP_POS_FRAMES)
        result, position = self._executor.submit(update).result()
        if result and prop == cv2.CAP_PROP_POS_FRAMES:
            self._position = position
        else:
            # A failed seek must still return the next logical frame, even if
            # the native decoder has already advanced one frame ahead.
            self._pending = pending
        return result

    def release(self):
        if self._closed:
            return
        self._closed = True
        interrupt = getattr(self._capture, 'interrupt', None)
        if interrupt is not None:
            interrupt()
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        try:
            self._executor.submit(self._capture.release).result()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def interrupt(self):
        interrupt=getattr(self._capture,'interrupt',None)
        if interrupt is not None:
            interrupt()


def video_playback_quality_key(value, default="smooth"):
    """Normalize persisted video-player scaling quality names."""
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fast": "fast",
        "performance": "fast",
        "nearest": "fast",
        "smooth": "smooth",
        "balanced": "smooth",
        "antialiased": "smooth",
        "anti_aliased": "smooth",
        "high": "high",
        "high_quality": "high",
        "quality": "high",
        "lanczos": "high",
    }
    return aliases.get(key, default)


def video_playback_renderer_key(value, default="auto"):
    """Normalize persisted video-player renderer names."""
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "auto": "auto",
        "automatic": "auto",
        "recommended": "auto",
        "gpu": "gpu",
        "hardware": "gpu",
        "hardware_accelerated": "gpu",
        "opengl": "gpu",
        "open_gl": "gpu",
        "cpu": "cpu",
        "software": "cpu",
        "compatibility": "cpu",
    }
    return aliases.get(key, default)


def get_video_codec_fourcc(codec_name=None):
    """Get OpenCV VideoWriter FOURCC code from codec name preference.

    Args:
        codec_name: str like "mp4v (standard)", "H264 (better compression)", "MJPG (high quality)"

    Returns:
        tuple of 4 characters for cv2.VideoWriter_fourcc()
    """
    if codec_name and "H264" in codec_name:
        return ("H", "2", "6", "4")
    elif codec_name and "MJPG" in codec_name:
        return ("M", "J", "P", "G")
    else:  # Default to mp4v
        return ("m", "p", "4", "v")


def get_video_output_fps(fps_value=None):
    """Get output FPS for video generation. Default: 45 FPS.

    Args:
        fps_value: int FPS value (15-120)

    Returns:
        float FPS value
    """
    try:
        fps = float(fps_value or 45)
        return max(15.0, min(120.0, fps))  # Clamp 15-120
    except (TypeError, ValueError):
        return 45.0


def video_resize_policy_key(value, default="smart"):
    key = str(value or default).strip().lower()
    aliases = {
        "auto": "smart",
        "model": "model_exact",
        "exact": "model_exact",
        "stretch": "model_exact",
        "fit": "letterbox",
        "keep_aspect": "letterbox",
        "manual w/h": "manual",
        "manual": "manual",
    }
    key = aliases.get(key, key)
    if key not in {"smart", "model_exact", "letterbox", "manual"}:
        key = default
    return key


def resize_interpolation_for(frame, target_w, target_h):
    try:
        src_h, src_w = frame.shape[:2]
        if target_w < src_w or target_h < src_h:
            return cv2.INTER_AREA
    except Exception:
        pass
    return cv2.INTER_LINEAR


def resize_frame_to_wh(frame, target_w, target_h):
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    try:
        src_h, src_w = frame.shape[:2]
        if src_w == target_w and src_h == target_h:
            return np.ascontiguousarray(frame)
    except Exception:
        pass

    return cv2.resize(
        frame,
        (target_w, target_h),
        interpolation=resize_interpolation_for(frame, target_w, target_h),
    )


def letterbox_frame_to_wh(frame, target_w, target_h, pad_value=114):
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return None

    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = resize_frame_to_wh(frame, new_w, new_h)
    if resized is None or resized.size == 0:
        return None

    if len(frame.shape) == 2:
        canvas = np.full((target_h, target_w), pad_value, dtype=frame.dtype)
    else:
        channels = frame.shape[2]
        canvas = np.full((target_h, target_w, channels), pad_value, dtype=frame.dtype)

    x = max(0, (target_w - new_w) // 2)
    y = max(0, (target_h - new_h) // 2)
    canvas[y:y + new_h, x:x + new_w] = resized
    return np.ascontiguousarray(canvas)


def transform_video_frame_by_settings(frame, settings):
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None

    settings = dict(settings or {})
    transformed = frame
    frame_h, frame_w = transformed.shape[:2]

    try:
        frame_target_w = max(1, int(settings.get("width", frame_w) or frame_w))
        frame_target_h = max(1, int(settings.get("height", frame_h) or frame_h))
    except Exception:
        frame_target_w, frame_target_h = frame_w, frame_h

    # ROI never alters the image; it is only an inference boundary. Cropping the
    # saved/displayed frame is the separate Crop feature below.
    crop_enabled = bool(settings.get("crop_enabled"))

    if crop_enabled:
        crop_w = min(frame_target_w, frame_w)
        crop_h = min(frame_target_h, frame_h)
        start_x = max(0, (frame_w - crop_w) // 2)
        start_y = max(0, (frame_h - crop_h) // 2)
        transformed = transformed[start_y:start_y + crop_h, start_x:start_x + crop_w]

    if transformed is None or not isinstance(transformed, np.ndarray) or transformed.size == 0:
        return None

    if not settings.get("resize_enabled"):
        return np.ascontiguousarray(transformed)

    policy = video_resize_policy_key(settings.get("resize_policy", "smart"))
    model_size_available = bool(settings.get("model_size_available"))

    try:
        network_w = max(1, int(settings.get("network_width", frame_target_w) or frame_target_w))
        network_h = max(1, int(settings.get("network_height", frame_target_h) or frame_target_h))
    except Exception:
        network_w, network_h = frame_target_w, frame_target_h

    if policy == "smart":
        if crop_enabled or not model_size_available:
            policy = "manual"
        else:
            src_h, src_w = transformed.shape[:2]
            src_aspect = src_w / max(1.0, float(src_h))
            model_aspect = network_w / max(1.0, float(network_h))
            policy = "model_exact" if abs(src_aspect - model_aspect) <= 0.15 else "letterbox"

    if policy == "model_exact":
        return resize_frame_to_wh(transformed, network_w, network_h)

    if policy == "letterbox":
        return letterbox_frame_to_wh(transformed, network_w, network_h)

    return resize_frame_to_wh(transformed, frame_target_w, frame_target_h)
