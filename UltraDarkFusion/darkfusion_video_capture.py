"""One-frame decode lookahead for a local FFmpeg capture; no Qt/model objects."""
from concurrent.futures import ThreadPoolExecutor
import os
import re
import shutil
import subprocess
import time

import cv2


def negotiated_capture_properties(device_index, properties):
    """Select a confirmed MJPEG rate at the existing resolution, off the GUI."""
    if os.name != 'nt' or not properties or properties.get(cv2.CAP_PROP_FOURCC) != cv2.VideoWriter_fourcc(*'MJPG'):
        return properties
    executable = shutil.which('ffmpeg')
    if not executable:
        return properties
    try:
        def query(args):
            result = subprocess.run([executable,'-hide_banner','-f','dshow',*args],
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                timeout=1,creationflags=subprocess.CREATE_NO_WINDOW)
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
