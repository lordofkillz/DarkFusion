"""Optional NVDEC decoding with verified BGR output and native seek fallback."""
import ctypes
from collections import OrderedDict
from fractions import Fraction
import json
import math
import os
import shutil
import subprocess
import threading

import cv2
import numpy as np


_CLOCK_CACHE = OrderedDict()
_CLOCK_LOCK = threading.Lock()


def _remember_frame_clock(key, value):
    with _CLOCK_LOCK:
        _CLOCK_CACHE[key] = value
        _CLOCK_CACHE.move_to_end(key)
        while len(_CLOCK_CACHE) > 32:
            _CLOCK_CACHE.popitem(last=False)
    return value


def _clock_probe_command(source, executable):
    return [os.path.join(os.path.dirname(executable), 'ffprobe.exe'),
        '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,r_frame_rate,time_base,start_pts,nb_frames:packet=pts',
        '-of', 'json', source]


def _clock_metadata_is_constant(data, count, fps):
    stream, = data['streams']
    rate = Fraction(stream['avg_frame_rate'])
    tick = Fraction(stream['time_base'])
    if (rate <= 0 or tick <= 0 or rate != Fraction(stream['r_frame_rate'])
            or int(stream['start_pts']) != 0 or int(stream['nb_frames']) != count
            or not math.isclose(float(rate), fps, rel_tol=1e-8)):
        return False
    step = 1 / (rate * tick)
    if step.denominator != 1:
        return False
    timestamps = sorted(int(packet['pts']) for packet in data['packets'])
    return len(timestamps) == count and all(value == index * step.numerator
        for index, value in enumerate(timestamps))


def _constant_frame_clock(source, size, modified_ns, executable, count, fps):
    """Validate packet timestamps without decoding or writing the source video."""
    key = (source, size, modified_ns, executable, count, fps)
    with _CLOCK_LOCK:
        if key in _CLOCK_CACHE:
            value = _CLOCK_CACHE[key]
            _CLOCK_CACHE.move_to_end(key)
            return value
    if count <= 0 or count > 1_000_000:
        return False
    command = _clock_probe_command(source, executable)
    if not os.path.isfile(command[0]):
        return False
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=10, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return _remember_frame_clock(key, _clock_metadata_is_constant(json.loads(result.stdout), count, fps))
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError):
        return _remember_frame_clock(key, False)


def _clear_frame_clocks():
    with _CLOCK_LOCK:
        _CLOCK_CACHE.clear()


_constant_frame_clock.cache_clear = _clear_frame_clocks


class _FrameClockProbe:
    """Prevalidate a local file while it plays; cancellation owns its subprocess."""

    def __init__(self, key):
        self.key = key
        self._lock = threading.Lock()
        self._process = None
        self._cancelled = False
        self._done = threading.Event()
        self._result = False
        threading.Thread(target=self._run, name='video-frame-clock', daemon=True).start()

    def _run(self):
        process = None
        try:
            source, _, _, executable, count, fps = self.key
            with self._lock:
                if self._cancelled:
                    return
                process = subprocess.Popen(_clock_probe_command(source, executable),
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self._process = process
            output, _ = process.communicate(timeout=10)
            result = process.returncode == 0 and _clock_metadata_is_constant(json.loads(output), count, fps)
            with self._lock:
                if not self._cancelled:
                    self._result = _remember_frame_clock(self.key, result)
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError):
            pass  # An unverified clock retains native exact seeking.
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
            with self._lock:
                self._process = None
            self._done.set()

    def result(self):
        self._done.wait(11)
        return self._result

    def close(self):
        with self._lock:
            self._cancelled = True
            if self._process is not None and self._process.poll() is None:
                try:
                    self._process.terminate()
                except OSError:
                    pass  # A completed probe may race capture shutdown.
        self._done.set()


def _popen_with_video_pipe(args, pipe_bytes):
    # The default Windows anonymous pipe holds only a few KiB. That forces
    # hundreds of Python/GIL handoffs for each full-HD frame. A bounded pipe
    # lets FFmpeg finish writing a frame while the GUI works on its predecessor.
    import _winapi
    import msvcrt
    read_handle,write_handle = _winapi.CreatePipe(None,pipe_bytes)
    reader = os.fdopen(msvcrt.open_osfhandle(read_handle,os.O_RDONLY|os.O_BINARY),'rb',buffering=1024*1024)
    writer = os.fdopen(msvcrt.open_osfhandle(write_handle,os.O_WRONLY|os.O_BINARY),'wb',buffering=0)
    try:
        with writer:
            process = subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=writer,
                stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        process.stdout=reader
        return process
    except BaseException:
        reader.close()
        raise


def hardware_capture(capture, source):
    """Only finite, 8-bit planar video; unsupported inputs keep their capture."""
    try:
        if os.name != 'nt':
            return capture
        from darkfusion_media_tools import bundle_directory
        bundled = os.path.join(bundle_directory(), 'ffmpeg', 'ffmpeg.exe')
        executable = bundled if os.path.isfile(bundled) else shutil.which('ffmpeg')
        if not executable:
            return capture
        if not isinstance(source, str) or not (os.path.isfile(source) or source.startswith(('http://','https://'))):
            return capture
        if (capture.get(cv2.CAP_PROP_FRAME_COUNT) <= 0
                or capture.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT) != int.from_bytes(b'I420','little')):
            return capture
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        name = ''.join(chr((fourcc >> (8*i)) & 255) for i in range(4)).lower()
        # AV1 is the validated win over software decoding here. Keep inexpensive
        # codecs on their existing path rather than adding a pipe/process cost.
        codec = {'av01':'av1_cuvid'}.get(name)
        if codec is None:
            return capture
        # Loading a driver is cheap; a missing/unsupported FFmpeg decoder still
        # falls back on its first read without changing the returned frame.
        ctypes.WinDLL('nvcuda.dll')
        return CudaVideoCapture(capture, source, codec, executable)
    except (OSError, ValueError, cv2.error):
        return capture


class CudaVideoCapture:
    """Own one hidden decoder process; never retain a model or Qt object.

    The first hardware frame is compared with the original capture. Seeks also
    verify pixels unless the complete local-file timestamp clock was validated.
    Any mismatch, unsupported codec/driver, partial frame or subprocess failure
    resumes the native capture at the next logical frame. File labels and
    thumbnails never consume this decoder's buffers.
    """

    # read() allocates a new array for each hardware frame; the native fallback
    # also uses VideoCapture.read() without a caller-supplied reusable array.
    _darkfusion_owns_frame_buffers = True

    def __init__(self, capture, source, codec, executable):
        self._capture, self._source, self._codec, self._executable = capture, source, codec, executable
        self._position = capture.get(cv2.CAP_PROP_POS_FRAMES)
        self._fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(self._fps) or self._fps <= 0:
            raise ValueError('A finite frame rate is required for verified seeking')
        self._width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._process = None
        self._process_lock = threading.Lock()
        self._stopping = threading.Event()
        self._verify_next = True
        self._pixels_verified = False
        self._fallback = False
        self._read_seek_frame = False
        self._closed = False
        self.hardware_frames = 0
        self._clock_probe = None
        self._begin_clock_probe()

    def _begin_clock_probe(self):
        if self._clock_probe is not None or not 0 < self._count <= 1_000_000:
            return
        try:
            if not os.path.isfile(self._source):
                return
            stat = os.stat(self._source)
            key = (self._source, stat.st_size, stat.st_mtime_ns, self._executable, self._count, self._fps)
            with _CLOCK_LOCK:
                cached = key in _CLOCK_CACHE
            if not cached and os.path.isfile(_clock_probe_command(self._source, self._executable)[0]):
                self._clock_probe = _FrameClockProbe(key)
        except OSError:
            pass  # The normal decoder/fallback handles a disappearing source.

    def __getattr__(self, name):
        return getattr(self._capture, name)

    def _start(self):
        # Two filter workers outperformed both one and larger pools on the
        # validated AV1 path; the raw-video output encoder stays single-threaded.
        args = [self._executable,'-hide_banner','-loglevel','error','-nostdin','-filter_threads','2',
                '-hwaccel','cuda','-hwaccel_output_format','cuda','-c:v',self._codec]
        # Explicit zero seeking also discards negative-timestamp preroll from
        # cut MP4 files, matching OpenCV's logical first frame. CUVID otherwise
        # emits those earlier frames and the pixel guard falls back to CPU.
        args += ['-ss',format(self._position/self._fps,'.12f')]
        args += ['-i',self._source,'-an','-sn','-fps_mode','passthrough',
                 '-vf','hwdownload,format=nv12,format=yuv420p,scale=in_color_matrix=bt601:out_color_matrix=bt601:flags=bilinear',
                 '-pix_fmt','bgr24','-threads:v','1','-f','rawvideo','pipe:1']
        with self._process_lock:
            if self._stopping.is_set():
                return False
            self._process = _popen_with_video_pipe(args,min(16*1024*1024,self._width*self._height*6))
        return True

    def _stop_process(self):
        with self._process_lock:
            process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        finally:
            process.stdout.close()

    def interrupt(self):
        # Terminating our own child also unblocks an outstanding pipe read.
        self._stopping.set()
        if self._clock_probe is not None:
            self._clock_probe.close()
        with self._process_lock:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()

    def _native_read(self):
        ok, image = self._capture.read()
        self._position = self._capture.get(cv2.CAP_PROP_POS_FRAMES)
        return ok, image

    def _resume_native(self):
        target=self._position
        self._capture.set(cv2.CAP_PROP_POS_FRAMES,target)
        position=self._capture.get(cv2.CAP_PROP_POS_FRAMES)
        if position>target:
            if not self._capture.set(cv2.CAP_PROP_POS_FRAMES,0):
                raise OSError('Cannot align the software decoder after hardware failure')
            position=self._capture.get(cv2.CAP_PROP_POS_FRAMES)
        # Some finite HTTP streams cannot seek. Continue their original decoder
        # from its verification frame instead of repeating stale frames.
        while position<target and not self._stopping.is_set():
            ok,_=self._capture.read()
            if not ok:
                return False,None
            advanced=self._capture.get(cv2.CAP_PROP_POS_FRAMES)
            if advanced<=position:
                raise OSError('The software decoder did not advance to the fallback frame')
            position=advanced
        if self._stopping.is_set():
            return False,None
        return self._native_read()

    def read(self):
        if self._stopping.is_set():
            return False, None
        if self._read_seek_frame:
            self._read_seek_frame = False
            return self._native_read()
        if self._fallback:
            return self._native_read()
        try:
            if self._process is None and not self._start():
                return False, None
            image = np.empty((self._height,self._width,3),np.uint8)
            data = memoryview(image).cast('B')
            received = 0
            while received < len(data) and not self._stopping.is_set():
                count = self._process.stdout.readinto(data[received:])
                if not count:
                    break
                received += count
            if self._stopping.is_set():
                return False, None
            if received != len(data):
                if received == 0 and self._position >= self._count:
                    self._stop_process()
                    return False, None
                raise OSError('NVDEC ended or returned an incomplete video frame')
            if self._verify_next:
                ok, expected = self._capture.read()
                self._verify_next = False
                if not ok or expected.shape != image.shape or not np.array_equal(image,expected):
                    self._fallback = True
                    self._stop_process()
                    self._position = self._capture.get(cv2.CAP_PROP_POS_FRAMES)
                    return ok, expected
                self._pixels_verified = True
            self._position += 1
            self.hardware_frames += 1
            return True, image
        except (OSError, ValueError):
            self._stop_process()
            if self._stopping.is_set():
                return False, None
            self._fallback = True
            return self._resume_native()

    def get(self, prop):
        return self._position if prop == cv2.CAP_PROP_POS_FRAMES else self._capture.get(prop)

    def set(self, prop, value):
        if self._stopping.is_set():
            return False
        if (prop == cv2.CAP_PROP_POS_FRAMES and self._process is not None
                and not self._fallback and not self._read_seek_frame
                and value == self._position and not self._stopping.is_set()):
            return True  # Next Frame already points at the next unread GPU frame.
        if (prop == cv2.CAP_PROP_POS_FRAMES and self._pixels_verified and not self._fallback
                and not self._stopping.is_set() and os.path.isfile(self._source)):
            target = float(value)
            if math.isfinite(target) and target.is_integer() and 0 <= target < self._count:
                stat = os.stat(self._source)
                key = (self._source, stat.st_size, stat.st_mtime_ns, self._executable, self._count, self._fps)
                if self._clock_probe is not None and self._clock_probe.key == key:
                    constant = self._clock_probe.result()
                else:
                    constant = _constant_frame_clock(*key)
                if self._stopping.is_set():
                    return False
                if constant:
                    # Accurate input seeking decodes from the preceding keyframe
                    # on NVDEC. Avoid paying for a second full software seek.
                    # First-frame pixel parity and a complete CFR timestamp check
                    # are required; other inputs retain native seeking below.
                    self._stop_process()
                    self._position = int(target)
                    self._verify_next = False
                    self._read_seek_frame = False
                    return True
        result = self._capture.set(prop,value)
        if result and prop == cv2.CAP_PROP_POS_FRAMES:
            self._stop_process()
            self._position = self._capture.get(prop)
            self._verify_next = True
            # Step once with the already-open exact software seek. If playback
            # continues, restart/verify NVDEC at the next logical frame instead
            # of permanently losing GPU decoding after the first timeline jump.
            # An actual decoder failure keeps its permanent fallback intact.
            self._read_seek_frame = True
        return result

    def release(self):
        if self._closed:
            return
        self._closed = True
        self.interrupt()
        try:
            self._stop_process()
        finally:
            self._capture.release()
