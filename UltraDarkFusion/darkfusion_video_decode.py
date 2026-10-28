"""Optional NVDEC decoding with verified BGR output and native seek fallback."""
import ctypes
import math
import os
import shutil
import subprocess
import threading

import cv2
import numpy as np


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
        if os.name != 'nt' or not shutil.which('ffmpeg'):
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
        return CudaVideoCapture(capture, source, codec, shutil.which('ffmpeg'))
    except (OSError, ValueError, cv2.error):
        return capture


class CudaVideoCapture:
    """Own one hidden decoder process; never retain a model or Qt object.

    A restart/seek compares its first decoded frame with the original capture.
    Any mismatch, unsupported codec/driver, partial frame or subprocess failure
    resumes the native capture at the next logical frame. File labels and
    thumbnails never consume this decoder's buffers.
    """

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
        self._fallback = False
        self._closed = False
        self.hardware_frames = 0

    def __getattr__(self, name):
        return getattr(self._capture, name)

    def _start(self):
        # Two filter workers outperformed both one and larger pools on the
        # validated AV1 path; the raw-video output encoder stays single-threaded.
        args = [self._executable,'-hide_banner','-loglevel','error','-nostdin','-filter_threads','2',
                '-hwaccel','cuda','-c:v',self._codec]
        if self._position > 0:
            args += ['-ss',format(self._position/self._fps,'.12f')]
        args += ['-i',self._source,'-an','-sn','-fps_mode','passthrough',
                 '-vf','format=yuv420p,scale=in_color_matrix=bt601:out_color_matrix=bt601:flags=bilinear',
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
        result = self._capture.set(prop,value)
        if result and prop == cv2.CAP_PROP_POS_FRAMES:
            self._stop_process()
            self._position = self._capture.get(prop)
            self._verify_next = True
            # Scrubbing/stepping must not pay a fresh CUDA-process startup for
            # each frame. The already-open software decoder has the exact seek.
            self._fallback = True
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
