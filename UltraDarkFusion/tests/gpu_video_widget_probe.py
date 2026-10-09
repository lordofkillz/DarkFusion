"""Manual, dependency-free Qt OpenGL capability and video-scaling probe.

Run with the Fusion interpreter. The probe never opens a lasting window; it
uses an offscreen surface to inspect the driver, then a hidden QOpenGLWidget
to render one synthetic frame with nearest, bilinear, and bicubic scaling.
"""

from __future__ import annotations

import sys
import ctypes

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


VERTEX_SHADER = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 uv;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    uv = texcoord;
}
"""


FRAGMENT_SHADER = """
uniform sampler2D frameTexture;
uniform vec2 textureSize;
uniform int qualityMode; // 0 nearest, 1 bilinear, 2 bicubic
varying vec2 uv;

float cubic(float x) {
    // Catmull-Rom cubic reconstruction kernel (a = -0.5).
    x = abs(x);
    if (x <= 1.0)
        return 1.5*x*x*x - 2.5*x*x + 1.0;
    if (x < 2.0)
        return -0.5*x*x*x + 2.5*x*x - 4.0*x + 2.0;
    return 0.0;
}

vec4 bicubicSample(vec2 coord) {
    vec2 pixel = coord * textureSize - vec2(0.5);
    vec2 base = floor(pixel);
    vec2 fraction = pixel - base;
    vec4 color = vec4(0.0);
    float total = 0.0;
    for (int row = -1; row <= 2; ++row) {
        for (int col = -1; col <= 2; ++col) {
            vec2 offset = vec2(float(col), float(row));
            float weight = cubic(offset.x - fraction.x)
                         * cubic(fraction.y - offset.y);
            vec2 sampleUv = (base + offset + vec2(0.5)) / textureSize;
            color += texture2D(frameTexture, clamp(sampleUv, vec2(0.0), vec2(1.0))) * weight;
            total += weight;
        }
    }
    return color / max(total, 0.00001);
}

void main() {
    if (qualityMode == 2)
        gl_FragColor = bicubicSample(uv);
    else
        gl_FragColor = texture2D(frameTexture, uv);
}
"""


def _gl_function(name: bytes, restype, *argtypes):
    """Resolve a current-context GL function without a PyOpenGL dependency."""
    address = QtGui.QOpenGLContext.currentContext().getProcAddress(name)
    if not address:
        raise RuntimeError(f"OpenGL function unavailable: {name.decode()}")
    return ctypes.CFUNCTYPE(restype, *argtypes)(int(address))


class GpuVideoWidget(QtWidgets.QOpenGLWidget):
    """Small proof-of-concept RGB video surface using only Qt OpenGL APIs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = None
        self._texture = None
        self._blitter = None
        self._quality = 1
        self._texture_dirty = False

    def set_quality(self, quality: str) -> None:
        self._quality = {"nearest": 0, "bilinear": 1, "high": 2}[quality]
        self._texture_dirty = True  # nearest/bilinear use different filters
        self.update()

    def set_frame(self, rgb_frame: np.ndarray) -> None:
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError("expected an HxWx3 RGB frame")
        self._frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        self._texture_dirty = True
        self.update()

    def initializeGL(self) -> None:
        self._blitter = QtGui.QOpenGLTextureBlitter()
        if not self._blitter.create():
            raise RuntimeError("QOpenGLTextureBlitter creation failed")

    def _upload_texture(self) -> None:
        if self._frame is None or not self._texture_dirty:
            return
        if self._texture is not None:
            self._texture.destroy()
        height, width = self._frame.shape[:2]
        texture = QtGui.QOpenGLTexture(QtGui.QOpenGLTexture.Target2D)
        texture.setFormat(QtGui.QOpenGLTexture.RGB8_UNorm)
        texture.setSize(width, height)
        if self._quality == 2:
            texture.setMipLevels(texture.maximumMipLevels())
        texture.allocateStorage(QtGui.QOpenGLTexture.RGB, QtGui.QOpenGLTexture.UInt8)
        texture.setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)
        if self._quality == 0:
            texture.setMinMagFilters(QtGui.QOpenGLTexture.Nearest, QtGui.QOpenGLTexture.Nearest)
        elif self._quality == 2:
            texture.setMinMagFilters(QtGui.QOpenGLTexture.LinearMipMapLinear, QtGui.QOpenGLTexture.Linear)
        else:
            texture.setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        texture.setData(QtGui.QOpenGLTexture.RGB, QtGui.QOpenGLTexture.UInt8, self._frame)
        if self._quality == 2:
            texture.generateMipMaps()
        self._texture = texture
        self._texture_dirty = False

    def paintGL(self) -> None:
        gl_clear_color = _gl_function(
            b"glClearColor", None, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
        )
        gl_clear = _gl_function(b"glClear", None, ctypes.c_uint)
        gl_clear_color(0.0, 0.0, 0.0, 1.0)
        gl_clear(0x00004000)  # GL_COLOR_BUFFER_BIT
        self._upload_texture()
        if self._texture is None or self._blitter is None:
            return
        target = QtCore.QRectF(0.0, 0.0, float(self.width()), float(self.height()))
        transform = QtGui.QOpenGLTextureBlitter.targetTransform(target, self.rect())
        self._blitter.bind()
        self._blitter.blit(self._texture.textureId(), transform, QtGui.QOpenGLTextureBlitter.OriginTopLeft)
        self._blitter.release()

    def closeEvent(self, event) -> None:
        self.makeCurrent()
        if self._texture is not None:
            self._texture.destroy()
            self._texture = None
        if self._blitter is not None:
            self._blitter.destroy()
            self._blitter = None
        self.doneCurrent()
        super().closeEvent(event)


def probe_context() -> tuple[bool, str]:
    attempts = []
    candidates = []
    default = QtGui.QSurfaceFormat.defaultFormat()
    candidates.append(("default", default))
    for name, renderable, version in (
        ("desktop-2.1", QtGui.QSurfaceFormat.OpenGL, (2, 1)),
        ("desktop-3.3", QtGui.QSurfaceFormat.OpenGL, (3, 3)),
        ("gles-2.0", QtGui.QSurfaceFormat.OpenGLES, (2, 0)),
    ):
        fmt = QtGui.QSurfaceFormat()
        fmt.setRenderableType(renderable)
        fmt.setVersion(*version)
        fmt.setProfile(QtGui.QSurfaceFormat.NoProfile)
        candidates.append((name, fmt))

    for name, fmt in candidates:
        surface = QtGui.QOffscreenSurface()
        surface.setFormat(fmt)
        surface.create()
        context = QtGui.QOpenGLContext()
        context.setFormat(fmt)
        created = context.create()
        current = bool(created and surface.isValid() and context.makeCurrent(surface))
        attempts.append(f"{name}:surface={surface.isValid()},context={created},current={current}")
        if not current:
            continue
        actual = context.format()
        try:
            gl_get_string = _gl_function(b"glGetString", ctypes.c_char_p, ctypes.c_uint)
            details = ", ".join(
                gl_get_string(enum).decode("ascii", "replace")
                for enum in (0x1F00, 0x1F01, 0x1F02)  # vendor, renderer, version
            )
        except Exception:
            details = f"OpenGL {actual.majorVersion()}.{actual.minorVersion()}"
        context.doneCurrent()
        return True, f"{name}: {details}; " + "; ".join(attempts)
    return False, "; ".join(attempts)


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    available, details = probe_context()
    print(f"platform={app.platformName()} gpu_context={available} details={details}")
    if not available:
        return 2

    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    frame[:, ::2] = (255, 255, 255)
    widget = GpuVideoWidget()
    widget.resize(320, 180)
    widget.set_frame(frame)
    widget.show()
    app.processEvents()
    for quality in ("nearest", "bilinear", "high"):
        widget.set_quality(quality)
        app.processEvents()
        image = widget.grabFramebuffer()
        print(quality, "framebuffer", image.width(), image.height(), "null", image.isNull())
    widget.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
