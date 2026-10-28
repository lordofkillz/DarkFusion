"""Native Qt OpenGL upload for the existing video ImageItem; raster fallback."""
import logging
import ctypes

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui


class VideoImageItem(pg.ImageItem):
    """Reuse the view's GL context and one texture instead of QPainter RGB upload."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._video_context = None
        self._video_viewport = None
        self._video_texture = None
        self._video_blitter = None
        self._video_uploaded = None
        self._video_gpu_failed = False

    def setImage(self, *args, **kwargs):
        self._video_uploaded = None
        return super().setImage(*args, **kwargs)

    def clear(self):
        self._video_uploaded = None
        super().clear()

    def _release_video_gl(self):
        viewport = self._video_viewport
        try:
            if viewport is not None:
                viewport.makeCurrent()
            if self._video_texture is not None:
                self._video_texture.destroy()
            if self._video_blitter is not None:
                self._video_blitter.destroy()
        except RuntimeError:
            pass  # Qt may already be destroying the native surface.
        finally:
            if viewport is not None:
                try:
                    viewport.doneCurrent()
                except RuntimeError:
                    pass
            self._video_context = self._video_viewport = None
            self._video_texture = self._video_blitter = self._video_uploaded = None

    def paint(self, painter, option, widget=None):
        image = self.image
        context = QtGui.QOpenGLContext.currentContext()
        if (self._video_gpu_failed or context is None or image is None
                or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
                or not image.flags.c_contiguous or self.levels is not None or self.lut is not None
                or self.opacity() != 1.0
                or painter.paintEngine().type() != QtGui.QPaintEngine.OpenGL2):
            return super().paint(painter, option, widget)
        native = False
        try:
            viewport = self.getViewWidget().viewport()
            painter.beginNativePainting()
            native = True
            if self._video_context is not context:
                if self._video_context is not None:
                    self._release_video_gl()
                    viewport.makeCurrent()
                self._video_context, self._video_viewport = context, viewport
                context.aboutToBeDestroyed.connect(self._release_video_gl)
                blitter = QtGui.QOpenGLTextureBlitter()
                if not blitter.create():
                    raise RuntimeError('Qt OpenGL texture blitter creation failed')
                self._video_blitter = blitter
                def proc(name, result, *params):
                    address = context.getProcAddress(name.encode())
                    if not address:
                        raise RuntimeError('OpenGL function unavailable: '+name)
                    return ctypes.CFUNCTYPE(result,*params)(int(address))
                self._video_pixel_store = proc('glPixelStorei',None,ctypes.c_uint,ctypes.c_int)
                self._video_upload = proc('glTexSubImage2D',None,ctypes.c_uint,ctypes.c_int,
                    ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint,ctypes.c_void_p)
            height, width = image.shape[:2]
            texture = self._video_texture
            if texture is None or texture.width() != width or texture.height() != height:
                if texture is not None:
                    texture.destroy()
                texture = QtGui.QOpenGLTexture(QtGui.QOpenGLTexture.Target2D)
                texture.setFormat(QtGui.QOpenGLTexture.RGBFormat if context.isOpenGLES() and context.format().majorVersion()<3
                                  else QtGui.QOpenGLTexture.RGB8_UNorm)
                texture.setSize(width, height)
                texture.setMipLevels(1)
                texture.allocateStorage(QtGui.QOpenGLTexture.RGB, QtGui.QOpenGLTexture.UInt8)
                texture.setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)
                self._video_texture = texture
                self._video_uploaded = None
            filtering = (QtGui.QOpenGLTexture.Linear if painter.testRenderHint(QtGui.QPainter.SmoothPixmapTransform)
                         else QtGui.QOpenGLTexture.Nearest)
            texture.setMinificationFilter(filtering)
            texture.setMagnificationFilter(filtering)
            target = painter.deviceTransform().mapRect(self.boundingRect())
            dpr = viewport.devicePixelRatioF()
            bounds = QtCore.QRect(0, 0, round(viewport.width()*dpr), round(viewport.height()*dpr))
            transform = QtGui.QOpenGLTextureBlitter.targetTransform(target, bounds)
            if self._video_uploaded is not image:
                texture.bind()
                self._video_pixel_store(0x0CF5,1)
                self._video_upload(0x0DE1,0,0,0,width,height,0x1907,0x1401,image.ctypes.data)
                self._video_pixel_store(0x0CF5,4)
                texture.release()
                self._video_uploaded = image
            self._video_blitter.bind()
            self._video_blitter.blit(texture.textureId(), transform, QtGui.QOpenGLTextureBlitter.OriginTopLeft)
            self._video_blitter.release()
        except Exception as error:
            self._video_gpu_failed = True
            logging.getLogger(__name__).warning('Native video upload unavailable; using QPainter: %s', error)
        finally:
            if native:
                painter.endNativePainting()
        if self._video_gpu_failed:
            return super().paint(painter, option, widget)
