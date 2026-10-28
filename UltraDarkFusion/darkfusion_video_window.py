"""Independent Qt video swapchain, using the existing RGB display buffers."""
import ctypes as C
import cv2
import numpy as np

from PyQt5 import QtCore,QtGui


class VideoWindow(QtGui.QOpenGLWindow):
    failed=QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__(QtGui.QOpenGLWindow.NoPartialUpdate)
        fmt=QtGui.QSurfaceFormat.defaultFormat()
        fmt.setSwapBehavior(QtGui.QSurfaceFormat.DoubleBuffer)
        # This child surface has its own swapchain. Desktop composition controls
        # scan-out; avoid holding the entire GUI on an additional swap wait.
        fmt.setSwapInterval(0)
        self.setFormat(fmt)
        self.setFlags(QtCore.Qt.FramelessWindowHint|QtCore.Qt.WindowDoesNotAcceptFocus|QtCore.Qt.WindowTransparentForInput)
        self._frame=self._uploaded=None
        self._texture=self._blitter=None
        self._smooth=True
        self._rgba_scratch=None
        self._failed=False
        self.paint_count=0
        self.immediate=False

    def devicePixelRatioF(self):
        return self.devicePixelRatio()

    def set_frame(self,frame,smooth=True):
        self._frame=frame
        self._uploaded=None
        if frame is None:
            self._rgba_scratch=None
        self._smooth=smooth
        if self.immediate and self.context() is not None and self.isExposed() and self._blitter is not None:
            self.makeCurrent()
            self.paintGL()
            self.context().swapBuffers(self)
            self.frameSwapped.emit()
            self.doneCurrent()
        else:
            self.update()

    def initializeGL(self):
        context=self.context()
        def fn(name,result,*params):
            pointer=context.getProcAddress(name.encode())
            if not pointer:
                raise RuntimeError('OpenGL function unavailable: '+name)
            return C.CFUNCTYPE(result,*params)(int(pointer))
        try:
            self._clear_color=fn('glClearColor',None,C.c_float,C.c_float,C.c_float,C.c_float)
            self._clear=fn('glClear',None,C.c_uint)
            self._viewport=fn('glViewport',None,C.c_int,C.c_int,C.c_int,C.c_int)
            self._pixel_store=fn('glPixelStorei',None,C.c_uint,C.c_int)
            self._upload=fn('glTexSubImage2D',None,C.c_uint,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_uint,C.c_uint,C.c_void_p)
            self._blitter=QtGui.QOpenGLTextureBlitter()
            if not self._blitter.create():
                raise RuntimeError('OpenGL texture blitter creation failed')
            context.aboutToBeDestroyed.connect(self.shutdown)
        except Exception as error:
            self._failed=True
            self.failed.emit(str(error))

    def paintGL(self):
        if self._failed:
            return
        try:
            dpr=self.devicePixelRatio()
            bounds=QtCore.QRect(0,0,round(self.width()*dpr),round(self.height()*dpr))
            self._viewport(0,0,bounds.width(),bounds.height())
            self._clear_color(0.,0.,0.,1.)
            self._clear(0x4000)
            image=self._frame
            if image is None:
                return
            height,width=image.shape[:2]
            # ANGLE expands packed RGB on upload. At large resolutions a
            # reusable CPU RGBA buffer avoids that more expensive driver path.
            rgba = width * height >= 1024 * 1024
            texture=self._texture
            if texture is None or texture.width()!=width or texture.height()!=height:
                if texture is not None:
                    texture.destroy()
                texture=QtGui.QOpenGLTexture(QtGui.QOpenGLTexture.Target2D)
                context=self.context()
                legacy = context.isOpenGLES() and context.format().majorVersion()<3
                texture.setFormat((QtGui.QOpenGLTexture.RGBAFormat if legacy else QtGui.QOpenGLTexture.RGBA8_UNorm)
                                  if rgba else (QtGui.QOpenGLTexture.RGBFormat if legacy else QtGui.QOpenGLTexture.RGB8_UNorm))
                texture.setSize(width,height)
                texture.setMipLevels(1)
                texture.allocateStorage(QtGui.QOpenGLTexture.RGBA if rgba else QtGui.QOpenGLTexture.RGB,
                                        QtGui.QOpenGLTexture.UInt8)
                texture.setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)
                self._texture=texture
                self._uploaded=None
                if not rgba:
                    self._rgba_scratch=None
            filtering=QtGui.QOpenGLTexture.Linear if self._smooth else QtGui.QOpenGLTexture.Nearest
            texture.setMinificationFilter(filtering)
            texture.setMagnificationFilter(filtering)
            if self._uploaded is not image:
                upload = image
                if rgba:
                    scratch = self._rgba_scratch
                    if scratch is None or scratch.shape[:2] != image.shape[:2]:
                        scratch = self._rgba_scratch = np.empty((height,width,4),np.uint8)
                    upload = cv2.cvtColor(image,cv2.COLOR_RGB2RGBA,dst=scratch)
                texture.bind()
                self._pixel_store(0x0cf5,1)
                self._upload(0x0de1,0,0,0,width,height,0x1908 if rgba else 0x1907,0x1401,upload.ctypes.data)
                self._pixel_store(0x0cf5,4)
                texture.release()
                self._uploaded=image
            # Match the existing graphics view's inner margins and aspect fit.
            scale=min(max(1,self.width()-18)/width,max(1,self.height()-18)/height)
            w,h=width*scale,height*scale
            target=QtCore.QRectF((self.width()-w)*dpr/2,(self.height()-h)*dpr/2,w*dpr,h*dpr)
            transform=QtGui.QOpenGLTextureBlitter.targetTransform(target,bounds)
            self._blitter.bind()
            self._blitter.blit(texture.textureId(),transform,QtGui.QOpenGLTextureBlitter.OriginTopLeft)
            self._blitter.release()
            self.paint_count+=1
        except Exception as error:
            self._failed=True
            self.failed.emit(str(error))

    def shutdown(self):
        try:
            self.makeCurrent()
            if self._texture is not None:
                self._texture.destroy()
            if self._blitter is not None:
                self._blitter.destroy()
            self.doneCurrent()
        except RuntimeError:
            pass
        self._texture=self._blitter=None
        self._frame=self._uploaded=None
        self._rgba_scratch=None
