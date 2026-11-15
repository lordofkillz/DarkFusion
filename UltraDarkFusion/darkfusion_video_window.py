"""Independent Qt video swapchain, accepting RGB or native OpenCV BGR buffers."""
import ctypes as C
import cv2
import numpy as np

from PyQt5 import QtCore,QtGui


class VideoWindow(QtGui.QOpenGLWindow):
    failed=QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__(QtGui.QOpenGLWindow.NoPartialUpdate)
        fmt=QtGui.QSurfaceFormat.defaultFormat()
        if QtGui.QGuiApplication.platformName()=='windows' and fmt.majorVersion()<3:
            # Keep the older context when this driver cannot create ES/GL 3.
            # Pixel-buffer uploads are optional; the original upload remains
            # available on the same surface and on older graphics hardware.
            preferred=QtGui.QSurfaceFormat(fmt)
            preferred.setVersion(3,0)
            probe=QtGui.QOpenGLContext()
            probe.setFormat(preferred)
            if probe.create() and probe.isValid():
                fmt=preferred
        fmt.setSwapBehavior(QtGui.QSurfaceFormat.DoubleBuffer)
        # This child surface has its own swapchain. Desktop composition controls
        # scan-out; avoid holding the entire GUI on an additional swap wait.
        fmt.setSwapInterval(0)
        self.setFormat(fmt)
        self.setFlags(QtCore.Qt.FramelessWindowHint|QtCore.Qt.WindowDoesNotAcceptFocus|QtCore.Qt.WindowTransparentForInput)
        self._frame=self._uploaded=None
        self._texture=self._blitter=None
        self._smooth=True
        self._bgr=False
        self._rgba_scratch=None
        self._pixel_buffers=None
        self._pixel_buffer_index=0
        self._pixel_buffer_sizes=[0,0]
        self._pixel_buffer_failed=False
        self._gen_buffers=self._bind_buffer=self._buffer_data=None
        self._map_buffer=self._unmap_buffer=self._delete_buffers=None
        self._failed=False
        self.paint_count=0
        self.immediate=False

    def devicePixelRatioF(self):
        return self.devicePixelRatio()

    def set_frame(self,frame,smooth=True,*,bgr=False):
        if frame is not None and not frame.flags.c_contiguous:
            frame=np.ascontiguousarray(frame)
        self._frame=frame
        self._uploaded=None
        if frame is None:
            self._rgba_scratch=None
            if self._pixel_buffers is not None and self.context() is not None:
                self.makeCurrent()
                try:self._release_pixel_buffers()
                finally:self.doneCurrent()
        self._smooth=smooth
        self._bgr=bool(bgr)
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
            if context.format().majorVersion()>=3:
                try:
                    self._gen_buffers=fn('glGenBuffers',None,C.c_int,C.POINTER(C.c_uint))
                    self._bind_buffer=fn('glBindBuffer',None,C.c_uint,C.c_uint)
                    self._buffer_data=fn('glBufferData',None,C.c_uint,C.c_ssize_t,C.c_void_p,C.c_uint)
                    self._map_buffer=fn('glMapBufferRange',C.c_void_p,C.c_uint,C.c_ssize_t,C.c_ssize_t,C.c_uint)
                    self._unmap_buffer=fn('glUnmapBuffer',C.c_ubyte,C.c_uint)
                    self._delete_buffers=fn('glDeleteBuffers',None,C.c_int,C.POINTER(C.c_uint))
                except RuntimeError:
                    self._pixel_buffer_failed=True
            self._blitter=QtGui.QOpenGLTextureBlitter()
            if not self._blitter.create():
                raise RuntimeError('OpenGL texture blitter creation failed')
            context.aboutToBeDestroyed.connect(self.shutdown)
        except Exception as error:
            self._failed=True
            self.failed.emit(str(error))

    def _release_pixel_buffers(self):
        buffers,self._pixel_buffers=self._pixel_buffers,None
        self._pixel_buffer_index=0
        self._pixel_buffer_sizes=[0,0]
        if buffers is not None:
            self._bind_buffer(0x88EC,0)
            self._delete_buffers(len(buffers),buffers)

    def _upload_mapped_rgba(self,image):
        """Convert straight into a bounded upload buffer, retaining the CPU fallback."""
        height,width=image.shape[:2]
        size=height*width*4
        if (self._pixel_buffer_failed or self._map_buffer is None
                or size*2>64*1024*1024):
            self._release_pixel_buffers()
            return False
        target=0x88EC # PIXEL_UNPACK_BUFFER
        mapped=False
        try:
            if self._pixel_buffers is None:
                self._pixel_buffers=(C.c_uint*2)()
                self._gen_buffers(2,self._pixel_buffers)
                if not all(self._pixel_buffers):
                    raise RuntimeError('Pixel-buffer allocation unavailable')
            slot=self._pixel_buffer_index
            self._bind_buffer(target,self._pixel_buffers[slot])
            self._pixel_buffer_index=1-self._pixel_buffer_index
            if self._pixel_buffer_sizes[slot]!=size:
                self._buffer_data(target,size,None,0x88E0) # STREAM_DRAW
                self._pixel_buffer_sizes[slot]=size
            pointer=self._map_buffer(target,0,size,0x0002|0x0008) # WRITE | INVALIDATE_BUFFER
            if not pointer:
                raise RuntimeError('Pixel-buffer mapping unavailable')
            mapped=True
            storage=(C.c_ubyte*size).from_address(pointer)
            pixels=np.ctypeslib.as_array(storage).reshape(height,width,4)
            cv2.cvtColor(image,cv2.COLOR_RGB2RGBA,dst=pixels)
            intact=self._unmap_buffer(target)
            mapped=False
            if not intact:
                raise RuntimeError('Pixel-buffer contents invalidated')
            self._upload(0x0de1,0,0,0,width,height,0x1908,0x1401,None)
            self._rgba_scratch=None
            return True
        except (RuntimeError,ValueError,cv2.error):
            self._pixel_buffer_failed=True
            return False
        finally:
            if mapped:
                self._unmap_buffer(target)
            self._bind_buffer(target,0)
            if self._pixel_buffer_failed:
                self._release_pixel_buffers()

    def paintGL(self):
        if self._failed:
            return
        try:
            dpr=self.devicePixelRatio()
            bounds=QtCore.QRect(0,0,round(self.width()*dpr),round(self.height()*dpr))
            self._viewport(0,0,bounds.width(),bounds.height())
            self._clear_color(0.,0.,0.,1.)
            self._clear(0x4000)
            source_image=self._frame
            image=source_image
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
                    self._release_pixel_buffers()
            filtering=QtGui.QOpenGLTexture.Linear if self._smooth else QtGui.QOpenGLTexture.Nearest
            texture.setMinificationFilter(filtering)
            texture.setMagnificationFilter(filtering)
            if self._uploaded is not source_image:
                texture.bind()
                self._pixel_store(0x0cf5,1)
                if not rgba or not self._upload_mapped_rgba(image):
                    upload = image
                    if rgba:
                        scratch = self._rgba_scratch
                        if scratch is None or scratch.shape[:2] != image.shape[:2]:
                            scratch = self._rgba_scratch = np.empty((height,width,4),np.uint8)
                        upload = cv2.cvtColor(image,cv2.COLOR_RGB2RGBA,dst=scratch)
                    pixel_format=0x1908 if rgba else 0x1907
                    self._upload(0x0de1,0,0,0,width,height,pixel_format,0x1401,upload.ctypes.data)
                self._pixel_store(0x0cf5,4)
                texture.release()
                self._uploaded=source_image
            # Match the existing graphics view's inner margins and aspect fit.
            scale=min(max(1,self.width()-18)/width,max(1,self.height()-18)/height)
            w,h=width*scale,height*scale
            target=QtCore.QRectF((self.width()-w)*dpr/2,(self.height()-h)*dpr/2,w*dpr,h*dpr)
            transform=QtGui.QOpenGLTextureBlitter.targetTransform(target,bounds)
            # OpenCV already supplies BGR. Swap channels in the existing blit
            # shader rather than copying the full source frame on the CPU.
            self._blitter.setRedBlueSwizzle(self._bgr)
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
            self._release_pixel_buffers()
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
