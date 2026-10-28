"""Windows Desktop Duplication with owned BGR frames and no extra dependency."""
import ctypes as C
import os
import uuid

import cv2
import numpy as np

U = C.c_uint32
P = C.c_void_p
H = C.c_int32
WINFUN = getattr(C,'WINFUNCTYPE',C.CFUNCTYPE)


class GUID(C.Structure):
    _fields_=[('a',U),('b',C.c_uint16),('c',C.c_uint16),('d',C.c_ubyte*8)]


class RECT(C.Structure):
    _fields_=[('left',H),('top',H),('right',H),('bottom',H)]


class OutputDesc(C.Structure):
    _fields_=[('name',C.c_wchar*32),('rect',RECT),('attached',H),('rotation',U),('monitor',P)]


class FrameInfo(C.Structure):
    _fields_=[('present',C.c_int64),('mouse',C.c_int64),('accumulated',U),('coalesced',H),
              ('protected',H),('mouse_x',H),('mouse_y',H),('visible',H),('metadata',U),('pointer_bytes',U)]


class TextureDesc(C.Structure):
    _fields_=[('width',U),('height',U),('mips',U),('arrays',U),('format',U),
              ('sample_count',U),('sample_quality',U),('usage',U),('bind',U),('cpu',U),('misc',U)]


class Mapped(C.Structure):
    _fields_=[('data',P),('row_pitch',U),('depth_pitch',U)]


def check(result):
    if result<0:
        raise OSError('Desktop Duplication HRESULT 0x%08x'%(result & 0xffffffff))


class Com:
    def __init__(self,ptr):
        self.ptr=P(ptr.value if isinstance(ptr,P) else ptr)
    def call(self,index,types,values,restype=H):
        table=C.cast(self.ptr,C.POINTER(C.POINTER(P))).contents
        method=WINFUN(restype,P,*types)(table[index])
        return method(self.ptr,*values)
    def query(self,guid):
        iid=GUID.from_buffer_copy(uuid.UUID(guid).bytes_le)
        result=P()
        check(self.call(0,(P,P),(C.byref(iid),C.byref(result))))
        return Com(result)
    def release(self):
        if self.ptr:
            self.call(2,(),(),U)
            self.ptr=P()


class DesktopDuplication:
    """One selected, unrotated monitor; caller falls back to MSS on any failure.

    Creation, grabbing and release belong to the capture thread. GPU staging
    memory is converted while mapped, so consumers never retain driver buffers.
    Static desktop polls return independent copies just as ordinary screenshots.
    """
    def __init__(self,monitor):
        if os.name!='nt':
            raise OSError('Desktop Duplication requires Windows')
        self._device=self._context=self._dup=self._stage=None
        self._cache=None
        self.new_frame_count=0
        self._last_present=0
        self._width,self._height=int(monitor['width']),int(monitor['height'])
        self._bounds=(int(monitor['left']),int(monitor['top']),
                      int(monitor['left'])+self._width,int(monitor['top'])+self._height)
        try:
            self._open()
        except BaseException:
            self.close()
            raise

    def _open(self):
        dxgi=C.WinDLL('dxgi.dll')
        create=dxgi.CreateDXGIFactory1
        create.argtypes=(P,P)
        create.restype=H
        iid=GUID.from_buffer_copy(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
        ptr=P()
        check(create(C.byref(iid),C.byref(ptr)))
        factory=Com(ptr)
        try:
            for adapter_index in range(16):
                ptr=P()
                result=factory.call(12,(U,P),(adapter_index,C.byref(ptr)))
                if result & 0xffffffff == 0x887a0002:
                    break
                check(result)
                adapter=Com(ptr)
                try:
                    for output_index in range(32):
                        ptr=P()
                        result=adapter.call(7,(U,P),(output_index,C.byref(ptr)))
                        if result & 0xffffffff == 0x887a0002:
                            break
                        check(result)
                        output=Com(ptr)
                        try:
                            desc=OutputDesc()
                            check(output.call(7,(P,),(C.byref(desc),)))
                            bounds=(desc.rect.left,desc.rect.top,desc.rect.right,desc.rect.bottom)
                            if desc.attached and desc.rotation==1 and bounds==self._bounds:
                                self._create_device(adapter,output)
                                return
                        finally:
                            output.release()
                finally:
                    adapter.release()
        finally:
            factory.release()
        raise OSError('Selected desktop geometry requires the MSS capture path')

    def _create_device(self,adapter,output):
        d3d=C.WinDLL('d3d11.dll')
        create=d3d.D3D11CreateDevice
        create.argtypes=(P,U,P,U,P,U,U,P,P,P)
        create.restype=H
        device,context,level=P(),P(),U()
        result=create(adapter.ptr,0,None,0x20,None,0,7,C.byref(device),C.byref(level),C.byref(context))
        if device: self._device=Com(device)
        if context: self._context=Com(context)
        check(result)
        output1=output.query('00cddea8-939b-4b83-a340-a685226666cc')
        try:
            ptr=P()
            check(output1.call(22,(P,P),(device,C.byref(ptr))))
            self._dup=Com(ptr)
        finally:
            output1.release()

    def grab_bgr(self):
        info=FrameInfo()
        ptr=P()
        result=self._dup.call(8,(U,P,P),(0,C.byref(info),C.byref(ptr)))
        if result & 0xffffffff == 0x887a0027:
            return self._cache.copy() if self._cache is not None else None
        check(result)
        resource=Com(ptr)
        texture=None
        mapped=False
        try:
            if not info.present or (self._cache is not None and info.present<=self._last_present):
                return self._cache.copy() if self._cache is not None else None
            texture=resource.query('6f15aaf2-d208-4e89-9ab4-489535d34f9c')
            desc=TextureDesc()
            texture.call(10,(P,),(C.byref(desc),),None)
            if desc.width!=self._width or desc.height!=self._height or desc.format!=87:
                raise OSError('Desktop size/format changed; reopen through MSS')
            if self._stage is None:
                desc.mips=desc.arrays=desc.sample_count=1
                desc.sample_quality=desc.bind=desc.misc=0
                desc.usage=3
                desc.cpu=0x20000
                ptr=P()
                check(self._device.call(5,(P,P,P),(C.byref(desc),None,C.byref(ptr))))
                self._stage=Com(ptr)
            self._context.call(47,(P,P),(self._stage.ptr,texture.ptr),None)
            pixels=Mapped()
            check(self._context.call(14,(P,U,U,U,P),(self._stage.ptr,0,1,0,C.byref(pixels))))
            mapped=True
            raw=(C.c_ubyte*(pixels.row_pitch*self._height)).from_address(pixels.data)
            image=np.ndarray((self._height,self._width,4),np.uint8,buffer=raw,
                             strides=(pixels.row_pitch,4,1))
            frame=cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
            self._cache=frame.copy()
            if info.present>self._last_present:
                self.new_frame_count+=1
                self._last_present=info.present
            return frame
        finally:
            if mapped:
                self._context.call(15,(P,U),(self._stage.ptr,0),None)
            if texture is not None:
                texture.release()
            resource.release()
            self._dup.call(14,(),())

    def close(self):
        for name in ('_stage','_dup','_context','_device'):
            resource=getattr(self,name,None)
            if resource is not None:
                resource.release()
                setattr(self,name,None)
        self._cache=None
