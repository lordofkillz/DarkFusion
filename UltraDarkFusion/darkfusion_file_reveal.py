"""Reveal an exact filesystem item without passing its path through a shell."""
import ctypes
import os
import subprocess
import sys


def _windows_select_file(path, shell=None, ole=None):
    """Use a literal Unicode PIDL; spaces, commas and ampersands are not syntax.

    SHOpenFolderAndSelectItems with cidl=0 selects the fully specified item in
    its parent folder. Call from the reveal worker, not the GUI thread.
    https://learn.microsoft.com/windows/win32/api/shlobj_core/nf-shlobj_core-shopenfolderandselectitems
    """
    from ctypes import wintypes
    shell = shell if shell is not None else ctypes.WinDLL("shell32", use_last_error=True)
    ole = ole if ole is not None else ctypes.WinDLL("ole32", use_last_error=True)
    ole.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole.CoInitializeEx.restype = ctypes.c_long
    ole.CoUninitialize.argtypes = []
    ole.CoUninitialize.restype = None
    ole.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole.CoTaskMemFree.restype = None
    shell.SHParseDisplayName.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_void_p), wintypes.ULONG,
                                        ctypes.POINTER(wintypes.ULONG)]
    shell.SHParseDisplayName.restype = ctypes.c_long
    shell.SHOpenFolderAndSelectItems.argtypes = [ctypes.c_void_p, wintypes.UINT,
                                                ctypes.c_void_p, wintypes.DWORD]
    shell.SHOpenFolderAndSelectItems.restype = ctypes.c_long

    def checked(status, operation):
        if status & 0x80000000:
            raise OSError(f"{operation} failed (0x{status & 0xffffffff:08X})")

    initialized = ole.CoInitializeEx(None, 2)
    # A thread already initialized with a different apartment may still use
    # the Shell. Balance only successful CoInitializeEx calls made here.
    if initialized & 0xffffffff != 0x80010106:
        checked(initialized, "Initialize Windows Shell")
    pidl = ctypes.c_void_p()
    try:
        checked(shell.SHParseDisplayName(path, None, ctypes.byref(pidl), 0, None), "Find file")
        if not pidl.value:
            raise OSError("Windows Shell did not resolve the file")
        checked(shell.SHOpenFolderAndSelectItems(pidl, 0, None, 0), "Select file in Explorer")
    finally:
        if pidl.value:
            ole.CoTaskMemFree(pidl)
        if initialized in (0, 1):
            ole.CoUninitialize()


def reveal_file(path):
    """Return whether the file was selected (Linux opens its containing folder)."""
    path = os.path.normpath(os.path.abspath(os.fspath(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if sys.platform.startswith("win"):
        _windows_select_file(path)
        return True
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
        return True
    subprocess.Popen(["xdg-open", os.path.dirname(path)])
    return False
