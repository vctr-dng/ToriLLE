"""RGB capture of the Toribash window (Linux, X11).

The game has no frame buffer API, so frames are read from the X server: the
window of the game process is located and its pixels are fetched with
`XGetImage` (through `ctypes`, no extra dependency). Between two turns the
game is paused waiting for the action, so a grab taken right after
`get_state()` shows the state that was just read.

Pure helpers (`square_crop_box`, `letterbox_geometry`, `resize_area`,
`process_frame`) do not need a display and are unit tested without the game.
"""

import ctypes
import ctypes.util
import itertools
import os
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

CAPTURE_CROPS = ("center", "letterbox")

# Frames with fewer distinct levels than this on every channel are suspect.
_FLAT_FRAME_STD = 1.0


def square_crop_box(width, height):
    """Largest centred square of a `width` x `height` image: `(x, y, side)`."""
    side = min(width, height)
    return (width - side) // 2, (height - side) // 2, side


def letterbox_geometry(width, height, size):
    """
    Size and offset of a `width` x `height` image scaled to fit a `size` x
    `size` canvas with the aspect ratio kept: `(out_w, out_h, x, y)`.
    """
    scale = size / max(width, height)
    out_w = max(1, min(size, round(width * scale)))
    out_h = max(1, min(size, round(height * scale)))
    return out_w, out_h, (size - out_w) // 2, (size - out_h) // 2


def _bilinear(image, out_h, out_w):
    """Bilinear resize (pixel centres aligned) of a small `(H, W, C)` image."""

    def axis_weights(source, target):
        position = (np.arange(target) + 0.5) * source / target - 0.5
        position = position.clip(0, source - 1)
        low = np.floor(position).astype(np.int64)
        high = np.minimum(low + 1, source - 1)
        return low, high, (position - low).astype(np.float32)

    y0, y1, wy = axis_weights(image.shape[0], out_h)
    x0, x1, wx = axis_weights(image.shape[1], out_w)
    rows = image[y0].astype(np.float32) * (1 - wy)[:, None, None]
    rows += image[y1] * wy[:, None, None]
    out = rows[:, x0] * (1 - wx)[None, :, None]
    out += rows[:, x1] * wx[None, :, None]
    return np.rint(out).clip(0, 255).astype(np.uint8)


_THREADS = 4
_pool = None


def _block_means(image, factor_h, factor_w):
    out_h, out_w = image.shape[0] // factor_h, image.shape[1] // factor_w
    blocks = image[: out_h * factor_h, : out_w * factor_w].reshape(
        out_h, factor_h, out_w, factor_w, image.shape[2]
    )
    count = factor_h * factor_w
    dtype = np.uint16 if count * 255 < 2**16 else np.uint32
    total = np.zeros((out_h, out_w, image.shape[2]), dtype)
    for i in range(factor_h):  # strided slice sums are faster than np.sum
        for j in range(factor_w):
            total += blocks[:, i, :, j]
    return ((total + count // 2) // count).astype(np.uint8)


def _shrink_by_integer(image, factor_h, factor_w):
    """
    Mean of `factor_h` x `factor_w` pixel blocks (a trailing remainder is
    dropped). Large images are split over a few threads (numpy releases the
    GIL), which makes it about three times faster.
    """
    global _pool
    out_h = image.shape[0] // factor_h
    if out_h < 4 * _THREADS or image.size < 2_000_000:
        return _block_means(image, factor_h, factor_w)
    if _pool is None:
        _pool = ThreadPoolExecutor(_THREADS)
    bounds = np.linspace(0, out_h, _THREADS + 1).astype(int)
    futures = [
        _pool.submit(
            _block_means, image[lo * factor_h : hi * factor_h], factor_h, factor_w
        )
        for lo, hi in itertools.pairwise(bounds)
    ]
    return np.concatenate([future.result() for future in futures])


def resize_area(image, out_h, out_w):
    """
    Resize an `(H, W, C)` uint8 image to `(out_h, out_w, C)`.

    Shrinking averages blocks of pixels (a box filter, so it is anti-aliased):
    first by the integer factor `H // out_h` (a few pixels at the edge are
    dropped when it does not divide evenly), then bilinearly down to the
    requested size. Growing repeats the nearest pixel. Pure numpy (about 30 ms for
    a 1440 px square to 256 px while the game runs).
    """
    height, width = image.shape[:2]
    if (height, width) == (out_h, out_w):
        return np.ascontiguousarray(image)
    if out_h > height or out_w > width:
        rows = (np.arange(out_h) + 0.5) * height // out_h
        cols = (np.arange(out_w) + 0.5) * width // out_w
        return image[rows.astype(np.int64)][:, cols.astype(np.int64)]
    factor_h, factor_w = height // out_h, width // out_w
    if factor_h > 1 or factor_w > 1:
        image = _shrink_by_integer(image, factor_h, factor_w)
    return _bilinear(image, out_h, out_w)


def process_frame(image, size, crop="center"):
    """
    Turn a window image (`(H, W, 3)` uint8 RGB) into a `(size, size, 3)` uint8
    frame.

    crop="center": centred square crop of side `min(H, W)`, then resized.
        No bars, but the outer parts of the long side are cut off.
    crop="letterbox": the whole image scaled to fit, centred on a black
        `size` x `size` canvas.
    """
    if crop not in CAPTURE_CROPS:
        raise ValueError(f"capture_crop must be one of {CAPTURE_CROPS}, got {crop!r}")
    height, width = image.shape[:2]
    if crop == "center":
        x, y, side = square_crop_box(width, height)
        return resize_area(image[y : y + side, x : x + side], size, size)
    out_w, out_h, x, y = letterbox_geometry(width, height, size)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[y : y + out_h, x : x + out_w] = resize_area(image, out_h, out_w)
    return canvas


def is_flat_frame(frame):
    """True if the frame is (nearly) one colour: black, white, a lock screen..."""
    return float(frame.std()) < _FLAT_FRAME_STD


def capture_observation_space(size):
    """The `Box` space of one frame."""
    from gymnasium import spaces

    return spaces.Box(0, 255, (size, size, 3), np.uint8)


def validate_capture_options(draw_game, capture_size, capture_crop):
    """Raise a clear error for options that cannot work. Used by the envs."""
    if not isinstance(capture_size, int) or isinstance(capture_size, bool):
        raise TypeError(f"capture_size must be an int, got {capture_size!r}")
    if not 16 <= capture_size <= 1024:
        raise ValueError(f"capture_size must be in [16, 1024], got {capture_size}")
    if capture_crop not in CAPTURE_CROPS:
        raise ValueError(
            f"capture_crop must be one of {CAPTURE_CROPS}, got {capture_crop!r}"
        )
    if not draw_game:
        raise ValueError(
            "RGB capture needs the game to be drawn: pass draw_game=True "
            "(without it Toribash does not render the window)"
        )


# ---------------------------------------------------------------- X11 backend


class _XImage(ctypes.Structure):
    _fields_ = (
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("xoffset", ctypes.c_int),
        ("format", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("byte_order", ctypes.c_int),
        ("bitmap_unit", ctypes.c_int),
        ("bitmap_bit_order", ctypes.c_int),
        ("bitmap_pad", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("bytes_per_line", ctypes.c_int),
        ("bits_per_pixel", ctypes.c_int),
        ("red_mask", ctypes.c_ulong),
        ("green_mask", ctypes.c_ulong),
        ("blue_mask", ctypes.c_ulong),
    )


class _XWindowAttributes(ctypes.Structure):
    _fields_ = (
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("border_width", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("visual", ctypes.c_void_p),
        ("root", ctypes.c_ulong),
        ("class_", ctypes.c_int),
        ("bit_gravity", ctypes.c_int),
        ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int),
        ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong),
        ("save_under", ctypes.c_int),
        ("colormap", ctypes.c_ulong),
        ("map_installed", ctypes.c_int),
        ("map_state", ctypes.c_int),
        ("all_event_masks", ctypes.c_long),
        ("your_event_mask", ctypes.c_long),
        ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int),
        ("screen", ctypes.c_void_p),
    )


class _XClassHint(ctypes.Structure):
    _fields_ = (("res_name", ctypes.c_char_p), ("res_class", ctypes.c_char_p))


class _XShmSegmentInfo(ctypes.Structure):
    _fields_ = (
        ("shmseg", ctypes.c_ulong),
        ("shmid", ctypes.c_int),
        ("shmaddr", ctypes.c_void_p),
        ("readOnly", ctypes.c_int),
    )


_IPC_PRIVATE = 0
_IPC_CREAT = 0o1000
_IPC_RMID = 0
_Z_PIXMAP = 2
_ALL_PLANES = ctypes.c_ulong(~0)
_IS_VIEWABLE = 2


class CaptureError(RuntimeError):
    """The window could not be found or read."""


def _ximage_to_rgb(raw):
    """RGB copy of an XImage (32 bit ZPixmap, B G R X in memory)."""
    if raw.bits_per_pixel != 32:
        raise CaptureError(f"unsupported pixel depth {raw.bits_per_pixel}")
    buffer = (ctypes.c_ubyte * (raw.bytes_per_line * raw.height)).from_address(raw.data)
    pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(
        raw.height, raw.bytes_per_line // 4, 4
    )
    return pixels[:, : raw.width, 2::-1].copy()


class _SharedMemoryGrabber:
    """
    `XShmGetImage`: the X server writes the pixels into a System V shared
    memory segment instead of sending them over the socket (about 5x faster
    for a 1440 px square). Returns None from `grab` if anything is unavailable.
    """

    def __init__(self, x11, display):
        self._x11 = x11
        self._display = display
        self._image = None
        self._info = None
        self._size = None
        self._libc = ctypes.CDLL(ctypes.util.find_library("c") or None)
        self._libc.shmat.restype = ctypes.c_void_p
        self._libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
        self._libc.shmget.argtypes = (ctypes.c_int, ctypes.c_size_t, ctypes.c_int)
        self._libc.shmctl.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        self._libc.shmdt.argtypes = (ctypes.c_void_p,)
        library = ctypes.util.find_library("Xext")
        self._xext = ctypes.CDLL(library) if library else None
        if self._xext is None:
            self._unavailable = True
            return
        xext = self._xext
        xext.XShmQueryExtension.argtypes = (ctypes.c_void_p,)
        xext.XShmCreateImage.restype = ctypes.POINTER(_XImage)
        xext.XShmCreateImage.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.POINTER(_XShmSegmentInfo),
            ctypes.c_uint,
            ctypes.c_uint,
        )
        xext.XShmAttach.argtypes = (ctypes.c_void_p, ctypes.POINTER(_XShmSegmentInfo))
        xext.XShmDetach.argtypes = (ctypes.c_void_p, ctypes.POINTER(_XShmSegmentInfo))
        xext.XShmGetImage.argtypes = (
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(_XImage),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulong,
        )
        self._unavailable = not xext.XShmQueryExtension(display)

    def _release(self):
        if self._image is not None:
            self._xext.XShmDetach(self._display, ctypes.byref(self._info))
            self._x11.XSync.argtypes = (ctypes.c_void_p, ctypes.c_int)
            self._x11.XSync(self._display, 0)
            self._libc.shmdt(self._info.shmaddr)
            self._image.contents.data = None
            self._x11.XDestroyImage(self._image)
            self._image = self._info = self._size = None

    def _create(self, info, width, height):
        self._release()
        image_info = _XShmSegmentInfo()
        image = self._xext.XShmCreateImage(
            self._display, info.visual, info.depth, _Z_PIXMAP, None,
            ctypes.byref(image_info), width, height,
        )  # fmt: skip
        if not image:
            return False
        size = image.contents.bytes_per_line * height
        image_info.shmid = self._libc.shmget(_IPC_PRIVATE, size, _IPC_CREAT | 0o600)
        if image_info.shmid < 0:
            return False
        address = self._libc.shmat(image_info.shmid, None, 0)
        if address in (None, ctypes.c_void_p(-1).value):
            return False
        image_info.shmaddr = address
        image.contents.data = address
        image_info.readOnly = 0
        ok = self._xext.XShmAttach(self._display, ctypes.byref(image_info))
        self._x11.XSync.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self._x11.XSync(self._display, 0)
        # The segment lives until detached, the id is no longer needed.
        self._libc.shmctl(image_info.shmid, _IPC_RMID, None)
        if not ok:
            return False
        self._image, self._info, self._size = image, image_info, (width, height)
        return True

    def grab(self, window, info, x, y, width, height):
        if self._unavailable:
            return None
        if self._size != (width, height) and not self._create(info, width, height):
            self._unavailable = True
            return None
        ok = self._xext.XShmGetImage(
            self._display, window, self._image, x, y, _ALL_PLANES
        )
        return _ximage_to_rgb(self._image.contents) if ok else None

    def close(self):
        if not self._unavailable:
            self._release()


class X11WindowGrabber:
    """
    Grab the pixels of a window with `XGetImage`.

    The window is the one with the WM class `window_class` (the Steam build of
    Toribash is `toribash_steam`, found by scanning the window tree). The
    request is made on the window itself, not on the screen: with a compositing
    window manager (GNOME/mutter) windows hidden behind other windows are
    still returned correctly, see docs/envs.md for what was verified.
    """

    def __init__(self, window_class=b"toribash_steam", display=None, use_shm=True):
        library = ctypes.util.find_library("X11")
        if library is None:
            raise CaptureError("libX11 was not found, RGB capture needs X11")
        x11 = ctypes.CDLL(library)
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = (ctypes.c_char_p,)
        x11.XDefaultRootWindow.restype = ctypes.c_ulong
        x11.XDefaultRootWindow.argtypes = (ctypes.c_void_p,)
        x11.XQueryTree.argtypes = (
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
            ctypes.POINTER(ctypes.c_uint),
        )
        x11.XFree.argtypes = (ctypes.c_void_p,)
        x11.XGetClassHint.argtypes = (
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(_XClassHint),
        )
        x11.XGetWindowAttributes.argtypes = (
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(_XWindowAttributes),
        )
        x11.XGetImage.restype = ctypes.POINTER(_XImage)
        x11.XGetImage.argtypes = (
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_ulong,
            ctypes.c_int,
        )
        x11.XDestroyImage.argtypes = (ctypes.POINTER(_XImage),)
        x11.XSetErrorHandler.argtypes = (ctypes.c_void_p,)
        self._x11 = x11
        self.window_class = window_class
        name = display or os.environ.get("DISPLAY")
        if not name:
            raise CaptureError("DISPLAY is not set, RGB capture needs an X display")
        self.display_name = name
        self._display = x11.XOpenDisplay(name.encode())
        if not self._display:
            raise CaptureError(f"cannot open the X display {name!r}")
        self._shm = _SharedMemoryGrabber(x11, self._display) if use_shm else None
        self._root = x11.XDefaultRootWindow(self._display)
        # Xlib's default error handler exits the process. A window that
        # vanishes (game closed) must be an exception instead.
        handler_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        self._errors = 0

        def on_error(_display, _event):
            self._errors += 1
            return 0

        self._handler = handler_type(on_error)
        x11.XSetErrorHandler(self._handler)
        self.window = None

    def _children(self, window):
        root = ctypes.c_ulong()
        parent = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint()
        ok = self._x11.XQueryTree(
            self._display,
            window,
            ctypes.byref(root),
            ctypes.byref(parent),
            ctypes.byref(children),
            ctypes.byref(count),
        )
        if not ok or not children:
            return []
        found = list(children[: count.value])
        self._x11.XFree(children)
        return found

    def _matches(self, window):
        hint = _XClassHint()
        if not self._x11.XGetClassHint(self._display, window, ctypes.byref(hint)):
            return False
        try:
            return self.window_class in (hint.res_name, hint.res_class)
        finally:
            for pointer in (hint.res_name, hint.res_class):
                del pointer  # strings are owned by Xlib, leaked on purpose (tiny)

    def find_window(self):
        """Window id of the game (searching two levels below the root)."""
        stack = [(w, 0) for w in self._children(self._root)]
        while stack:
            window, depth = stack.pop()
            if self._matches(window):
                return window
            if depth < 2:
                stack.extend((w, depth + 1) for w in self._children(window))
        return None

    def attributes(self, window=None):
        info = _XWindowAttributes()
        self._errors = 0
        ok = self._x11.XGetWindowAttributes(
            self._display, window or self.window, ctypes.byref(info)
        )
        if not ok or self._errors:
            raise CaptureError("the game window is gone")
        return info

    def wait_for_window(self, timeout=30.0, is_alive=None, poll=0.1):
        """Find the game window, waiting for it to be mapped."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if is_alive is not None and not is_alive():
                raise CaptureError("the game stopped before its window appeared")
            window = self.find_window()
            if window is not None:
                info = self.attributes(window)
                if info.map_state == _IS_VIEWABLE and info.width > 1:
                    self.window = window
                    return window
            time.sleep(poll)
        raise CaptureError(
            f"no window of class {self.window_class.decode()} appeared in "
            f"{timeout:.0f} s on display {self.display_name}"
        )

    def size(self):
        info = self.attributes()
        return info.width, info.height

    def grab(self, x=0, y=0, width=None, height=None):
        """Pixels of a region of the window as an `(H, W, 3)` uint8 RGB array."""
        info = self.attributes()
        width = info.width - x if width is None else width
        height = info.height - y if height is None else height
        if self._shm is not None:
            self._errors = 0
            rgb = self._shm.grab(self.window, info, x, y, width, height)
            if rgb is not None and not self._errors:
                return rgb
            self._shm = None  # not usable here, use plain XGetImage from now on
        self._errors = 0
        image = self._x11.XGetImage(
            self._display, self.window, x, y, width, height, _ALL_PLANES, _Z_PIXMAP
        )
        if not image or self._errors:
            raise CaptureError("XGetImage failed (window moved off screen or closed)")
        try:
            return _ximage_to_rgb(image.contents)
        finally:
            self._x11.XDestroyImage(image)

    def close(self):
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self._display:
            self._x11.XCloseDisplay.argtypes = (ctypes.c_void_p,)
            self._x11.XCloseDisplay(self._display)
            self._display = None


def ffmpeg_grab(x, y, width, height, display=None):
    """One screen region through `ffmpeg -f x11grab` (slower, used for comparison)."""
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "x11grab",
        "-draw_mouse",
        "0",
        "-video_size",
        f"{width}x{height}",
        "-i",
        f"{display or os.environ.get('DISPLAY')}+{x},{y}",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    out = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(height, width, 3)


class FrameCapture:
    """
    Turn-boundary frame capture of the Toribash window.

    `grab()` returns a `(size, size, 3)` uint8 frame, see `process_frame` for
    `crop`. `display` is the X display of the game (default: `$DISPLAY`); the
    window and its geometry are looked up on that display. The first call finds
    the window. Frames that are one colour
    (black window, lock screen, minimised game) trigger a warning, once per
    run of bad frames.
    """

    def __init__(
        self,
        size=256,
        crop="center",
        window_class=b"toribash_steam",
        startup_timeout=5.0,
        startup_settle=0.3,
        display=None,
    ):
        self.startup_settle = startup_settle
        self.display = display
        self.size = size
        self.startup_timeout = startup_timeout
        self._seen_content = False
        self.crop = crop
        self.window_class = window_class
        self._grabber = None
        self._warned = False
        self.last_frame = None

    def start(self, is_alive=None, timeout=30.0):
        """Open the display and find the window (call once the game runs)."""
        if self._grabber is None:
            self._grabber = X11WindowGrabber(self.window_class, self.display)
            self._grabber.wait_for_window(timeout, is_alive)
        return self

    def _grab_once(self):
        grabber = self._grabber
        width, height = grabber.size()
        if self.crop == "center":
            x, y, side = square_crop_box(width, height)
            image = grabber.grab(x, y, side, side)
            return process_frame(image, self.size, "center")
        return process_frame(grabber.grab(), self.size, "letterbox")

    def grab(self):
        """
        The current frame. Until the window has shown something once, a flat
        frame is not reported: the game has not drawn yet (the first state of
        a launch is read before the first frame), so the grab is retried for up
        to `startup_timeout` seconds.
        """
        if self._grabber is None:
            self.start()
        frame = self._grab_once()
        if not self._seen_content:
            deadline = time.monotonic() + self.startup_timeout
            while is_flat_frame(frame) and time.monotonic() < deadline:
                time.sleep(0.05)
                frame = self._grab_once()
            if not is_flat_frame(frame):
                # The first picture after a launch may predate the camera and
                # option changes made by the controller script.
                time.sleep(self.startup_settle)
                frame = self._grab_once()
        if is_flat_frame(frame):
            if not self._warned:
                warnings.warn(
                    "The captured frame is a single colour. The Toribash window "
                    "may be hidden, minimised or not drawn yet, so RGB "
                    "observations are not showing the game.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._warned = True
        else:
            self._warned = False
            self._seen_content = True
        self.last_frame = frame
        return frame

    def close(self):
        if self._grabber is not None:
            self._grabber.close()
            self._grabber = None
