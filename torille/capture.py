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
import os
import subprocess
import warnings

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


def _area_weights(source, target):
    """`(target, source)` matrix averaging `source` samples into `target` bins."""
    edges = np.linspace(0.0, source, target + 1)
    pixels = np.arange(source)
    low = np.maximum(edges[:-1, None], pixels[None, :])
    high = np.minimum(edges[1:, None], pixels[None, :] + 1)
    weights = np.clip(high - low, 0.0, None)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


_WEIGHT_CACHE = {}


def _cached_weights(source, target):
    key = (source, target)
    if key not in _WEIGHT_CACHE:
        _WEIGHT_CACHE[key] = _area_weights(source, target)
    return _WEIGHT_CACHE[key]


def resize_area(image, out_h, out_w):
    """
    Resize an `(H, W, C)` uint8 image to `(out_h, out_w, C)` by area averaging
    (exact box filter, anti-aliased when shrinking). Pure numpy.
    """
    height, width = image.shape[:2]
    if (height, width) == (out_h, out_w):
        return np.ascontiguousarray(image)
    rows = _cached_weights(height, out_h)
    cols = _cached_weights(width, out_w)
    # Contract the longer axis first to keep the intermediate small.
    tmp = np.tensordot(rows, image.astype(np.float32), axes=(1, 0))
    out = np.tensordot(tmp, cols, axes=(1, 1)).transpose(0, 2, 1)
    return np.rint(out).clip(0, 255).astype(np.uint8)


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


_Z_PIXMAP = 2
_ALL_PLANES = ctypes.c_ulong(~0)
_IS_VIEWABLE = 2


class CaptureError(RuntimeError):
    """The window could not be found or read."""


class X11WindowGrabber:
    """
    Grab the pixels of a window with `XGetImage`.

    The window is the one with the WM class `window_class` (the Steam build of
    Toribash is `toribash_steam`, found by scanning the window tree). The
    request is made on the window itself, not on the screen: with a compositing
    window manager (GNOME/mutter) windows hidden behind other windows are
    still returned correctly, see docs/envs.md for what was verified.
    """

    def __init__(self, window_class=b"toribash_steam", display=None):
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
        self._display = x11.XOpenDisplay(name.encode())
        if not self._display:
            raise CaptureError(f"cannot open the X display {name!r}")
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
        import time

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
            f"{timeout:.0f} s on display {os.environ.get('DISPLAY')}"
        )

    def size(self):
        info = self.attributes()
        return info.width, info.height

    def grab(self, x=0, y=0, width=None, height=None):
        """Pixels of a region of the window as an `(H, W, 3)` uint8 RGB array."""
        info = self.attributes()
        width = info.width - x if width is None else width
        height = info.height - y if height is None else height
        self._errors = 0
        image = self._x11.XGetImage(
            self._display, self.window, x, y, width, height, _ALL_PLANES, _Z_PIXMAP
        )
        if not image or self._errors:
            raise CaptureError("XGetImage failed (window moved off screen or closed)")
        try:
            raw = image.contents
            if raw.bits_per_pixel != 32:
                raise CaptureError(f"unsupported pixel depth {raw.bits_per_pixel}")
            buffer = (ctypes.c_ubyte * (raw.bytes_per_line * raw.height)).from_address(
                raw.data
            )
            pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(
                raw.height, raw.bytes_per_line // 4, 4
            )
            # Little-endian ZPixmap is B, G, R, X.
            rgb = pixels[:, : raw.width, 2::-1].copy()
        finally:
            self._x11.XDestroyImage(image)
        return rgb

    def close(self):
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
    `crop`. The first call finds the window. Frames that are one colour
    (black window, lock screen, minimised game) trigger a warning, once per
    run of bad frames.
    """

    def __init__(self, size=256, crop="center", window_class=b"toribash_steam"):
        self.size = size
        self.crop = crop
        self.window_class = window_class
        self._grabber = None
        self._warned = False
        self.last_frame = None

    def start(self, is_alive=None, timeout=30.0):
        """Open the display and find the window (call once the game runs)."""
        if self._grabber is None:
            self._grabber = X11WindowGrabber(self.window_class)
            self._grabber.wait_for_window(timeout, is_alive)
        return self

    def grab(self):
        if self._grabber is None:
            self.start()
        grabber = self._grabber
        width, height = grabber.size()
        if self.crop == "center":
            x, y, side = square_crop_box(width, height)
            image = grabber.grab(x, y, side, side)
            frame = process_frame(image, self.size, "center")
        else:
            frame = process_frame(grabber.grab(), self.size, "letterbox")
        if is_flat_frame(frame):
            if not self._warned:
                warnings.warn(
                    "The captured frame is a single colour. The Toribash window "
                    "may be hidden, minimised or the screen locked, so RGB "
                    "observations are not showing the game.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._warned = True
        else:
            self._warned = False
        self.last_frame = frame
        return frame

    def close(self):
        if self._grabber is not None:
            self._grabber.close()
            self._grabber = None
