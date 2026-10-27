"""Several Toribash instances at once: copies of the game directory, one Xvfb each.

`steam -applaunch` starts one game per machine. With `launch_mode="direct"`
(see `ToribashControl`) every instance can run from its own copy of the game
directory. `InstancePool` makes those copies and the virtual X servers and
removes them again:

    with InstancePool(4) as pool:
        envs = [TwoPlayerToriEnv(draw_game=True, **kw) for kw in pool]

The Steam client must be running (it only provides Steamworks IPC). Pure
standard library.
"""

import atexit
import errno
import fcntl
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import warnings
import weakref
from dataclasses import dataclass
from pathlib import Path

from . import constants
from .torille import _cache_directory, _pid_alive

# Linux FICLONE ioctl: make `dst` a copy-on-write clone of `src`
_FICLONE = 0x40049409
_MARKER = ".torille-pool"
_X11_SOCKETS = Path("/tmp/.X11-unix")
_XVFB_SCREEN = "1280x720x24"
# Free disk space needed, as a multiple of the copies' size
DISK_SAFETY_FACTOR = 3
_IGNORED = (".torille", ".launchlock")


def default_base_dir():
    return _cache_directory() / "instances"


def directory_size(path, ignore=()):
    """Sum of the file sizes under `path` in bytes (symlinks are not followed)."""
    total = 0
    for root, directories, files in os.walk(path):
        directories[:] = [d for d in directories if d not in ignore]
        for name in files:
            if name in ignore:
                continue
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def display_is_taken(number, x11_sockets=_X11_SOCKETS, lock_directory=Path("/tmp")):
    """True if an X server seems to own display `number`."""
    return (x11_sockets / f"X{number}").exists() or (
        lock_directory / f".X{number}-lock"
    ).exists()


def find_free_display(start=100, is_taken=display_is_taken, exclude=(), limit=1000):
    """First display number from `start` that is not taken (and not in `exclude`)."""
    for number in range(start, start + limit):
        if number not in exclude and not is_taken(number):
            return number
    raise RuntimeError(f"no free X display number in {start}..{start + limit - 1}")


def _stop_group(popen, timeout=3.0):
    """Terminate the process group started by `popen` (SIGTERM, then SIGKILL)."""
    if popen.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, None)):
        try:
            os.killpg(popen.pid, sig)
        except ProcessLookupError:
            return
        try:
            popen.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


def start_xvfb(
    start=100,
    screen=_XVFB_SCREEN,
    exclude=(),
    attempts=10,
    timeout=10.0,
    is_taken=display_is_taken,
    command="Xvfb",
):
    """Start an Xvfb on a free display. Returns `(popen, display_name)`."""
    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError(
            "Xvfb is not installed (needed for display='xvfb'); install it or "
            "pass an existing display such as display=':0'"
        )
    tried = set(exclude)
    for _ in range(attempts):
        number = find_free_display(start, is_taken, tried)
        tried.add(number)
        popen = subprocess.Popen(
            (executable, f":{number}", "-screen", "0", screen, "-nolisten", "tcp"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if popen.poll() is not None:
                break  # lost a race for this number, try the next one
            if (_X11_SOCKETS / f"X{number}").exists():
                return popen, f":{number}"
            time.sleep(0.02)
        else:
            _stop_group(popen)
            raise RuntimeError(f"Xvfb :{number} did not start in {timeout:.0f} s")
    raise RuntimeError(f"Xvfb did not start after {attempts} display numbers")


class _Cloner:
    """Copy files, as reflinks when the filesystem has them, else plain copies."""

    def __init__(self):
        self.reflink = True
        self.reflinked = 0
        self.copied = 0

    def __call__(self, source, destination):
        if self.reflink:
            try:
                with open(source, "rb") as src, open(destination, "wb") as dst:
                    fcntl.ioctl(dst.fileno(), _FICLONE, src.fileno())
            except OSError as error:
                Path(destination).unlink(missing_ok=True)
                if error.errno not in (
                    errno.EOPNOTSUPP,
                    errno.EXDEV,
                    errno.EINVAL,
                    errno.ENOTTY,
                ):
                    raise
                self.reflink = False  # not supported here, do not retry
            else:
                shutil.copystat(source, destination)
                self.reflinked += 1
                return destination
        shutil.copy2(source, destination)
        self.copied += 1
        return destination


def copy_game_directory(source, destination, cloner=None):
    """Copy a game directory (without the bridge and lock files of a running game)."""
    cloner = cloner or _Cloner()
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(*_IGNORED),
        copy_function=cloner,
    )
    return cloner


def processes_running_from(root, proc=Path("/proc")):
    """PIDs of processes whose executable (argv[0]) lives under `root`."""
    root = os.path.realpath(root) + os.sep
    pids = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0", 1)[0]
        except OSError:
            continue
        if command and (os.path.realpath(os.fsdecode(command)) + os.sep).startswith(
            root
        ):
            pids.append(int(entry.name))
    return pids


def _terminate_pids(pids, timeout=3.0):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError, PermissionError:
                pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and any(map(_pid_alive, pids)):
            time.sleep(0.05)
        pids = [pid for pid in pids if _pid_alive(pid)]
        if not pids:
            return


def remove_stale_pools(base_dir=None):
    """Remove pool directories of Python processes that died without cleaning up.

    Only directories with a pool marker whose owner PID is gone are removed.
    Returns the removed paths.
    """
    base_dir = Path(base_dir or default_base_dir())
    removed = []
    if not base_dir.is_dir():
        return removed
    for pool_directory in base_dir.iterdir():
        try:
            owner = json.loads((pool_directory / _MARKER).read_text())["pid"]
        except OSError, ValueError, KeyError:
            continue
        if _pid_alive(owner):
            continue
        _terminate_pids(processes_running_from(pool_directory))
        shutil.rmtree(pool_directory, ignore_errors=True)
        removed.append(pool_directory)
    return removed


@dataclass
class Instance:
    """One instance of the pool: a game directory and a display."""

    game_dir: str
    display: str
    capture_display: str
    xvfb: subprocess.Popen | None = None

    @property
    def kwargs(self):
        """Keyword arguments for `ToriEnv`, `TwoPlayerToriEnv` or `gym.make`."""
        return {
            "launch_mode": "direct",
            "game_dir": self.game_dir,
            "display": self.display,
            "capture_display": self.capture_display,
        }


def _close_pool(pool_ref):
    pool = pool_ref()
    if pool is not None:
        pool.close()


class InstancePool:
    """N game directories (copies of the installed game) and displays.

    Parameters:
        n: Number of instances.
        base_dir: Where the copies live (default `~/.cache/torille/instances`,
            or `$XDG_CACHE_HOME/torille/instances`). Each pool makes its own
            subdirectory there and removes only that.
        display: `"xvfb"` (default): one Xvfb (1280x720x24) per instance on
            a free display number, so each game window is alone on its
            display. Or an existing display such as `":0"`: all instances
            are put on it (windows land on that desktop, and with n > 1 the
            RGB capture cannot tell the windows apart).
        source: Game directory to copy (default: the Steam install).

    Use it as a context manager (or call `start()` and `close()`); the copies,
    Xvfb processes and any game still running from a copy are removed at exit
    and with `close()`. Iterating gives one keyword dict per instance.
    """

    def __init__(self, n, base_dir=None, display="xvfb", source=None):
        if n < 1:
            raise ValueError("n must be at least 1")
        if display != "xvfb" and not str(display).startswith(":"):
            raise ValueError("display must be 'xvfb' or an X display like ':0'")
        self.n = n
        self.display = display
        self.base_dir = Path(base_dir) if base_dir else default_base_dir()
        self.source = Path(source or Path(constants.get_platform_executable()).parent)
        self.root = None
        self.instances = []
        self.disk_bytes = 0
        self.copy_method = None
        self._closed = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *_exc):
        self.close()

    def __iter__(self):
        return iter(i.kwargs for i in self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index].kwargs

    def required_bytes(self):
        """Disk needed for all copies."""
        return directory_size(self.source, _IGNORED) * self.n

    def check_disk_space(self):
        need = self.required_bytes()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.base_dir).free
        if free < DISK_SAFETY_FACTOR * need:
            raise RuntimeError(
                f"not enough disk space in {self.base_dir}: {self.n} copies need "
                f"{need / 1e9:.1f} GB and {DISK_SAFETY_FACTOR}x that must be free, "
                f"{free / 1e9:.1f} GB are"
            )
        return need

    def start(self):
        if self.root is not None:
            raise RuntimeError("the pool is already started")
        if not (self.source / "toribash_steam").is_file():
            raise FileNotFoundError(f"no Toribash game in {self.source}")
        self.check_disk_space()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(_close_pool, weakref.ref(self))
        self.root = Path(tempfile.mkdtemp(prefix="pool-", dir=self.base_dir))
        (self.root / _MARKER).write_text(json.dumps({"pid": os.getpid()}))
        try:
            cloner = _Cloner()
            for index in range(self.n):
                target = self.root / f"instance-{index}"
                copy_game_directory(self.source, target, cloner)
                self.instances.append(Instance(str(target), "", ""))
            self.copy_method = "reflink" if cloner.reflinked else "copy"
            self.disk_bytes = directory_size(self.root, (_MARKER,))
            if self.display == "xvfb":
                taken = set()
                for instance in self.instances:
                    popen, name = start_xvfb(exclude=taken)
                    taken.add(int(name[1:]))
                    instance.xvfb = popen
                    instance.display = instance.capture_display = name
            else:
                if self.n > 1:
                    warnings.warn(
                        f"{self.n} instances share display {self.display}: "
                        "RGB capture finds the window by class and cannot tell "
                        "them apart. Use display='xvfb' for capture.",
                        stacklevel=2,
                    )
                for instance in self.instances:
                    instance.display = instance.capture_display = self.display
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        """Stop games and Xvfb processes of the pool and delete the copies."""
        if self._closed or self.root is None:
            return
        self._closed = True
        root = self.root
        try:
            # Games still running from a copy would lose their files
            _terminate_pids(processes_running_from(root))
            for instance in self.instances:
                if instance.xvfb is not None:
                    _stop_group(instance.xvfb)
                    instance.xvfb = None
        finally:
            if (root / _MARKER).is_file():
                shutil.rmtree(root, ignore_errors=True)
