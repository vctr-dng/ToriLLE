"""Direct launch mode (`launch_mode="direct"`), tested without the game."""

import os
import signal
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest
from filelock import FileLock, Timeout

from torille import torille as tl
from torille.torille import (
    NativeDirectProcess,
    SharedOptionSnapshot,
    ToribashControl,
    steam_client_running,
)


def make_game(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "toribash_steam").touch()
    (path / "data" / "script").mkdir(parents=True)
    (path / "profile.tbs").write_text("# profile\n", encoding="utf-8")
    return path


@pytest.fixture
def game(tmp_path):
    return make_game(tmp_path / "game")


def test_default_mode_is_steam(game):
    control = ToribashControl(executable=str(game / "toribash_steam"))
    assert control.launch_mode == "steam"
    assert control.display is None


def test_game_dir_sets_the_executable_and_lock_next_to_it(game):
    control = ToribashControl(game_dir=str(game), launch_mode="direct")
    assert control.executable_path == str(game / "toribash_steam")
    assert control.lock_file == str(game / ".launchlock")


def test_conflicting_executable_and_game_dir(tmp_path, game):
    other = make_game(tmp_path / "other")
    with pytest.raises(ValueError, match="different games"):
        ToribashControl(executable=str(other / "toribash_steam"), game_dir=str(game))


def test_unknown_launch_mode(game):
    with pytest.raises(ValueError, match="launch_mode"):
        ToribashControl(game_dir=str(game), launch_mode="docker")


def test_display_needs_direct_mode(game):
    with pytest.raises(ValueError, match="display"):
        ToribashControl(game_dir=str(game), display=":5")


def test_capture_display_defaults_to_the_game_display(game):
    direct = ToribashControl(game_dir=str(game), launch_mode="direct", display=":7")
    assert direct.capture_display == ":7"
    explicit = ToribashControl(
        game_dir=str(game), launch_mode="direct", display=":7", capture_display=":8"
    )
    assert explicit.capture_display == ":8"
    assert ToribashControl(game_dir=str(game)).capture_display is None


def test_launch_locks_are_per_game_directory(tmp_path):
    first = ToribashControl(game_dir=str(make_game(tmp_path / "a")))
    second = ToribashControl(game_dir=str(make_game(tmp_path / "b")))
    assert first.lock_file != second.lock_file
    with FileLock(first.lock_file, timeout=1):
        # another copy is not blocked...
        with FileLock(second.lock_file, timeout=0.2):
            pass
        # ...the same directory is
        with pytest.raises(Timeout), FileLock(first.lock_file, timeout=0.2):
            pass


def fake_proc(tmp_path, pids_by_executable):
    """A fake /proc with processes started from the given executables."""
    root = tmp_path / "proc"
    for pid, executable in pids_by_executable.items():
        (root / str(pid)).mkdir(parents=True)
        (root / str(pid) / "cmdline").write_bytes(f"{executable}\0arg".encode())
    return root


def test_running_check_is_per_executable(tmp_path):
    first = ToribashControl(game_dir=str(make_game(tmp_path / "a")))
    second = ToribashControl(game_dir=str(make_game(tmp_path / "b")))
    proc = fake_proc(tmp_path, {41: first.executable_path})
    real_path = tl.Path

    def path_factory(*args):
        if args == ("/proc",):
            return proc
        return real_path(*args)

    with patch("torille.torille.Path", side_effect=path_factory):
        assert first._native_game_pids() == [41]
        assert second._native_game_pids() == []


def test_direct_launch_needs_steam(game):
    control = ToribashControl(game_dir=str(game), launch_mode="direct")
    with (
        patch("torille.torille.steam_client_running", return_value=False),
        patch("torille.torille.subprocess.Popen") as popen,
        pytest.raises(RuntimeError, match="Steam client"),
    ):
        control._launch_direct_game()
    popen.assert_not_called()


def test_direct_launch_command_and_environment(game):
    control = ToribashControl(game_dir=str(game), launch_mode="direct", display=":42")
    with (
        patch("torille.torille.steam_client_running", return_value=True),
        patch("torille.torille.subprocess.Popen") as popen,
    ):
        process = control._launch_direct_game()
    args, kwargs = popen.call_args
    assert args == ((str(game / "toribash_steam"),),)
    assert kwargs["cwd"] == str(game)
    assert kwargs["start_new_session"] is True
    environment = kwargs["env"]
    assert environment["DISPLAY"] == ":42"
    assert environment["SteamAppId"] == "248570"
    assert environment["SteamGameId"] == "248570"
    assert isinstance(process, NativeDirectProcess)


def test_direct_launch_does_not_call_steam(game):
    control = ToribashControl(game_dir=str(game), launch_mode="direct")
    with (
        patch("torille.torille.steam_client_running", return_value=True),
        patch("torille.torille.subprocess.Popen") as popen,
        patch("torille.torille.shutil.which") as which,
    ):
        control._launch_direct_game()
    which.assert_not_called()
    assert "applaunch" not in str(popen.call_args)


def test_direct_launch_refuses_when_this_copy_is_running(game):
    control = ToribashControl(game_dir=str(game), launch_mode="direct")
    with (
        patch("torille.torille.steam_client_running", return_value=True),
        patch.object(ToribashControl, "_native_game_pids", return_value=[5]),
        patch("torille.torille.subprocess.Popen") as popen,
        pytest.raises(RuntimeError, match="already running"),
    ):
        control._launch_direct_game()
    popen.assert_not_called()


def init_direct(game, **launch):
    with (
        patch("torille.torille.sys.platform", "linux"),
        patch("torille.torille.utils.check_linux_sanity") as sanity,
        patch.object(ToribashControl, "_launch_direct_game", **launch),
        patch.object(ToribashControl, "_launch_native_game") as steam_launch,
    ):
        control = ToribashControl(
            game_dir=str(game), launch_mode="direct", display=":9"
        )
        control.init()
    steam_launch.assert_not_called()
    sanity.assert_called_once_with(":9")
    return control


def assert_clean(game):
    assert (game / "profile.tbs").read_text() == "# profile\n"
    assert not (game / "data" / "script" / "remotecontrol_native.lua").exists()
    assert not (game / ".torille").exists()


def test_close_kills_the_process_and_cleans_the_copy(game):
    process = MagicMock()
    control = init_direct(game, return_value=process)
    assert (game / ".torille").is_dir()
    control.close()
    process.kill.assert_called_once()
    assert_clean(game)


def test_failed_direct_launch_cleans_the_copy(game):
    with pytest.raises(RuntimeError, match="Steam"):
        init_direct(game, side_effect=RuntimeError("Steam client"))
    assert_clean(game)


def test_direct_game_is_killed_when_python_exits(game):
    script = f"""
import sys
from unittest.mock import patch
from torille.torille import ToribashControl, NativeDirectProcess
import subprocess
ToribashControl.__del__ = lambda self: None
game = {str(game)!r}
sleeper = subprocess.Popen(
    ["sleep", "60"], start_new_session=True, stdout=subprocess.DEVNULL
)
print(sleeper.pid, flush=True)
with (
    patch("torille.torille.sys.platform", "linux"),
    patch("torille.torille.utils.check_linux_sanity"),
    patch.object(
        ToribashControl, "_launch_direct_game",
        return_value=NativeDirectProcess(sleeper),
    ),
):
    controller = ToribashControl(game_dir=game, launch_mode="direct", display=":9")
    controller.init()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    pid = int(result.stdout.split()[0])
    time.sleep(0.2)
    # killed by the exit handler, and reaped (it was our child)
    assert not os.path.exists(f"/proc/{pid}")
    assert_clean(game)


def test_native_direct_process_terminates_the_whole_group():
    popen = subprocess.Popen(["sh", "-c", "sleep 60 & wait"], start_new_session=True)
    process = NativeDirectProcess(popen)
    child = None
    for _ in range(50):
        children = subprocess.run(
            ["pgrep", "-g", str(popen.pid), "sleep"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.split()
        if children:
            child = int(children[0])
            break
        time.sleep(0.05)
    assert child is not None
    assert process.is_alive()
    process.kill()
    assert not process.is_alive()
    time.sleep(0.1)
    assert not os.path.exists(f"/proc/{child}")
    process.kill()  # twice is fine


def test_native_direct_process_escalates_to_sigkill():
    popen = subprocess.Popen(
        ["sh", "-c", "trap '' TERM; while true; do sleep 1; done"],
        start_new_session=True,
    )
    process = NativeDirectProcess(popen)
    process.TERMINATE_TIMEOUT = 0.3
    time.sleep(0.2)
    process.kill()
    assert popen.returncode == -signal.SIGKILL


def test_exit_message_mentions_steam():
    popen = MagicMock()
    popen.poll.return_value = 1
    assert "Steam" in NativeDirectProcess(popen).exit_message()
    popen.poll.return_value = None
    assert NativeDirectProcess(popen).exit_message() == ""


def test_game_exit_error_names_the_likely_cause(game):
    control = ToribashControl(game_dir=str(game), launch_mode="direct")
    popen = MagicMock()
    popen.poll.return_value = 1
    control.process = NativeDirectProcess(popen)
    control.native_bridge = MagicMock()
    control.native_bridge.receive_state.side_effect = RuntimeError(
        "Native Toribash exited unexpectedly"
    )
    with pytest.raises(RuntimeError, match="exited unexpectedly.*Steam"):
        control._recv_line(None)
    control.process = None


def test_steam_client_running(tmp_path):
    home = tmp_path / "home"
    (home / ".steam").mkdir(parents=True)
    with patch("torille.torille.Path.home", return_value=home):
        assert not steam_client_running()  # no pid file
        (home / ".steam" / "steam.pid").write_text("999999999\n")
        assert not steam_client_running()  # stale pid
        (home / ".steam" / "steam.pid").write_text(f"{os.getpid()}\n")
        assert not steam_client_running()  # a live process that is not steam


CONFIG = 'jointflash "1"\nother "3"\n'


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "custom.cfg"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def make_snapshot(config, tmp_path):
    return SharedOptionSnapshot(
        [config], ("jointflash",), state_path=tmp_path / "state" / "options.json"
    )


def test_shared_snapshot_restores_when_the_last_holder_leaves(config, tmp_path):
    first = make_snapshot(config, tmp_path)
    second = make_snapshot(config, tmp_path)
    first.acquire()
    config.write_text('jointflash "0"\nother "3"\n', encoding="utf-8")
    # The second holder must not snapshot the changed value
    second.acquire()
    first.release()
    assert 'jointflash "0"' in config.read_text()  # second still running
    second.release()
    assert config.read_text() == CONFIG
    assert not (tmp_path / "state" / "options.json").exists()


def test_shared_snapshot_recovers_from_a_crashed_run(config, tmp_path):
    crashed = make_snapshot(config, tmp_path)
    crashed.acquire()
    config.write_text('jointflash "0"\nother "3"\n', encoding="utf-8")
    state = tmp_path / "state" / "options.json"
    # The holder died without releasing
    text = state.read_text().replace(str(os.getpid()), "999999999")
    state.write_text(text)
    fresh = make_snapshot(config, tmp_path)
    fresh.acquire()
    assert config.read_text() == CONFIG  # restored before the new snapshot
    fresh.release()
    assert config.read_text() == CONFIG


def test_capture_in_direct_mode_uses_the_shared_snapshot(game, tmp_path):
    with (
        patch("torille.torille.sys.platform", "linux"),
        patch("torille.torille.utils.check_linux_sanity"),
        patch("torille.torille._cache_directory", return_value=tmp_path / "cache"),
        patch.object(ToribashControl, "_launch_direct_game", return_value=MagicMock()),
    ):
        control = ToribashControl(
            game_dir=str(game),
            launch_mode="direct",
            display=":9",
            draw_game=True,
            capture_size=64,
        )
        control.init()
        assert isinstance(control._option_snapshot, SharedOptionSnapshot)
        assert (tmp_path / "cache" / "options.json").exists()
        control.close()
    assert not (tmp_path / "cache" / "options.json").exists()
