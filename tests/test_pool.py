"""InstancePool, tested without the game (Xvfb tests need Xvfb, not Steam)."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from torille import pool as pool_module
from torille.pool import (
    InstancePool,
    copy_game_directory,
    directory_size,
    find_free_display,
    processes_running_from,
    remove_stale_pools,
    start_xvfb,
)

needs_xvfb = pytest.mark.skipif(shutil.which("Xvfb") is None, reason="no Xvfb")


@pytest.fixture
def source(tmp_path):
    game = tmp_path / "game"
    (game / "data" / "script").mkdir(parents=True)
    (game / "toribash_steam").write_bytes(b"x" * 1000)
    (game / "toribash_steam").chmod(0o755)
    (game / "data" / "model.bin").write_bytes(b"m" * 5000)
    (game / "profile.tbs").write_text("# profile\n")
    (game / ".launchlock").touch()
    (game / ".torille").mkdir()
    (game / ".torille" / "state").write_text("stale")
    return game


def test_find_free_display_skips_taken_numbers():
    taken = {100, 101, 103}
    assert find_free_display(100, is_taken=taken.__contains__) == 102
    assert find_free_display(100, is_taken=taken.__contains__, exclude={102}) == 104


def test_find_free_display_gives_up():
    with pytest.raises(RuntimeError, match="no free X display"):
        find_free_display(5, is_taken=lambda n: True, limit=3)


def test_display_is_taken_looks_at_socket_and_lock(tmp_path):
    sockets, locks = tmp_path / "sockets", tmp_path / "locks"
    sockets.mkdir()
    locks.mkdir()
    assert not pool_module.display_is_taken(7, sockets, locks)
    (sockets / "X7").touch()
    assert pool_module.display_is_taken(7, sockets, locks)
    (sockets / "X7").unlink()
    (locks / ".X7-lock").touch()
    assert pool_module.display_is_taken(7, sockets, locks)


def test_start_xvfb_without_xvfb_installed():
    with pytest.raises(RuntimeError, match="not installed"):
        start_xvfb(command="definitely-not-xvfb")


def test_start_xvfb_retries_the_next_number_when_a_display_is_lost():
    first = MagicMock()
    first.poll.return_value = 1  # exited: somebody else got :100
    second = MagicMock()
    second.poll.return_value = None
    sockets = MagicMock()
    sockets.__truediv__.side_effect = lambda name: MagicMock(
        exists=lambda: name == "X101"
    )
    with (
        patch("torille.pool.shutil.which", return_value="/usr/bin/Xvfb"),
        patch("torille.pool.subprocess.Popen", side_effect=[first, second]) as popen,
        patch("torille.pool._X11_SOCKETS", sockets),
    ):
        process, name = start_xvfb(is_taken=lambda n: False)
    assert process is second
    assert name == ":101"
    commands = [call.args[0] for call in popen.call_args_list]
    assert commands[0][1] == ":100"
    assert commands[1][1] == ":101"
    assert popen.call_args.kwargs["start_new_session"] is True
    assert "1280x720x24" in commands[1]


def test_copy_skips_bridge_and_lock_files_and_keeps_modes(source, tmp_path):
    destination = tmp_path / "copy"
    copy_game_directory(source, destination)
    assert (destination / "data" / "model.bin").read_bytes() == b"m" * 5000
    assert os.access(destination / "toribash_steam", os.X_OK)
    assert not (destination / ".torille").exists()
    assert not (destination / ".launchlock").exists()
    # a real copy: writing to it does not change the source
    (destination / "profile.tbs").write_text("changed")
    assert (source / "profile.tbs").read_text() == "# profile\n"
    assert directory_size(destination) == 1000 + 5000 + len("changed")


def test_pool_creates_copies_displays_and_cleans_up(source, tmp_path):
    base = tmp_path / "instances"
    with patch("torille.pool.start_xvfb") as xvfb:
        xvfb.side_effect = [(MagicMock(), ":120"), (MagicMock(), ":121")]
        with InstancePool(2, base_dir=base, source=source) as pool:
            root = pool.root
            assert root.parent == base
            assert len(pool) == 2
            kwargs = list(pool)
            assert [k["display"] for k in kwargs] == [":120", ":121"]
            assert [k["capture_display"] for k in kwargs] == [":120", ":121"]
            assert all(k["launch_mode"] == "direct" for k in kwargs)
            assert len({k["game_dir"] for k in kwargs}) == 2
            for k in kwargs:
                assert (Path(k["game_dir"]) / "toribash_steam").is_file()
            assert pool[0] == kwargs[0]
            assert pool.disk_bytes == 2 * (1000 + 5000 + 10)
            assert pool.copy_method in ("copy", "reflink")
            instances = list(pool.instances)
        for instance in instances:
            assert instance.xvfb is None
        assert not root.exists()
    # the source and the base directory stay
    assert (source / "toribash_steam").is_file()
    assert base.is_dir()


def test_pool_stops_xvfb_process_groups(source, tmp_path):
    process = MagicMock()
    with (
        patch("torille.pool.start_xvfb", return_value=(process, ":120")),
        patch("torille.pool._stop_group") as stop,
        InstancePool(1, base_dir=tmp_path / "i", source=source),
    ):
        pass
    stop.assert_called_once_with(process)


def test_pool_with_a_given_display_starts_no_xvfb(source, tmp_path):
    with (
        patch("torille.pool.start_xvfb") as xvfb,
        InstancePool(1, base_dir=tmp_path / "i", display=":0", source=source) as pool,
    ):
        assert pool[0]["display"] == ":0"
        assert pool[0]["capture_display"] == ":0"
    xvfb.assert_not_called()


def test_shared_display_with_several_instances_warns(source, tmp_path):
    with (
        pytest.warns(UserWarning, match="share display"),
        InstancePool(2, base_dir=tmp_path / "i", display=":0", source=source),
    ):
        pass


def test_pool_refuses_when_the_disk_is_too_small(source, tmp_path):
    base = tmp_path / "instances"
    need = 2 * (1000 + 5000 + 10)
    usage = shutil.disk_usage(tmp_path)._replace(free=3 * need - 1)
    with (
        patch("torille.pool.shutil.disk_usage", return_value=usage),
        pytest.raises(RuntimeError, match="not enough disk space"),
        InstancePool(2, base_dir=base, source=source),
    ):
        pass
    assert not list(base.iterdir())  # nothing was copied

    usage = usage._replace(free=3 * need)
    with (
        patch("torille.pool.shutil.disk_usage", return_value=usage),
        patch("torille.pool.start_xvfb", return_value=(MagicMock(), ":1")),
        InstancePool(2, base_dir=base, source=source),
    ):
        pass


def test_pool_cleans_up_when_starting_fails(source, tmp_path):
    base = tmp_path / "instances"
    first = MagicMock()
    with (
        patch(
            "torille.pool.start_xvfb",
            side_effect=[(first, ":120"), RuntimeError("no display")],
        ),
        patch("torille.pool._stop_group") as stop,
        pytest.raises(RuntimeError, match="no display"),
        InstancePool(2, base_dir=base, source=source),
    ):
        pass
    assert list(base.iterdir()) == []
    stop.assert_called_once_with(first)


def test_pool_validates_arguments(source, tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        InstancePool(0)
    with pytest.raises(ValueError, match="display"):
        InstancePool(1, display="somewhere")
    with pytest.raises(FileNotFoundError):
        InstancePool(1, base_dir=tmp_path, source=tmp_path / "nothing").start()


def test_processes_running_from_matches_whole_path_components(tmp_path):
    proc = tmp_path / "proc"
    for pid, executable in {
        11: "/a/pool/instance-0/toribash_steam",
        12: "/b/x",
    }.items():
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "cmdline").write_bytes(f"{executable}\0".encode())
    assert processes_running_from("/a/pool", proc) == [11]
    assert processes_running_from("/a/poo", proc) == []  # whole path components


def test_close_kills_a_game_started_from_a_copy(source, tmp_path):
    with patch("torille.pool.start_xvfb", return_value=(MagicMock(), ":1")):
        with InstancePool(1, base_dir=tmp_path / "i", source=source) as pool:
            game = Path(pool[0]["game_dir"])
            # a real process whose argv[0] is a file in the copy
            shutil.copy("/bin/sleep", game / "toribash_steam")
            child = subprocess.Popen(
                [str(game / "toribash_steam"), "60"], start_new_session=True
            )
            time.sleep(0.2)
            assert child.poll() is None
        assert child.wait(timeout=5) != 0
    assert not game.exists()


def test_remove_stale_pools_only_removes_dead_owners(tmp_path):
    base = tmp_path / "instances"
    dead, alive, unmarked = (base / name for name in ("dead", "alive", "unmarked"))
    for directory in (dead, alive, unmarked):
        directory.mkdir(parents=True)
    (dead / ".torille-pool").write_text(json.dumps({"pid": 999999999}))
    (alive / ".torille-pool").write_text(json.dumps({"pid": os.getpid()}))
    removed = remove_stale_pools(base)
    assert removed == [dead]
    assert not dead.exists()
    assert alive.exists()
    assert unmarked.exists()  # not ours: no marker


def test_pool_directory_is_removed_when_python_exits(source, tmp_path):
    base = tmp_path / "instances"
    script = f"""
from unittest.mock import MagicMock, patch
from torille.pool import InstancePool
with patch("torille.pool.start_xvfb", return_value=(MagicMock(), ":1")):
    pool = InstancePool(1, base_dir={str(base)!r}, source={str(source)!r})
    pool.start()
print(pool.root, flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not Path(result.stdout.strip()).exists()


@needs_xvfb
def test_real_xvfb_start_and_stop():
    process, name = start_xvfb(start=300)
    try:
        assert process.poll() is None
        assert Path(f"/tmp/.X11-unix/X{name[1:]}").exists()
        # a second server gets another number
        other, other_name = start_xvfb(start=300)
        assert other_name != name
        pool_module._stop_group(other)
    finally:
        pool_module._stop_group(process)
    assert process.poll() is not None
    time.sleep(0.2)
    assert not Path(f"/tmp/.X11-unix/X{name[1:]}").exists()
