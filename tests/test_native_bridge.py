import os
import threading
import time
from unittest.mock import patch

import pytest

from torille.torille import NativeFileBridge, NativeSteamProcess


@pytest.fixture
def bridge(tmp_path):
    bridge = NativeFileBridge(tmp_path)
    bridge.prepare()
    return bridge


def test_prepare_clears_stale_files(tmp_path):
    bridge = NativeFileBridge(tmp_path)
    bridge.prepare()
    bridge.state_path.write_text("stale\n")

    bridge.prepare()

    assert not bridge.state_path.exists()


def test_receive_state_returns_complete_message_and_consumes_it(bridge):
    bridge.state_path.write_text("1,2,3\n")

    assert bridge.receive_state(timeout=1) == b"1,2,3\n"
    assert not bridge.state_path.exists()


def test_receive_state_ignores_partially_written_message(bridge):
    bridge.state_path.write_text("1,2,")

    def finish_write():
        time.sleep(0.05)
        bridge.state_path.write_text("1,2,3\n")

    writer = threading.Thread(target=finish_write)
    writer.start()
    try:
        assert bridge.receive_state(timeout=2) == b"1,2,3\n"
    finally:
        writer.join()


def test_receive_state_times_out_without_complete_message(bridge):
    bridge.state_path.write_text("1,2,")

    with pytest.raises(TimeoutError):
        bridge.receive_state(timeout=0.05)


def test_receive_state_fails_fast_when_game_exits(bridge):
    start = time.monotonic()

    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        bridge.receive_state(timeout=30, is_alive=lambda: False)

    assert time.monotonic() - start < 1


def test_receive_state_returns_state_written_just_before_exit(bridge):
    bridge.state_path.write_text("end:1,0\n")

    assert bridge.receive_state(timeout=1, is_alive=lambda: False) == b"end:1,0\n"


def test_steam_process_reports_liveness():
    process = NativeSteamProcess(os.getpid())

    assert process.is_alive()

    with patch("torille.torille.os.kill", side_effect=ProcessLookupError):
        assert not process.is_alive()


@pytest.mark.parametrize("use_inotify", [True, False])
def test_receive_state_wakes_promptly_on_write(tmp_path, use_inotify):
    bridge = NativeFileBridge(tmp_path, poll_interval=0.001, use_inotify=use_inotify)
    bridge.prepare()
    delays = []
    for _ in range(20):
        written = []

        def write(written=written):
            time.sleep(0.02)
            bridge.state_path.write_text("1,2,3\n")
            written.append(time.perf_counter())

        writer = threading.Thread(target=write)
        writer.start()
        assert bridge.receive_state(timeout=2) == b"1,2,3\n"
        received = time.perf_counter()
        writer.join()
        delays.append(received - written[0])
    # A 10 ms sleep would average 5 ms; waiting must not be much slower than that
    assert sorted(delays)[len(delays) // 2] < 0.004
    bridge.cleanup()


def test_receive_state_falls_back_to_polling_without_inotify(tmp_path):
    bridge = NativeFileBridge(tmp_path)
    bridge.prepare()
    with patch("torille.torille._StateWatcher", side_effect=OSError("no inotify")):
        bridge.state_path.write_text("1,2,3\n")
        assert bridge.receive_state(timeout=1) == b"1,2,3\n"
    assert not bridge.use_inotify


def test_state_written_before_waiting_is_not_lost(bridge):
    bridge._get_watcher()
    bridge.state_path.write_text("1,2,3\n")
    # The event is queued; the state is returned without waiting out a timeout
    start = time.monotonic()
    assert bridge.receive_state(timeout=5) == b"1,2,3\n"
    assert time.monotonic() - start < 0.5


def test_settings_file_carries_spin_window(tmp_path):
    bridge = NativeFileBridge(tmp_path, spin_ms=7.5)
    bridge.prepare()
    bridge.write_settings(False, [1, 2])
    assert bridge.settings_path.read_text() == "0\n1,2\n7.5\n"
