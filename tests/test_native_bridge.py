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
