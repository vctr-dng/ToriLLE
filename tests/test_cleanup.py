import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from torille.torille import ToribashControl

ORIGINAL_PROFILE = "# Existing profile\n"


@pytest.fixture
def game_directory(tmp_path):
    (tmp_path / "toribash_steam").touch()
    (tmp_path / "data" / "script").mkdir(parents=True)
    (tmp_path / "profile.tbs").write_text(ORIGINAL_PROFILE, encoding="utf-8")
    return tmp_path


def assert_game_directory_restored(game_directory):
    assert (game_directory / "profile.tbs").read_text() == ORIGINAL_PROFILE
    assert not (
        game_directory / "data" / "script" / "remotecontrol_native.lua"
    ).exists()
    assert not (game_directory / ".torille").exists()


def init_controller(game_directory, **launch):
    with (
        patch("torille.torille.sys.platform", "linux"),
        patch("torille.torille.utils.check_linux_sanity"),
        patch.object(ToribashControl, "_launch_native_game", **launch),
    ):
        controller = ToribashControl(executable=str(game_directory / "toribash_steam"))
        controller.init()
    return controller


def test_init_installs_controller_files(game_directory):
    process = MagicMock()
    controller = init_controller(game_directory, return_value=process)

    assert (game_directory / "data" / "script" / "remotecontrol_native.lua").is_file()
    assert (game_directory / ".torille").is_dir()
    controller.close()


def test_close_kills_game_and_restores_game_directory(game_directory):
    process = MagicMock()
    controller = init_controller(game_directory, return_value=process)

    controller.close()

    process.kill.assert_called_once()
    assert_game_directory_restored(game_directory)


def test_failed_launch_restores_game_directory(game_directory):
    with pytest.raises(TimeoutError):
        init_controller(game_directory, side_effect=TimeoutError("no game"))

    assert_game_directory_restored(game_directory)


def test_cleanup_can_run_more_than_once(game_directory):
    process = MagicMock()
    controller = init_controller(game_directory, return_value=process)

    controller._cleanup()
    controller._cleanup()

    process.kill.assert_called_once()
    assert_game_directory_restored(game_directory)


def test_game_directory_is_restored_when_python_exits_without_close(game_directory):
    killed_marker = game_directory / "killed"
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from unittest.mock import patch

        from torille.torille import ToribashControl

        game_directory = Path(sys.argv[1])
        # __del__ is not guaranteed to run at shutdown, so only atexit may clean up
        ToribashControl.__del__ = lambda self: None

        class Process:
            def kill(self):
                (game_directory / "killed").touch()

        with (
            patch("torille.torille.sys.platform", "linux"),
            patch("torille.torille.utils.check_linux_sanity"),
            patch.object(
                ToribashControl, "_launch_native_game", return_value=Process()
            ),
        ):
            controller = ToribashControl(
                executable=str(game_directory / "toribash_steam")
            )
            controller.init()
        # Exit without calling close()
        """
    )

    subprocess.run(
        [sys.executable, "-I", "-c", script, str(game_directory)], check=True
    )

    assert killed_marker.exists()
    assert_game_directory_restored(game_directory)
