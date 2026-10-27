import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from torille import constants
from torille.envs.gym_env import ToriEnv
from torille.torille import ToribashControl


class _TestEnv(ToriEnv):
    def _preprocess_observation(self, state):
        return state

    def _preprocess_action(self, action):
        return action

    def _reward_function(self, old_state, new_state):
        return 1.0


class PlatformExecutableTests(unittest.TestCase):
    def test_linux_uses_xdg_steam_install(self):
        with patch.dict(
            constants.os.environ,
            {"XDG_DATA_HOME": "/var/lib/user-data"},
            clear=True,
        ):
            executable = constants.get_platform_executable("linux")

        self.assertEqual(
            executable,
            "/var/lib/user-data/Steam/steamapps/common/Toribash/toribash_steam",
        )

    def test_windows_uses_packaged_executable(self):
        self.assertEqual(
            constants.get_platform_executable("win32"), constants.PACKAGE_TORIBASH_EXE
        )

    def test_unsupported_platform_fails(self):
        with self.assertRaisesRegex(
            RuntimeError, "only supported on Linux and Windows"
        ):
            constants.get_platform_executable("darwin")

    def test_missing_default_executable_explains_override(self):
        with (
            patch.object(
                constants, "get_platform_executable", return_value="/missing/toribash"
            ),
            self.assertRaisesRegex(FileNotFoundError, "pass executable="),
        ):
            ToribashControl()


class NativeLauncherTests(unittest.TestCase):
    def test_linux_launches_native_executable_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "toribash_steam"
            executable.touch()
            connection = MagicMock()
            listener = MagicMock()
            listener.accept.return_value = (connection, ("127.0.0.1", 7788))

            with (
                patch("torille.torille.sys.platform", "linux"),
                patch("torille.torille.utils.check_linux_sanity"),
                patch("torille.torille.utils.set_file_readonly"),
                patch("torille.torille.socket.socket", return_value=listener),
                patch("torille.torille.subprocess.Popen") as popen,
            ):
                controller = ToribashControl(executable=str(executable))
                controller.init()

            popen.assert_called_once_with(
                (str(executable),),
                cwd=str(executable.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            controller.process = None


class GymExecutableTests(unittest.TestCase):
    def test_environment_forwards_executable_to_controller(self):
        with patch("torille.envs.gym_env.torille.ToribashControl") as controller:
            ToriEnv(executable="/opt/toribash_steam", matchframes=100)

        controller.assert_called_once()
        self.assertEqual(
            controller.call_args.kwargs["executable"], "/opt/toribash_steam"
        )
        self.assertEqual(
            controller.call_args.kwargs["settings"].get("matchframes"), 100
        )

    def test_environment_uses_gymnasium_reset_and_step_results(self):
        state = MagicMock()
        with patch("torille.envs.gym_env.torille.ToribashControl") as controller:
            controller.return_value.get_state.return_value = (state, False)
            environment = _TestEnv(executable="/opt/toribash_steam")

            observation, info = environment.reset(seed=3)
            result = environment.step([[1] * 22, [1] * 22])

        self.assertIs(observation, state)
        self.assertEqual(info, {})
        self.assertEqual(result, (state, 1.0, False, False, {}))
