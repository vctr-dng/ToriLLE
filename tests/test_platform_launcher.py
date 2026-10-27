import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np

import torille
from torille import constants
from torille.envs import DuoToriEnv, SoloToriEnv, UkeToriEnv
from torille.envs.gym_env import ToriEnv
from torille.torille import ToribashControl


class _TestEnv(ToriEnv):
    def _preprocess_observation(self, state):
        return state

    def _preprocess_action(self, action):
        return action

    def _reward_function(self, old_state, new_state):
        return 1.0


class _State:
    def __init__(self):
        self.limb_positions = np.zeros((2, constants.NUM_LIMBS, 3))
        self.limb_positions[0, 4, 2] = 1000

    def get_normalized_locations(self):
        return np.zeros((2, 2, constants.NUM_LIMBS, 3))


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
    def test_native_controller_is_removed_from_profile_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            game_directory = Path(directory)
            executable = game_directory / "toribash_steam"
            executable.touch()
            script_directory = game_directory / "data" / "script"
            script_directory.mkdir(parents=True)
            profile = game_directory / "profile.tbs"
            profile.write_text("# Existing profile\n", encoding="utf-8")

            controller = ToribashControl(executable=str(executable))
            controller._install_native_controller()

            self.assertTrue((script_directory / "remotecontrol_native.lua").is_file())
            self.assertIn("ls remotecontrol_native.lua", profile.read_text())
            controller._restore_native_profile()
            self.assertEqual(
                profile.read_text(encoding="utf-8"), "# Existing profile\n"
            )

    def test_native_controller_uses_game_root_for_bridge_files(self):
        script = (
            Path(torille.__file__).parent
            / "toribash"
            / "data"
            / "script"
            / "remotecontrol_native.lua"
        ).read_text(encoding="utf-8")

        self.assertIn('io.open(path, "r", true)', script)
        self.assertIn('io.open(path, "w", true)', script)
        self.assertNotIn("os.rename", script)
        self.assertNotIn("os.remove", script)

    def test_linux_launches_native_game_through_steam(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "toribash_steam"
            executable.touch()
            with (
                patch("torille.torille.sys.platform", "linux"),
                patch("torille.torille.utils.check_linux_sanity"),
                patch("torille.torille.utils.set_file_readonly"),
                patch("torille.torille.NativeFileBridge") as bridge,
                patch("torille.torille.shutil.which", return_value="steam"),
                patch.object(
                    ToribashControl,
                    "_native_game_pids",
                    side_effect=[[], [1234]],
                ),
                patch("torille.torille.subprocess.Popen") as popen,
            ):
                controller = ToribashControl(
                    executable=str(executable), install_native_controller=False
                )
                controller.init()

            popen.assert_called_once_with(
                ("steam", "-applaunch", constants.STEAM_APP_ID),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            bridge.return_value.write_settings.assert_called_once()
            controller.process = None

    def test_linux_rejects_an_existing_toribash_game(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "toribash_steam"
            executable.touch()
            with (
                patch("torille.torille.shutil.which", return_value="steam"),
                patch.object(ToribashControl, "_native_game_pids", return_value=[1234]),
                self.assertRaisesRegex(RuntimeError, "already running"),
            ):
                ToribashControl(executable=str(executable))._launch_native_game()


class GymExecutableTests(unittest.TestCase):
    def test_import_registers_toribash_environments(self):
        self.assertIn("Toribash-DestroyUke-v0", gym.envs.registry)

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

    def test_environment_observations_match_declared_spaces(self):
        state = _State()
        with patch.object(ToriEnv, "__init__", return_value=None):
            solo = SoloToriEnv(reward_func=lambda *_: 0)
            uke = UkeToriEnv(reward_func=lambda *_: 0, random_uke=False)
            duo = DuoToriEnv(reward_func=lambda *_: 0)

        for environment in (solo, uke, duo):
            observation = environment._preprocess_observation(state)
            self.assertEqual(observation.dtype, np.float32)
            self.assertTrue(environment.observation_space.contains(observation))
