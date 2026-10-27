"""Plays real games in native Toribash. Run with `pytest -m native`."""

import os
import shutil
from pathlib import Path

import gymnasium as gym
import pytest

import torille.envs  # noqa: F401
from torille import constants

pytestmark = pytest.mark.native


@pytest.fixture(scope="module", autouse=True)
def _require_native_toribash():
    executable = Path(constants.get_platform_executable())
    if not executable.is_file():
        pytest.skip(f"Toribash is not installed at {executable}")
    if shutil.which("steam") is None:
        pytest.skip("Steam is not available")
    if not os.environ.get("DISPLAY"):
        pytest.skip("No DISPLAY for Toribash's OpenGL renderer")


@pytest.mark.parametrize("draw_game", [True, False])
def test_episodes_end_by_truncation_and_env_can_reset(draw_game):
    env = gym.make("Toribash-DestroyUke-v0", matchframes=100, turnframes=5)
    env.unwrapped.set_draw_game(draw_game)
    try:
        observation, _ = env.reset()
        # Play more than one episode to cover resets between games
        for _ in range(2):
            assert env.observation_space.contains(observation)

            terminated = truncated = False
            while not (terminated or truncated):
                observation, _, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )

            assert truncated
            assert not terminated
            assert info["match_frame"] == 100
            assert info["winner"] in (0, 1, 2)

            observation, _ = env.reset()
    finally:
        env.close()
