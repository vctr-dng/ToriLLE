"""RGB capture in real games. Run with `pytest -m native` (needs a display)."""

import itertools
import os
import shutil
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import torille.envs  # noqa: F401
from torille import constants
from torille.envs.two_player import AGENTS, TwoPlayerToriEnv

pytestmark = pytest.mark.native

SIZE = 128


@pytest.fixture(scope="module", autouse=True)
def _require_native_toribash():
    executable = Path(constants.get_platform_executable())
    if not executable.is_file():
        pytest.skip(f"Toribash is not installed at {executable}")
    if shutil.which("steam") is None:
        pytest.skip("Steam is not available")
    if not os.environ.get("DISPLAY"):
        pytest.skip("No DISPLAY for Toribash's OpenGL renderer")


def mean_difference(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


@pytest.fixture
def env():
    environment = gym.make(
        "Toribash-DestroyUke-v2",
        observation_type="dict",
        capture_size=SIZE,
        render_mode="rgb_array",
        draw_game=True,
        turnframes=5,
        matchframes=100,
    )
    yield environment
    environment.close()


def test_frames_are_aligned_with_turns_and_stable(env):
    observation, _ = env.reset(seed=0)
    frames = [observation["rgb"]]
    for _ in range(8):
        observation, *_ = env.step(env.action_space.sample())
        frames.append(observation["rgb"])
        assert env.observation_space.contains(observation)
        # `render()` is the frame of the same turn boundary
        np.testing.assert_array_equal(env.render(), observation["rgb"])
        # A second grab at the same boundary shows the same pixels: the game
        # waits for the action and nothing on screen moves meanwhile.
        again = env.unwrapped.game.grab_frame()
        assert mean_difference(again, observation["rgb"]) < 0.2

    for frame in frames:
        assert frame.shape == (SIZE, SIZE, 3)
        assert frame.dtype == np.uint8
        assert frame.std() > 5  # the game is on screen, not a flat colour
    # The fighters move between turns, so consecutive frames differ
    changes = [mean_difference(a, b) for a, b in itertools.pairwise(frames)]
    assert sum(change > 0.2 for change in changes) >= 6, changes


def test_first_frame_is_the_same_after_every_reset():
    """Fixed camera and the same start pose give the same picture each episode."""
    # v0 starts from the same pose every episode (v2 randomises the opponent)
    env = gym.make(
        "Toribash-DestroyUke-v0",
        observation_type="dict",
        capture_size=SIZE,
        draw_game=True,
        turnframes=5,
        matchframes=100,
    )
    try:
        first, _ = env.reset(seed=0)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        second, _ = env.reset()
        assert mean_difference(first["rgb"], second["rgb"]) < 0.5
        assert first["rgb"].std() > 5
    finally:
        env.close()


def test_capture_does_not_warn_on_a_working_display(env):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        env.reset(seed=0)
        env.step(env.action_space.sample())


def test_two_player_frames_and_render():
    env = TwoPlayerToriEnv(
        draw_game=True,
        capture_size=SIZE,
        render_mode="rgb_array",
        matchframes=100,
        turnframes=5,
    )
    try:
        observations, _ = env.reset(seed=0)
        first = observations["player_0"]["rgb"]
        actions = {agent: env.action_space(agent).sample() for agent in AGENTS}
        observations, *_ = env.step(actions)
        for agent in AGENTS:
            assert env.observation_space(agent).contains(observations[agent])
        np.testing.assert_array_equal(
            observations["player_0"]["rgb"], observations["player_1"]["rgb"]
        )
        np.testing.assert_array_equal(env.render(), observations["player_0"]["rgb"])
        assert mean_difference(first, env.render()) > 0.2
    finally:
        env.close()


def test_letterbox_frame_keeps_the_aspect_ratio():
    env = gym.make(
        "Toribash-DestroyUke-v2",
        render_mode="rgb_array",
        capture_size=None,
        draw_game=True,
        matchframes=100,
    )
    try:
        env.unwrapped.game.capture_crop = "letterbox"
        env.reset(seed=0)
        env.step(env.action_space.sample())
        frame = env.render()
        assert frame.shape == (256, 256, 3)
        # 16:9 window: black bars above and below
        assert (frame[:20] == 0).all()
        assert (frame[-20:] == 0).all()
        assert frame[100:156].std() > 5
    finally:
        env.close()


def test_game_directory_is_left_clean_after_capture():
    game_directory = Path(constants.get_platform_executable()).parent
    profile = (game_directory / "profile.tbs").read_text()
    env = gym.make(
        "Toribash-DestroyUke-v2",
        render_mode="rgb_array",
        draw_game=True,
        matchframes=100,
    )
    env.reset(seed=0)
    assert env.render().shape == (256, 256, 3)
    env.close()
    assert not (game_directory / ".torille").exists()
    assert not (game_directory / "data/script/remotecontrol_native.lua").exists()
    assert (game_directory / "profile.tbs").read_text() == profile
