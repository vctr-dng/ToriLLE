"""Plays real games in native Toribash. Run with `pytest -m native`."""

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
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


def test_game_is_stopped_and_directory_restored_when_python_exits_without_close():
    game_directory = Path(constants.get_platform_executable()).parent
    profile = game_directory / "profile.tbs"
    original_profile = profile.read_text(encoding="utf-8")
    script = textwrap.dedent(
        """
        import gymnasium as gym

        import torille.envs
        from torille.torille import ToribashControl

        # __del__ is not guaranteed to run at shutdown, so only atexit may clean up
        ToribashControl.__del__ = lambda self: None

        env = gym.make("Toribash-DestroyUke-v0", matchframes=100, turnframes=5)
        env.reset()
        env.step(env.action_space.sample())
        # Exit without calling env.close()
        """
    )

    subprocess.run([sys.executable, "-I", "-c", script], check=True, timeout=120)

    assert profile.read_text(encoding="utf-8") == original_profile
    assert not (game_directory / ".torille").exists()
    assert not (
        game_directory / "data" / "script" / "remotecontrol_native.lua"
    ).exists()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (
            subprocess.run(["pgrep", "-x", "toribash_steam"], check=False).returncode
            != 0
        ):
            break
        time.sleep(0.2)
    else:
        pytest.fail("Toribash is still running")


def _play(env, actions, seed=None):
    """Reset, then play `actions`; return all observations and rewards."""
    observation, _ = env.reset(seed=seed)
    observations, rewards = [observation], []
    for action in actions:
        observation, reward, terminated, truncated, _ = env.step(action)
        observations.append(observation)
        rewards.append(reward)
        if terminated or truncated:
            break
    return np.array(observations), np.array(rewards)


def test_reset_is_deterministic():
    env = gym.make("Toribash-DestroyUke-v0", matchframes=50, turnframes=5)
    try:
        first, _ = env.reset(seed=0)
        env.step(env.action_space.sample())
        # Finish the episode so that reset starts a new game
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        second, _ = env.reset(seed=1)
        np.testing.assert_array_equal(first, second)
    finally:
        env.close()


@pytest.mark.parametrize("draw_game", [True, False])
def test_replaying_the_same_actions_gives_the_same_trajectory(draw_game):
    env = gym.make("Toribash-DestroyUke-v0", matchframes=100, turnframes=5)
    env.unwrapped.set_draw_game(draw_game)
    env.action_space.seed(0)
    actions = [env.action_space.sample() for _ in range(20)]
    try:
        first_obs, first_rewards = _play(env, actions)
        assert len(first_obs) > 2
        for _ in range(2):
            obs, rewards = _play(env, actions)
            np.testing.assert_array_equal(obs, first_obs)
            np.testing.assert_array_equal(rewards, first_rewards)
    finally:
        env.close()


def test_dict_observation_env_plays_and_is_deterministic():
    env = gym.make(
        "Toribash-DestroyUke-v2",
        observation_type="dict",
        matchframes=100,
        turnframes=5,
    )
    try:
        observation, _ = env.reset(seed=3)
        assert env.observation_space.contains(observation)
        assert observation["match_frame"][0] == 0
        assert observation["frames_next_turn"][0] == 5
        observation, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        assert observation["match_frame"][0] > 0
        # The random opponent comes from np_random, so the seed fixes it
        # (the game itself is deterministic, see docs/envs.md)
        actions = [env.action_space.sample() for _ in range(10)]

        def finish_episode():
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step([0] * 22)
                done = terminated or truncated

        def final_observation(seed):
            env.reset(seed=seed)
            observations, done = [], False
            for action in actions:
                observation, _, terminated, truncated, _ = env.step(action)
                observations.append(observation)
                done = terminated or truncated
                if done:
                    break
            if not done:
                finish_episode()
            return observations[-1]

        # reset() is only allowed after the game has ended
        finish_episode()

        first = final_observation(5)
        second = final_observation(5)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
    finally:
        env.close()


def _play_with_think_time(draw_game, turnframes, spin_ms, delays, turns):
    """Play `turns` turns with fixed actions; return observations and frames."""
    env = gym.make(
        "Toribash-DestroyUke-v2",
        matchframes=300 * turnframes,
        turnframes=turnframes,
    )
    env.unwrapped.set_draw_game(draw_game)
    env.unwrapped.game.spin_ms = spin_ms
    env.action_space.seed(0)
    rng = np.random.default_rng(0)
    observations, frames, resets = [], [], []
    try:
        env.reset(seed=0)
        for turn in range(turns):
            time.sleep(delays[rng.integers(len(delays))])
            obs, _, terminated, truncated, info = env.step(env.action_space.sample())
            observations.append(obs["position"] if isinstance(obs, dict) else obs)
            frames.append(info["match_frame"])
            if terminated or truncated:
                resets.append(turn)
                env.reset()
    finally:
        env.close()
    return np.array(observations), frames, resets


@pytest.mark.parametrize(("draw_game", "turnframes"), [(False, 1), (True, 2)])
def test_bridge_does_not_desync_over_hundreds_of_turns(draw_game, turnframes):
    """Every turn yields exactly one state and the game advances predictably.

    Python's think time is varied around the Lua-side spin window, so both the
    in-hook pickup and the per-frame fallback are exercised (including
    responses landing right at the edge of the window). The trajectory must be
    identical whichever path picked up the actions.
    """
    delays = [0, 0, 0.001, 0.004, 0.0055, 0.015]
    turns = 400
    reference, ref_frames, ref_resets = _play_with_think_time(
        draw_game, turnframes, 0.0, [0], turns
    )
    observations, frames, resets = _play_with_think_time(
        draw_game, turnframes, 5.0, delays, turns
    )

    assert len(frames) == turns
    assert resets == ref_resets
    assert frames == ref_frames
    # frames advance monotonically within an episode, no skipped or repeated turns
    for index in range(1, turns):
        if index - 1 not in resets:
            assert frames[index] > frames[index - 1], f"desync at turn {index}"
    np.testing.assert_array_equal(observations, reference)
