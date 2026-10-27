"""Plays real two-player games in native Toribash. Run with `pytest -m native`."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from torille import constants
from torille.envs.two_player import AGENTS, TwoPlayerToriEnv

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


def play(env, actions_for_turn):
    """Play one episode, returns the list of (observations, rewards, infos) per turn."""
    observations, infos = env.reset(seed=0)
    assert set(observations) == set(infos) == set(AGENTS)
    history = []
    terminations = truncations = {}
    while env.agents:
        observations, rewards, terminations, truncations, infos = env.step(
            actions_for_turn(len(history))
        )
        assert set(observations) == set(rewards) == set(AGENTS)
        for agent in AGENTS:
            assert env.observation_space(agent).contains(observations[agent])
        history.append((observations, rewards, infos))
    return history, terminations, truncations


def test_full_random_episode_and_reset():
    env = TwoPlayerToriEnv(matchframes=100, turnframes=5)
    env.set_draw_game(False)
    try:
        observations, _ = env.reset(seed=0)
        # The game starts symmetric: both agents see the same thing. Only the
        # initial joint states differ (the game starts player 1 with 4 and
        # player 2 with 3 for every joint).
        for body in ("self", "opponent"):
            for key, value in observations["player_0"][body].items():
                if key == "joint_states":
                    continue
                np.testing.assert_allclose(
                    value, observations["player_1"][body][key], rtol=1e-4, atol=1e-4
                )
        for episode in range(2):
            if episode:
                observations, _ = env.reset()
            returns = dict.fromkeys(AGENTS, 0.0)
            terminations = truncations = {}
            while env.agents:
                actions = {a: env.action_space(a).sample() for a in env.agents}
                observations, rewards, terminations, truncations, infos = env.step(
                    actions
                )
                assert rewards["player_0"] == -rewards["player_1"]
                for agent in AGENTS:
                    assert env.observation_space(agent).contains(observations[agent])
                    returns[agent] += rewards[agent]
            assert all(truncations.values()) and not any(terminations.values())
            assert infos["player_0"]["match_frame"] == 100
            assert infos["player_0"]["winner"] in (0, 1, 2)
            assert returns["player_0"] == pytest.approx(-returns["player_1"])
            # Injuries reported per agent are mirrored
            assert infos["player_0"]["injury"] == infos["player_1"]["opponent_injury"]
            assert infos["player_1"]["injury"] == infos["player_0"]["opponent_injury"]
            with pytest.raises(RuntimeError, match="ended"):
                env.step({a: env.action_space(a).sample() for a in AGENTS})
    finally:
        env.close()


def test_each_agent_controls_its_own_body():
    """player_0 extends all joints, player_1 relaxes all: joint states show it."""
    env = TwoPlayerToriEnv(matchframes=30, turnframes=5)
    env.set_draw_game(False)
    try:
        env.reset(seed=0)
        actions = {"player_0": np.zeros(22, dtype=int), "player_1": np.full(22, 3)}
        observations, *_ = env.step(actions)
        for agent, state in (("player_0", 1), ("player_1", 4)):
            np.testing.assert_array_equal(
                observations[agent]["self"]["joint_states"], np.full(20, state)
            )
        # The other agent sees the same bodies the other way round
        np.testing.assert_array_equal(
            observations["player_0"]["opponent"]["joint_states"],
            observations["player_1"]["self"]["joint_states"],
        )
        np.testing.assert_array_equal(
            observations["player_1"]["opponent"]["joint_states"],
            observations["player_0"]["self"]["joint_states"],
        )
        # Grips: action 0 releases, action 3 grips
        np.testing.assert_array_equal(observations["player_0"]["self"]["grips"], [0, 0])
        np.testing.assert_array_equal(observations["player_1"]["self"]["grips"], [1, 1])
    finally:
        env.close()


def test_same_actions_give_identical_episodes():
    rng = np.random.default_rng(1)
    script = [{a: rng.integers(0, 4, size=22) for a in AGENTS} for _ in range(25)]
    env = TwoPlayerToriEnv(matchframes=100, turnframes=5)
    env.set_draw_game(False)
    try:
        runs = [play(env, lambda turn: script[turn])[0] for _ in range(2)]
    finally:
        env.close()
    first, second = runs
    assert len(first) == len(second)
    for (obs_a, rew_a, _), (obs_b, rew_b, _) in zip(first, second, strict=True):
        assert rew_a == rew_b
        for agent in AGENTS:
            for body in ("self", "opponent"):
                for key, value in obs_a[agent][body].items():
                    np.testing.assert_array_equal(value, obs_b[agent][body][key])
