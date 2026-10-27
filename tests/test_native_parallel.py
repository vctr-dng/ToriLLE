"""Several games at once in direct launch mode. Run with `pytest -m native`.

Needs Toribash, Xvfb and a running Steam client (the games use it for Steamworks IPC).
The instances use copies of the game directory and their own Xvfb displays, and do not
take the global game lock themselves (the native session lock of `conftest.py` is held
anyway while this file runs).
"""

import itertools
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from torille import constants
from torille.envs.two_player import AGENTS, TwoPlayerToriEnv
from torille.pool import InstancePool, processes_running_from
from torille.torille import steam_client_running

pytestmark = pytest.mark.native

SIZE = 128


@pytest.fixture(scope="module", autouse=True)
def _require_direct_launch():
    executable = Path(constants.get_platform_executable())
    if not executable.is_file():
        pytest.skip(f"Toribash is not installed at {executable}")
    if shutil.which("Xvfb") is None:
        pytest.skip("Xvfb is not installed")
    if not steam_client_running():
        pytest.skip("The Steam client is not running (direct launch needs it)")


def make_env(kwargs, **options):
    return TwoPlayerToriEnv(
        draw_game=True,
        capture_size=SIZE,
        matchframes=60,
        turnframes=5,
        **kwargs,
        **options,
    )


def play_episode(env, seed, fixed_action=None):
    """Play one episode, returns (frames, joint_states of player_0, final match_frame)."""
    rng = np.random.default_rng(seed)
    observations, _ = env.reset(seed=seed)
    frames = [observations["player_0"]["rgb"]]
    joints = [observations["player_0"]["self"]["joint_states"]]
    infos = {}
    while env.agents:
        actions = {
            agent: (
                np.full(22, fixed_action)
                if fixed_action is not None
                else rng.integers(0, 4, size=22)
            )
            for agent in AGENTS
        }
        observations, _, _, _, infos = env.step(actions)
        frames.append(observations["player_0"]["rgb"])
        joints.append(observations["player_0"]["self"]["joint_states"])
    return frames, joints, infos["player_0"]["match_frame"]


def test_two_instances_play_at_the_same_time():
    real_game = Path(constants.get_platform_executable()).parent
    real_profile = (real_game / "profile.tbs").read_text()
    with InstancePool(2) as pool:
        root = pool.root
        assert pool[0]["display"] != pool[1]["display"]
        envs = [make_env(kwargs) for kwargs in pool]
        try:
            # Different actions per instance, run in parallel
            with ThreadPoolExecutor(2) as executor:
                futures = [
                    executor.submit(play_episode, envs[0], 1, 0),  # all joints extend
                    executor.submit(play_episode, envs[1], 2, 3),  # all joints relax
                ]
                (frames_a, joints_a, end_a), (frames_b, joints_b, end_b) = (
                    f.result(timeout=300) for f in futures
                )
            assert end_a == end_b == 60
            for frames in (frames_a, frames_b):
                assert len(frames) > 5
                assert all(f.shape == (SIZE, SIZE, 3) for f in frames)
                assert all(f.std() > 5 for f in frames)  # a picture, not one colour
                changes = [
                    np.abs(a.astype(int) - b.astype(int)).mean()
                    for a, b in itertools.pairwise(frames)
                ]
                assert sum(c > 0.2 for c in changes) >= 3
            # Independent states: each game obeyed its own actions
            np.testing.assert_array_equal(joints_a[-1], np.full(20, 1))
            np.testing.assert_array_equal(joints_b[-1], np.full(20, 4))
            # Both games were running from different directories
            for env, kwargs in zip(envs, pool, strict=True):
                assert env.game.process.is_alive()
                assert Path(env.game.executable_path).parent == Path(kwargs["game_dir"])
            assert len(processes_running_from(root)) == 2
            # Each copy has its own bridge, the real game directory has none
            for kwargs in pool:
                assert (Path(kwargs["game_dir"]) / ".torille").is_dir()
            assert not (real_game / ".torille").exists()
        finally:
            for env in envs:
                env.close()
        assert processes_running_from(root) == []
    assert not root.exists()
    assert (real_game / "profile.tbs").read_text() == real_profile
    assert not (real_game / ".torille").exists()


def test_direct_instance_runs_next_to_a_steam_launched_game():
    """Default (Steam) launch and a direct copy at the same time."""
    with InstancePool(1) as pool:
        steam_env = TwoPlayerToriEnv(matchframes=30, turnframes=5)  # default launch
        direct_env = make_env(pool[0], matchframes=30)
        try:
            steam_env.reset(seed=0)
            frames, joints, _ = play_episode(direct_env, 0, 0)
            np.testing.assert_array_equal(joints[-1], np.full(20, 1))
            assert all(f.std() > 5 for f in frames)
            actions = {a: np.full(22, 3) for a in AGENTS}
            observations, *_ = steam_env.step(actions)
            np.testing.assert_array_equal(
                observations["player_0"]["self"]["joint_states"], np.full(20, 4)
            )
        finally:
            direct_env.close()
            steam_env.close()
