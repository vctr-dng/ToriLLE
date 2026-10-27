"""Observation spaces and conversions. These tests do not start the game."""

import gymnasium as gym
import numpy as np
import pytest

import torille.envs  # noqa: F401
from torille import constants
from torille.envs.observations import dict_observation_space, state_to_dict_observation
from torille.torille import ToribashState

ENV_IDS = [
    "Toribash-SelfDestruct-v0",
    "Toribash-DestroyUke-v0",
    "Toribash-DestroyUke-v2",
    "Toribash-DuoWinCombat-v0",
]


def make_state(match_frame=15, frames_next_turn=5, match_length=1000):
    """A synthetic state with distinct values in every field."""
    rng = np.random.default_rng(0)
    values = np.zeros(constants.STATE_LENGTH)
    for start in (0, 165):
        values[start : start + 63] = rng.normal(size=63)  # positions
        values[start + 63 : start + 126] = rng.normal(size=63)  # velocities
        values[start + 126 : start + 142] = np.eye(4).ravel()  # groin rotation
        joints = rng.integers(1, 5, size=20)
        values[start + 142 : start + 162] = joints
        values[start + 162 : start + 164] = [0, 1]  # grips
    values[330:334] = [0, match_length, match_frame, frames_next_turn]
    return ToribashState(list(values))


def test_conversion_matches_the_space():
    state = make_state()
    observation = state_to_dict_observation(state)

    assert dict_observation_space(1000).contains(observation)


def test_conversion_values():
    state = make_state(match_frame=15, frames_next_turn=5)
    observation = state_to_dict_observation(state)

    np.testing.assert_allclose(observation["velocities"], state.limb_velocities)
    np.testing.assert_array_equal(
        observation["joint_states"], state.joint_states[:, :20]
    )
    np.testing.assert_array_equal(observation["grips"], [[0, 1], [0, 1]])
    np.testing.assert_array_equal(observation["match_frame"], [15])
    np.testing.assert_array_equal(observation["frames_next_turn"], [5])
    np.testing.assert_allclose(
        observation["groin_height"], state.limb_positions[:, 4, 2], rtol=1e-6
    )
    # Positions are centred on player 1's groin
    np.testing.assert_allclose(observation["positions"][0, 4], 0, atol=1e-6)
    np.testing.assert_allclose(
        observation["positions"],
        state.get_normalized_locations()[0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_space_shapes_and_bounds():
    space = dict_observation_space(250)

    assert space["positions"].shape == (2, 21, 3)
    assert space["velocities"].shape == (2, 21, 3)
    assert space["joint_states"].shape == (2, 20)
    assert space["grips"].shape == (2, 2)
    assert space["groin_height"].shape == (2,)
    assert space["match_frame"].high[0] == 250
    assert space["frames_next_turn"].high[0] == 250
    assert space["joint_states"].high.max() == 4
    assert space["grips"].high.max() == 1
    assert all(sub.dtype == np.float32 for sub in space.spaces.values())


def test_out_of_range_observation_is_rejected():
    space = dict_observation_space(100)
    observation = state_to_dict_observation(make_state(match_length=100))
    assert space.contains(observation)

    observation["match_frame"] = np.array([101], dtype=np.float32)
    assert not space.contains(observation)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_dict_flag_changes_only_the_observation_space(env_id):
    box_env = gym.make(env_id)
    dict_env = gym.make(env_id, observation_type="dict")

    assert box_env.unwrapped.observation_type == "box"
    assert isinstance(box_env.observation_space, gym.spaces.Box)
    assert isinstance(dict_env.observation_space, gym.spaces.Dict)
    assert dict_env.action_space == box_env.action_space


def test_dict_space_follows_matchframes():
    env = gym.make("Toribash-DestroyUke-v0", observation_type="dict", matchframes=300)

    assert env.observation_space["match_frame"].high[0] == 300


def test_default_observation_space_is_unchanged():
    env = gym.make("Toribash-DestroyUke-v0")

    assert env.observation_space.shape == (21 * 3 * 2,)


def test_unknown_observation_type_is_rejected():
    with pytest.raises(ValueError, match="observation_type"):
        gym.make("Toribash-DestroyUke-v0", observation_type="image")


def test_box_conversion_still_works_for_dict_envs():
    env = gym.make("Toribash-DestroyUke-v0", observation_type="dict")
    state = make_state()

    assert env.observation_space.contains(env.unwrapped._observe(state))


def test_seed_is_reproducible_for_the_random_opponent():
    env = gym.make("Toribash-DestroyUke-v2").unwrapped

    def opponent_actions(seed):
        env.seed(seed)
        return [env._preprocess_action([0] * 22)[1] for _ in range(3)]

    first = opponent_actions(7)
    assert opponent_actions(7) == first
    assert opponent_actions(8) != first
    assert all(1 <= a <= 4 for turn in first for a in turn)
