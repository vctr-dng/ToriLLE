"""Two-player API: spaces, mirroring, actions, rewards. These tests do not start the game."""

import numpy as np
import pytest

from torille import constants
from torille.envs.two_player import (
    AGENTS,
    TwoPlayerToriEnv,
    action_space,
    agent_observation,
    merge_actions,
    observation_space,
    split_actions,
)
from torille.torille import ToribashState


def _swap_blocks(values):
    """Swap the player 1 and player 2 parts of a raw state vector."""
    swapped = values.copy()
    swapped[0:164] = values[165:329]
    swapped[165:329] = values[0:164]
    swapped[164], swapped[329] = values[329], values[164]
    return swapped


def make_values(seed=0, winner_frame=(15, 5, 1000)):
    """Raw state vector with different random values for each player."""
    rng = np.random.default_rng(seed)
    values = np.zeros(constants.STATE_LENGTH)
    for start in (0, 165):
        values[start : start + 63] = rng.normal(size=63)
        values[start + 63 : start + 126] = rng.normal(size=63)
        # A proper rotation: yaw and pitch
        a, b = rng.uniform(0, 2 * np.pi, size=2)
        yaw = np.array(
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        )
        pitch = np.array(
            [[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]]
        )
        rotation = np.eye(4)
        rotation[:3, :3] = yaw @ pitch
        values[start + 126 : start + 142] = rotation.ravel()
        values[start + 142 : start + 162] = rng.integers(1, 5, size=20)
        values[start + 162 : start + 164] = rng.integers(0, 2, size=2)
    values[164], values[329] = 120.0, 340.0
    match_frame, frames_next_turn, match_length = winner_frame
    values[330:334] = [0, match_length, match_frame, frames_next_turn]
    return values


def make_state(values=None, winner=None):
    return ToribashState(list(make_values() if values is None else values), winner)


@pytest.fixture
def env():
    # Never started: the tests that fake the game must not close it either
    return TwoPlayerToriEnv(matchframes=100, turnframes=5)


def test_spaces(env):
    assert env.possible_agents == ["player_0", "player_1"]
    for agent in AGENTS:
        assert env.action_space(agent) == action_space()
        assert env.action_space(agent).nvec.tolist() == [4] * 22
        assert env.observation_space(agent) == observation_space(100)
    space = env.observation_space("player_0")
    assert space["self"]["positions"].shape == (21, 3)
    assert space["opponent"]["velocities"].shape == (21, 3)
    assert space["self"]["joint_states"].shape == (20,)
    assert space["opponent"]["grips"].shape == (2,)
    assert space["self"]["groin_height"].shape == (1,)
    assert space["match_frame"].high[0] == 100
    with pytest.raises(ValueError, match="unknown agent"):
        env.action_space("player_2")


def test_observation_conforms_to_the_space():
    state = make_state(make_values(winner_frame=(15, 5, 100)))
    space = observation_space(100)
    for player in (0, 1):
        assert space.contains(agent_observation(state, player))


def test_self_and_opponent_are_swapped_between_agents():
    state = make_state()
    first = agent_observation(state, 0)
    second = agent_observation(state, 1)

    np.testing.assert_array_equal(
        first["self"]["joint_states"], state.joint_states[0, :20]
    )
    np.testing.assert_array_equal(
        second["self"]["joint_states"], state.joint_states[1, :20]
    )
    np.testing.assert_array_equal(
        first["opponent"]["joint_states"], second["self"]["joint_states"]
    )
    np.testing.assert_array_equal(first["self"]["grips"], state.joint_states[0, 20:])
    np.testing.assert_array_equal(
        second["opponent"]["grips"], state.joint_states[0, 20:]
    )
    np.testing.assert_allclose(
        first["self"]["groin_height"], state.limb_positions[0, 4, 2:3], rtol=1e-6
    )
    np.testing.assert_allclose(
        second["opponent"]["groin_height"], state.limb_positions[0, 4, 2:3], rtol=1e-6
    )
    # Each agent is at the origin of its own frame
    for observation in (first, second):
        np.testing.assert_allclose(observation["self"]["positions"][4], 0, atol=1e-6)
    np.testing.assert_array_equal(first["match_frame"], [15])
    np.testing.assert_array_equal(second["frames_next_turn"], [5])


def test_swapping_the_players_swaps_the_observations():
    """The view of a fighter does not depend on whether it is player 1 or 2."""
    values = make_values(seed=3)
    state = make_state(values)
    swapped = make_state(_swap_blocks(values))

    for player in (0, 1):
        original = agent_observation(state, player)
        mirrored = agent_observation(swapped, 1 - player)
        for body in ("self", "opponent"):
            for key, value in original[body].items():
                np.testing.assert_allclose(
                    value, mirrored[body][key], rtol=1e-5, atol=1e-5
                )


def test_a_symmetric_state_gives_identical_observations():
    """Player 2 is player 1 turned around the vertical axis: same view."""
    values = make_values(seed=5)
    turn = np.diag([-1.0, -1.0, 1.0])
    positions = values[0:63].reshape(21, 3)
    velocities = values[63:126].reshape(21, 3)
    values[165:228] = (positions @ turn.T).ravel()
    values[228:291] = (velocities @ turn.T).ravel()
    rotation = np.eye(4)
    rotation[:3, :3] = turn
    values[126:142] = np.eye(4).ravel()
    values[291:307] = rotation.ravel()
    values[165 + 142 : 165 + 164] = values[142:164]
    state = make_state(values)

    first = agent_observation(state, 0)
    second = agent_observation(state, 1)
    for body in ("self", "opponent"):
        for key, value in first[body].items():
            np.testing.assert_allclose(value, second[body][key], rtol=1e-5, atol=1e-5)


def test_velocities_are_in_the_agents_frame_and_positions_are_not_world():
    state = make_state()
    observation = agent_observation(state, 1)
    rotation = state.groin_rotations[1, :3, :3]

    np.testing.assert_allclose(
        observation["self"]["velocities"],
        state.limb_velocities[1] @ rotation,
        rtol=1e-5,
        atol=1e-5,
    )
    # Rotations preserve lengths
    np.testing.assert_allclose(
        np.linalg.norm(observation["self"]["velocities"], axis=1),
        np.linalg.norm(state.limb_velocities[1], axis=1),
        rtol=1e-5,
    )


def test_merge_and_split_actions_roundtrip():
    rng = np.random.default_rng(0)
    actions = {agent: rng.integers(0, 4, size=22) for agent in AGENTS}
    merged = merge_actions(actions)

    assert len(merged) == 2 and all(len(player) == 22 for player in merged)
    assert all(type(a) is int for player in merged for a in player)
    assert min(min(player) for player in merged) >= 1
    assert max(max(player) for player in merged) <= 4
    assert merged[0] == (actions["player_0"] + 1).tolist()
    assert merged[1] == (actions["player_1"] + 1).tolist()
    for agent, action in split_actions(merged).items():
        np.testing.assert_array_equal(action, actions[agent])


def test_actions_accept_lists_and_integer_dtypes():
    ok = {"player_0": [0] * 22, "player_1": np.full(22, 3, dtype=np.int8)}
    assert merge_actions(ok) == [[1] * 22, [4] * 22]


@pytest.mark.parametrize(
    "actions",
    [
        {"player_0": [0] * 22},
        {"player_0": [0] * 22, "player_1": [0] * 22, "player_2": [0] * 22},
        {"player_0": [0] * 22, "player_1": [0] * 21},
        {"player_0": [0] * 22, "player_1": [4] * 22},
        {"player_0": [0] * 22, "player_1": [-1] * 22},
        {"player_0": [0] * 22, "player_1": [0.0] * 22},
    ],
)
def test_invalid_actions_are_rejected(actions):
    with pytest.raises(ValueError):
        merge_actions(actions)


def test_step_requires_reset(env):
    with pytest.raises(RuntimeError, match="reset"):
        env.step({agent: [0] * 22 for agent in AGENTS})


def test_unknown_reward_is_rejected():
    with pytest.raises(ValueError, match="reward"):
        TwoPlayerToriEnv(reward="nope")


def test_default_settings_and_overrides():
    env = TwoPlayerToriEnv()
    assert (env.settings.get("matchframes"), env.settings.get("turnframes")) == (
        1000,
        5,
    )
    env = TwoPlayerToriEnv(matchframes=40, turnframes=2)
    assert (env.settings.get("matchframes"), env.settings.get("turnframes")) == (40, 2)
    assert env.observation_space("player_1")["match_frame"].high[0] == 40


def _run_step(env, old, new, terminal, monkeypatch):
    """Run `env.step` against fake control layer calls."""
    env._started = True
    env.agents = list(AGENTS)
    env._old_state = old
    monkeypatch.setattr(env.game, "make_actions", lambda actions: None)
    monkeypatch.setattr(env.game, "get_state", lambda: (new, terminal))
    return env.step({agent: [0] * 22 for agent in AGENTS})


def _state_with(injuries=(0.0, 0.0), winner=None, match_frame=10, match_length=100):
    values = make_values(winner_frame=(match_frame, 5, match_length))
    values[164], values[329] = injuries
    return make_state(values, winner)


def test_injury_reward_is_zero_sum(env, monkeypatch):
    old = _state_with(injuries=(100.0, 200.0))
    new = _state_with(injuries=(100.0, 700.0))  # player 2 took 500 more damage
    _, rewards, terminations, truncations, infos = _run_step(
        env, old, new, False, monkeypatch
    )

    assert rewards == {"player_0": 0.1, "player_1": -0.1}
    assert terminations == truncations == {"player_0": False, "player_1": False}
    assert env.agents == ["player_0", "player_1"]
    assert infos["player_0"]["injury"] == 100.0
    assert infos["player_0"]["opponent_injury"] == 700.0
    assert infos["player_1"]["injury"] == 700.0
    assert infos["player_1"]["opponent_injury"] == 100.0
    assert infos["player_0"]["winner"] is None


def test_injury_reward_for_damage_taken(env, monkeypatch):
    old = _state_with(injuries=(0.0, 0.0))
    new = _state_with(injuries=(1000.0, 500.0))
    _, rewards, *_ = _run_step(env, old, new, False, monkeypatch)

    assert rewards["player_0"] == pytest.approx(-0.1)
    assert rewards["player_1"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("winner", "expected", "winner_agent"),
    [(1, 1.0, "player_0"), (2, -1.0, "player_1"), (0, 0.0, None)],
)
def test_win_reward_and_winner_info(winner, expected, winner_agent, monkeypatch):
    env = TwoPlayerToriEnv(reward="win")
    old = _state_with()
    new = _state_with(winner=winner, match_frame=100)
    _, rewards, terminations, truncations, infos = _run_step(
        env, old, new, True, monkeypatch
    )

    assert rewards == {"player_0": expected, "player_1": -expected}
    # Reaching matchframes is a truncation
    assert truncations == {"player_0": True, "player_1": True}
    assert terminations == {"player_0": False, "player_1": False}
    assert env.agents == []
    for agent in AGENTS:
        assert infos[agent]["winner"] == winner
        assert infos[agent]["winner_agent"] == winner_agent
    with pytest.raises(RuntimeError, match="ended"):
        env.step({agent: [0] * 22 for agent in AGENTS})


def test_win_reward_is_zero_while_running(monkeypatch):
    env = TwoPlayerToriEnv(reward="win")
    _, rewards, *_ = _run_step(env, _state_with(), _state_with(), False, monkeypatch)
    assert rewards == {"player_0": 0.0, "player_1": 0.0}


def test_early_end_is_a_termination(env, monkeypatch):
    new = _state_with(winner=2, match_frame=40)
    _, _, terminations, truncations, _ = _run_step(
        env, _state_with(), new, True, monkeypatch
    )

    assert terminations == {"player_0": True, "player_1": True}
    assert truncations == {"player_0": False, "player_1": False}


def test_custom_reward_is_player_0_pov_and_negated(monkeypatch):
    env = TwoPlayerToriEnv(reward=lambda old, new: 2.5)
    _, rewards, *_ = _run_step(env, _state_with(), _state_with(), False, monkeypatch)
    assert rewards == {"player_0": 2.5, "player_1": -2.5}


def test_step_sends_merged_actions(env, monkeypatch):
    sent = []
    env._started = True
    env.agents = list(AGENTS)
    env._old_state = _state_with()
    monkeypatch.setattr(env.game, "make_actions", sent.append)
    monkeypatch.setattr(env.game, "get_state", lambda: (_state_with(), False))
    env.step({"player_0": [0] * 22, "player_1": [3] * 22})

    assert sent == [[[1] * 22, [4] * 22]]


def test_reset_seed_seeds_np_random(env, monkeypatch):
    env._started = True
    monkeypatch.setattr(env.game, "reset", lambda: _state_with())
    env.reset(seed=4)
    first = env.np_random.integers(1 << 30)
    env.reset(seed=4)
    assert env.np_random.integers(1 << 30) == first
