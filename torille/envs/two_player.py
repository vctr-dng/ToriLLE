"""Two-player control API: both fighters are controlled from outside.

`TwoPlayerToriEnv` runs one game with both bodies controlled by the caller.
It follows the shape of the PettingZoo parallel API (without depending on
PettingZoo): all agents act at the same time, and everything is a dict keyed
by the agent ids `"player_0"` (Toribash player 1) and `"player_1"` (Toribash
player 2).

    env = TwoPlayerToriEnv(matchframes=1000, turnframes=5)
    observations, infos = env.reset(seed=0)
    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

ToriLLE ships no policies. Whatever fills a slot (a network, a script, a human
interface) only has to map that agent's observation to an action.

Conventions (see docs/envs.md for the full description):

* Observation: every agent sees the game from its own frame. It appears as
  `"self"` and the other fighter as `"opponent"`. Positions are centred on
  the agent's own groin and rotated by the agent's own groin rotation, so at
  the start of the game both agents get identical observations.
* Action: `MultiDiscrete([4] * 22)`, 20 joints then 2 grips, of the agent's
  own body. Joint values 0..3 mean extend, contract, hold, relax. Grip values
  0..1 release the grip, 2..3 grip (the game only knows two grip states).
* Reward: zero-sum, see `reward_win` and `reward_injury` (the default).
"""

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from .. import constants, torille
from ..capture import capture_observation_space
from .duo_envs import reward_injury_player1_pov, reward_win_player1_pov

AGENTS = ("player_0", "player_1")

REWARDS = {
    "injury": reward_injury_player1_pov,
    "win": reward_win_player1_pov,
}


def action_space():
    """Per-agent action space: 20 joints and 2 grips, 4 states each."""
    return spaces.MultiDiscrete(
        [constants.NUM_JOINT_STATES] * constants.NUM_CONTROLLABLES
    )


def _body_space():
    limbs = (constants.NUM_LIMBS, 3)
    return spaces.Dict(
        {
            "positions": spaces.Box(-np.inf, np.inf, limbs, np.float32),
            "velocities": spaces.Box(-np.inf, np.inf, limbs, np.float32),
            "groin_height": spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "joint_states": spaces.Box(0.0, 4.0, (constants.NUM_JOINTS,), np.float32),
            "grips": spaces.Box(0.0, 1.0, (2,), np.float32),
        }
    )


def observation_space(match_frames, capture_size=None):
    """
    Per-agent observation space (the same for both agents).

    Keys:
        self, opponent: `Dict` of one body each (all float32):
            positions: (NUM_LIMBS, 3) limb positions, centred on the agent's
                groin and rotated by the agent's groin rotation.
            velocities: (NUM_LIMBS, 3) limb velocities, rotated by the agent's
                groin rotation (not centred, they are free vectors).
            groin_height: (1,) absolute z of the groin.
            joint_states: (NUM_JOINTS,) joint state 1..4 (extend, contract,
                hold, relax), 0 for a joint with no state yet.
            grips: (2,) left and right hand grip, 0 or 1.
        match_frame: (1,) frames played so far, in [0, match_frames].
        frames_next_turn: (1,) frames the next turn lasts.
        rgb: only with `capture_size`: (capture_size, capture_size, 3) uint8
            frame of the game window at the turn boundary. It is the same
            picture for both agents (the game has one camera).
    """
    frames = spaces.Box(0.0, float(match_frames), (1,), np.float32)
    observation = {
        "self": _body_space(),
        "opponent": _body_space(),
        "match_frame": frames,
        "frames_next_turn": spaces.Box(0.0, float(match_frames), (1,), np.float32),
    }
    if capture_size is not None:
        observation["rgb"] = capture_observation_space(capture_size)
    return spaces.Dict(observation)


def _body_observation(state, normalized, frame, body):
    """
    Observation of fighter `body` (0 or 1) from the frame of fighter `frame`.

    The frame is the one of `ToribashState.get_normalized_locations`
    (`normalized` is its result): origin at the groin of `frame`, axes of its
    groin rotation.
    """
    rotation = state.groin_rotations[frame, :3, :3]
    positions = normalized[frame, body]
    velocities = state.limb_velocities[body] @ rotation
    joints = state.joint_states[body]
    return {
        "positions": positions.astype(np.float32),
        "velocities": velocities.astype(np.float32),
        "groin_height": state.limb_positions[body, 4, 2:3].astype(np.float32),
        "joint_states": joints[: constants.NUM_JOINTS].astype(np.float32),
        "grips": joints[constants.NUM_JOINTS :].astype(np.float32),
    }


def agent_observation(state, player):
    """
    Observation of `player` (0 or 1) in `observation_space`: that player is
    "self" and the other one "opponent", both in the frame of `player`.
    """
    normalized = state.get_normalized_locations()
    return {
        "self": _body_observation(state, normalized, player, player),
        "opponent": _body_observation(state, normalized, player, 1 - player),
        "match_frame": np.array([state.match_frame], dtype=np.float32),
        "frames_next_turn": np.array([state.frames_next_turn], dtype=np.float32),
    }


def merge_actions(actions):
    """
    Turn `{"player_0": a0, "player_1": a1}` (each of the `action_space`) into
    the control layer's `[[...22], [...22]]` with joint states 1..4 as Python
    ints. Raises `ValueError` for missing or unknown agents and for actions
    outside the action space.
    """
    if set(actions) != set(AGENTS):
        raise ValueError(
            f"actions must have exactly the keys {list(AGENTS)}, got {list(actions)}"
        )
    space = action_space()
    merged = []
    for agent in AGENTS:
        action = np.asarray(actions[agent])
        if not np.issubdtype(action.dtype, np.integer) or not space.contains(action):
            raise ValueError(
                f"action of {agent} is not in {space}: shape {action.shape}, "
                f"dtype {action.dtype}, values {action.tolist()}"
            )
        merged.append((action + 1).tolist())
    return merged


def split_actions(merged):
    """Inverse of `merge_actions`: `[[...22], [...22]]` to the agent dict."""
    return {
        agent: np.asarray(player, dtype=np.int64) - 1
        for agent, player in zip(AGENTS, merged, strict=True)
    }


class TwoPlayerToriEnv:
    """
    One Toribash game where both players are driven from outside.

    Parameters:
        executable: Path to the native Toribash executable (default: the
            Steam install).
        install_native_controller: See `ToribashControl`.
        reward: `"injury"` (default), `"win"` or a callable
            `f(old_state, new_state) -> float` giving the reward of
            `player_0`. `player_1` always receives the negative, so the
            rewards sum to zero.
        draw_game: Render at a watchable frame rate (see `ToribashControl`).
        render_mode: None or "rgb_array" (`render()` returns the latest frame).
        capture_size: Side of the square RGB frame. If given, every agent's
            observation gets an `rgb` key. `render_mode="rgb_array"` alone
            captures 256 px frames for `render()` only. Capture needs
            `draw_game=True`, see `torille.capture` and docs/envs.md.
        capture_crop: "center" (square crop of the window) or "letterbox".
        capture_display: X display of the game window (default `$DISPLAY`,
            or `display` in direct mode).
        launch_mode: `"steam"` (default, `steam -applaunch`) or `"direct"`
            (start the executable ourselves, needs the Steam client running).
        game_dir: Directory of a (copy of the) game, see `torille.pool`.
        display: Direct mode only: X display to start the game on.
        **settings: Toribash settings, e.g. `matchframes` (default 1000) and
            `turnframes` (default 5).

    Rewards (`ToribashState.injuries[k]` is the damage player k+1 has taken,
    `winner` is 0 tie, 1 player 1, 2 player 2, None while the game runs):

    * `"injury"`: `(injuries[1] - injuries[0])` gained during the turn,
      divided by 5000. It is positive for player_0 when player_1 took
      damage. Same as `reward_injury_player1_pov`.
    * `"win"`: 0 every turn, and on the last turn +1 if player 1 won, -1 if
      player 2 won, 0 on a tie. Same as `reward_win_player1_pov`.

    `agents` is `["player_0", "player_1"]` after `reset()` and `[]` once the
    episode has ended (as in PettingZoo); `step()` then needs a new `reset()`.
    """

    possible_agents = list(AGENTS)  # noqa: RUF012
    metadata = {  # noqa: RUF012
        "name": "toribash_two_player_v0",
        "render_modes": [None, "rgb_array"],
    }

    def __init__(
        self,
        executable=None,
        install_native_controller=True,
        reward="injury",
        draw_game=False,
        render_mode=None,
        capture_size=None,
        capture_crop="center",
        capture_display=None,
        launch_mode="steam",
        game_dir=None,
        display=None,
        **settings,
    ):
        if callable(reward):
            self._reward_function = reward
        elif reward in REWARDS:
            self._reward_function = REWARDS[reward]
        else:
            raise ValueError(
                f"reward must be one of {list(REWARDS)} or a callable, got {reward!r}"
            )
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}, "
                f"got {render_mode!r}"
            )
        self.render_mode = render_mode
        self.capture_rgb_observation = capture_size is not None
        if render_mode == "rgb_array" and capture_size is None:
            capture_size = 256
        self.capture_size = capture_size
        settings.setdefault("matchframes", 1000)
        settings.setdefault("turnframes", 5)
        self.settings = torille.ToribashSettings(**settings)
        self.game = torille.ToribashControl(
            settings=self.settings,
            draw_game=draw_game,
            executable=executable,
            install_native_controller=install_native_controller,
            capture_size=capture_size,
            capture_crop=capture_crop,
            capture_display=capture_display,
            launch_mode=launch_mode,
            game_dir=game_dir,
            display=display,
        )
        self._frame = None
        self._frame_match_frame = None
        self._current_state = None
        self.agents = []
        self._old_state = None
        self._started = False
        self._np_random = None
        self._observation_space = observation_space(
            self.settings.get("matchframes"),
            capture_size if self.capture_rgb_observation else None,
        )
        self._action_space = action_space()

    def observation_space(self, agent):
        """Observation space of `agent`. Built at construction from `matchframes`."""
        self._check_agent(agent)
        return self._observation_space

    def action_space(self, agent):
        """Action space of `agent`: `MultiDiscrete([4] * 22)`."""
        self._check_agent(agent)
        return self._action_space

    @staticmethod
    def _check_agent(agent):
        if agent not in AGENTS:
            raise ValueError(f"unknown agent {agent!r}, expected one of {AGENTS}")

    @property
    def np_random(self):
        """Generator seeded by `reset(seed=...)`. Nothing in the env uses it."""
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()
        return self._np_random

    def set_draw_game(self, visibility):
        """Show the game (limits the frame rate). Only before the first `reset()`."""
        if self._started:
            raise RuntimeError("Can't change rendering after first `reset()`")
        self.game.draw_game = visibility

    def _capture_frame(self, state):
        """Frame of the displayed state `state` (grabbed once per state)."""
        if self._frame is None or self._frame_match_frame != state.match_frame:
            self._frame = self.game.grab_frame()
            self._frame_match_frame = state.match_frame
        return self._frame

    def _observations(self, state):
        self._current_state = state
        observations = {
            agent: agent_observation(state, player)
            for player, agent in enumerate(AGENTS)
        }
        if self.capture_rgb_observation:
            for observation in observations.values():
                observation["rgb"] = self._capture_frame(state).copy()
        return observations

    def render(self):
        """
        With `render_mode="rgb_array"`: the `(capture_size, capture_size, 3)`
        uint8 frame of the current turn boundary (state of the last `reset()`
        or `step()`). Otherwise None.
        """
        if self.render_mode != "rgb_array":
            return None
        if self._current_state is None:
            raise RuntimeError("`render` was called before `reset`.")
        return self._capture_frame(self._current_state).copy()

    def reset(self, seed=None, options=None):
        """
        Start a game (launching Toribash on the first call).

        Returns:
            observations, infos: dicts keyed by agent id.
        """
        if seed is not None:
            self._np_random, _ = seeding.np_random(seed)
        if not self._started:
            self.game.init()
            self._started = True
            state, _terminal = self.game.get_state()
        else:
            state = self.game.reset()
        self._frame = None
        self._old_state = state
        self.agents = list(AGENTS)
        return self._observations(state), {agent: {} for agent in AGENTS}

    def step(self, actions):
        """
        Play one turn with both players' actions.

        Parameters:
            actions: `{"player_0": a0, "player_1": a1}`, each in `action_space`.

        Returns:
            observations, rewards, terminations, truncations, infos: dicts
            keyed by agent id. Truncation means the game reached `matchframes`;
            any other end is a termination. `infos[agent]` has `winner` (0 tie,
            1 player 1, 2 player 2, None while running), `winner_agent`
            (`"player_0"`, `"player_1"` or None for a tie or a running game),
            `injury` (damage this agent has taken), `opponent_injury` and
            `match_frame`.
        """
        if not self._started:
            raise RuntimeError("`step` was called before `reset`.")
        if not self.agents:
            raise RuntimeError("The episode has ended, call `reset`.")

        self.game.make_actions(merge_actions(actions))
        state, terminal = self.game.get_state()

        reward = float(self._reward_function(self._old_state, state))
        self._old_state = state
        rewards = {AGENTS[0]: reward, AGENTS[1]: -reward}

        truncated = bool(terminal and state.match_frame >= state.match_length)
        terminated = bool(terminal and not truncated)
        winner_agent = {1: AGENTS[0], 2: AGENTS[1]}.get(state.winner)
        infos = {
            agent: {
                "winner": state.winner,
                "winner_agent": winner_agent,
                "injury": float(state.injuries[player]),
                "opponent_injury": float(state.injuries[1 - player]),
                "match_frame": state.match_frame,
            }
            for player, agent in enumerate(AGENTS)
        }
        if terminal:
            self.agents = []
        return (
            self._observations(state),
            rewards,
            dict.fromkeys(AGENTS, terminated),
            dict.fromkeys(AGENTS, truncated),
            infos,
        )

    def close(self):
        """Stop the game and restore the game directory."""
        if self._started:
            self._started = False
            self.agents = []
            self.game.close()
