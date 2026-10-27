#  gym_env.py
#  ToriLLE available as OpenAI Gym Environment
#
#  Shoutout to GitHub user "ppaquette" for Gym-Doom, which was used
#  as a base here.
#
#  Author: Anssi "Miffyli" Kanervisto, 2018
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

from copy import deepcopy
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .. import torille
from .observations import (
    OBSERVATION_TYPES,
    dict_observation_space,
    state_to_dict_observation,
)


class ToriEnv(gym.Env):
    """
    An (abstract) environment for Toribash environments.
    Needs implementations for:
        _preprocess_observation
        _preprocess_action
        _reward_function
    """

    # "box" (flat, per environment) or "dict" (see observations.py)
    observation_type = "box"
    metadata: ClassVar = {"render_modes": [None, "rgb_array"]}

    def __init__(
        self,
        executable=None,
        install_native_controller=True,
        observation_type="box",
        draw_game=False,
        render_mode=None,
        capture_size=None,
        capture_crop="center",
        capture_display=None,
        launch_mode="steam",
        game_dir=None,
        display=None,
        **kwargs,
    ):
        if observation_type not in OBSERVATION_TYPES:
            raise ValueError(
                f"observation_type must be one of {OBSERVATION_TYPES}, "
                f"got {observation_type!r}"
            )
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}, "
                f"got {render_mode!r}"
            )
        if capture_size is not None and observation_type != "dict":
            raise ValueError(
                "capture_size adds an `rgb` key to the observation and needs "
                "observation_type='dict'. For `render()` only, use "
                "render_mode='rgb_array'."
            )
        self.observation_type = observation_type
        self.render_mode = render_mode
        # Frames are captured for the `rgb` observation and for `render()`.
        self.capture_rgb_observation = capture_size is not None
        if render_mode == "rgb_array" and capture_size is None:
            capture_size = 256
        self.capture_size = capture_size
        self.settings = torille.ToribashSettings(**kwargs)
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

        # Previous state (ToribashState)
        # Used for reward function
        self.old_state = None

        # True if this object was created and there has been no calls to
        # ´step´ function.
        # This is to make sure first call to controller will be get_state()
        self.just_created = True

        self.action_space = spaces.MultiDiscrete(
            ([torille.constants.NUM_JOINT_STATES] * torille.constants.NUM_CONTROLLABLES)
            * 2
        )
        # For both players, position of all joints
        self._set_observation_space(
            spaces.Box(
                low=-30,
                high=30,
                dtype=np.float32,
                shape=(2, torille.constants.NUM_LIMBS * 3),
            )
        )

    def _set_observation_space(self, box_space):
        """
        Set `observation_space`: `box_space` (the environment's flat
        observation) by default, or the `Dict` space if the environment was
        created with `observation_type="dict"`. Subclasses call this instead
        of assigning `observation_space`.
        """
        if self.observation_type == "dict":
            self.observation_space = dict_observation_space(
                self.settings.get("matchframes"),
                self.capture_size if self.capture_rgb_observation else None,
            )
        else:
            self.observation_space = box_space

    def _observe(self, state):
        """Convert a ToribashState into an observation of `observation_space`."""
        if self.observation_type == "dict":
            observation = state_to_dict_observation(state)
            if self.capture_rgb_observation:
                observation["rgb"] = self._capture_frame(state).copy()
            return observation
        return self._preprocess_observation(state)

    def _preprocess_observation(self, state):
        """
        Preprocess ToribashState into more Numpyish
        Parameters:
            state: ToribashState
        Returns:
            observation: Object according to observation space
        """
        raise NotImplementedError

    def _preprocess_action(self, action):
        """
        Preprocess actions from agents into appropiate actions for Toribash
        Parameters:
            action: Action according to action space
        Returns:
            action: An appropiate action object for Torille (List of two Lists)
        """
        raise NotImplementedError

    def _reward_function(self, old_state, new_state):
        """
        Calculates reward
        Parameters:
            old_state: Old (previous) observation. None if `new_obs` is
                       first observation
            new_state: Current observation
        Returns:
            reward: Reward signal (float)
        """
        raise NotImplementedError

    def step(self, action):
        if self.just_created:
            # We cannot send action as a first call to controller
            # We should instead call `reset`.
            # Inform and fail
            raise RuntimeError(
                "`step` function was called "
                + "before calling `reset`. Call `reset` after creating "
                + "environment to get the first observation."
            )

        # Take copy of actions so we won't accidentally modify
        # it in place
        action = deepcopy(action)

        action = self._preprocess_action(action)
        self.game.make_actions(action)
        # Get new state
        state, terminal = self.game.get_state()
        # Compute reward (something we will define in other classes)
        reward = self._reward_function(self.old_state, state)
        # Remember to update the old state
        self.old_state = state
        # "obs" here is in the Gym format. "state" is ToribashState
        obs = self._observe(state)
        # Reaching the match length is a time limit (truncation). Any other
        # end of the game is a natural ending of the task (termination).
        truncated = terminal and state.match_frame >= state.match_length
        terminated = terminal and not truncated
        info = {
            "winner": state.winner,
            "injuries": state.injuries.copy(),
            "match_frame": state.match_frame,
        }
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = None
        if self.just_created:
            # Initialize game here to make it pickable before init
            self.game.init()
            # The env was just created and we need to start
            # by returning an observation.
            state, _terminal = self.game.get_state()
            self._frame = None
            obs = self._observe(state)
            # Remember to update the state here as well
            self.old_state = state
            self.just_created = False
        else:
            # Reset episode
            state = self.game.reset()
            self._frame = None
            # Get the state and return it
            obs = self._observe(state)
            self.old_state = state
        return obs, {}

    def set_draw_game(self, visibility):
        """
        Sets flag for drawing the game. Must be called before first reset()
        (drawing = render characters, limit FPS)
        Parameters:
            visibility: True = game will be drawn.
        """
        if not self.just_created:
            raise RuntimeError("Can't change rendering after first `reset()`")
        else:
            self.game.draw_game = visibility

    def _capture_frame(self, state):
        """Frame of the displayed state `state` (grabbed once per state)."""
        if self._frame is None or self._frame_match_frame != state.match_frame:
            self._frame = self.game.grab_frame()
            self._frame_match_frame = state.match_frame
        return self._frame

    def render(self):
        """
        With `render_mode="rgb_array"`: the `(capture_size, capture_size, 3)`
        uint8 frame of the current turn boundary (the state of the last
        `reset()` or `step()`). Without a render mode the game window is
        the display, see `set_draw_game`.
        """
        if self.render_mode != "rgb_array":
            return None
        if self.old_state is None:
            raise RuntimeError("`render` was called before `reset`.")
        return self._capture_frame(self.old_state).copy()

    def close(self, **kwargs):
        self.game.close()

    def seed(self, seed=None):
        """
        Reseed `np_random`, the generator for randomness in the environment
        (the random opponent of `Toribash-DestroyUke-v2`). Prefer
        `reset(seed=...)`. The game has no seed to set, see docs/envs.md.
        """
        self._np_random, self._np_random_seed = gym.utils.seeding.np_random(seed)
        return [self._np_random_seed]
