"""Structured (`Dict`) observations for the Toribash environments.

Enabled with the `observation_type="dict"` constructor flag of the
environments (for example `gym.make("Toribash-DestroyUke-v2",
observation_type="dict")`). The default (`"box"`) keeps the original flat
observation of each environment.

All arrays are float32 and have a leading player axis of length 2
(index 0 is player 1, index 1 is player 2).
"""

import numpy as np
from gymnasium import spaces

from .. import constants

OBSERVATION_TYPES = ("box", "dict")


def dict_observation_space(match_frames):
    """
    Space of the `Dict` observation.

    Keys:
        positions: (2, NUM_LIMBS, 3) limb positions of each player, centred on
            player 1's groin and rotated by player 1's groin rotation (the same
            frame as the flat observation of the Uke and Duo environments).
        groin_height: (2,) absolute z of each player's groin (the centred
            positions lose the height above the floor).
        velocities: (2, NUM_LIMBS, 3) limb velocities in the world frame.
        joint_states: (2, NUM_JOINTS) joint state, 1..4 (extend, contract,
            hold, relax); 0 is allowed for a joint with no state yet.
        grips: (2, 2) grip state of the left and right hand, 0 or 1.
        match_frame: (1,) frames played so far, in [0, match_frames].
        frames_next_turn: (1,) frames the next turn will last, in
            [0, match_frames].

    Positions and velocities are unbounded (+-inf): bodies can be flung
    arbitrarily far when joints are dismembered.
    """
    limbs = (2, constants.NUM_LIMBS, 3)
    return spaces.Dict(
        {
            "positions": spaces.Box(-np.inf, np.inf, limbs, np.float32),
            "groin_height": spaces.Box(-np.inf, np.inf, (2,), np.float32),
            "velocities": spaces.Box(-np.inf, np.inf, limbs, np.float32),
            "joint_states": spaces.Box(0.0, 4.0, (2, constants.NUM_JOINTS), np.float32),
            "grips": spaces.Box(0.0, 1.0, (2, 2), np.float32),
            "match_frame": spaces.Box(0.0, float(match_frames), (1,), np.float32),
            "frames_next_turn": spaces.Box(0.0, float(match_frames), (1,), np.float32),
        }
    )


def state_to_dict_observation(state):
    """Convert a `ToribashState` into an observation of `dict_observation_space`."""
    positions = state.get_normalized_locations()[0]
    groin_height = state.limb_positions[:, 4, 2]
    joints = state.joint_states[:, : constants.NUM_JOINTS]
    grips = state.joint_states[:, constants.NUM_JOINTS :]
    return {
        "positions": positions.astype(np.float32),
        "groin_height": groin_height.astype(np.float32),
        "velocities": state.limb_velocities.astype(np.float32),
        "joint_states": joints.astype(np.float32),
        "grips": grips.astype(np.float32),
        "match_frame": np.array([state.match_frame], dtype=np.float32),
        "frames_next_turn": np.array([state.frames_next_turn], dtype=np.float32),
    }
