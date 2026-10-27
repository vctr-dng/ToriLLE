# envs

ToriLLE comes with OpenAI Gym environment and several pre-defined tasks, which can be used
as a (almost) drop-in replacements to e.g. Atari environments.

Register these environments by importing `torille.envs`.

Limitations:
* `render()` function is not implemented (only one type of state available)

## Observation type

By default every environment returns its own flat `Box` observation, described per
environment below. Pass `observation_type="dict"` to get a structured `Dict` observation
instead; the action space and rewards do not change. Every environment id accepts the flag:

```python
env = gym.make("Toribash-DestroyUke-v2", observation_type="dict")
```

The `Dict` observation is the same for all environments and describes both players
(index 0 is player 1, index 1 is player 2), all float32. Defined in
`torille.envs.observations`:

Key | Shape | Bounds | Meaning
--- | ----- | ------ | -------
`positions` | (2, 21, 3) | +-inf | Limb positions of both players, centred on player 1's groin and rotated by its groin rotation (same frame as the flat Uke/Duo observation)
`groin_height` | (2,) | +-inf | Absolute `z` of each groin (centred positions lose the height above the floor)
`velocities` | (2, 21, 3) | +-inf | Limb velocities, world frame
`joint_states` | (2, 20) | [0, 4] | Joint states: 1 extend, 2 contract, 3 hold, 4 relax
`grips` | (2, 2) | [0, 1] | Left and right hand grip
`match_frame` | (1,) | [0, `matchframes`] | Frames played so far
`frames_next_turn` | (1,) | [0, `matchframes`] | Frames the next turn lasts

Note that `match_frame` advances by `turnframes + 1` per turn in the native game (for
example 0, 6, 12 with `turnframes=5`).

## Seeding and determinism

`reset(seed=...)` and `seed(...)` follow the Gymnasium convention and seed `env.np_random`.
That generator is the only source of randomness in ToriLLE: it draws the random opponent's
actions in `Toribash-DestroyUke-v2`. (Before, that opponent used Python's global `random`,
which could not be seeded through the environment.) Seed on the first `reset()`.

Toribash itself has no seed. What was measured (native game, Toribash build 20840792,
`Toribash-DestroyUke-v0`, random actions fixed beforehand, `matchframes` 200, 40 turns):

* The initial state after `reset()` is identical every time (bitwise). The classic mod has
  no random starting pose.
* Replaying the same action sequence gave bitwise identical observations and rewards,
  across episodes in one process, across separate processes, and with `draw_game` on or off.
  Measured divergence: 0.0.
* Hence an episode is a deterministic function of the settings and the actions, and with
  `np_random` seeded the whole of `Toribash-DestroyUke-v2` is reproducible too.

Limits of this claim: it covers only the settings and mod tested (classic, default
physics, two or three action sequences), one machine and one game build. It does not cover
other mods, replays, or a different game version or hardware (floating point results could
differ). `reset()` is only allowed after an episode ended (terminated or truncated).
The tests are in `tests/test_native_integration.py` (`pytest -m native`).

Additional functions/modifications for all environments:
* `set_draw_game(draw)`: Enables/Disables rendering of the game according to boolean parameter. Must be called before the first `reset()`.
* `settings` variable: This is a `ToribashSettings` object used to set Toribash's settings on each reset.

`step()` follows the Gymnasium API and returns `(observation, reward, terminated, truncated, info)`:
* `truncated` is `True` when the game ended by reaching `matchframes`. `terminated` is `True` when it ended any other way.
* `info` contains `winner` (`0` tie, `1`/`2` player; `None` unless the game ended), `injuries` (array of both players' injuries) and `match_frame`.

## Solo environments `torille.envs.solo_envs.SoloToriEnv`

These are tasks where only one character exists (observations/actions only include one player).

Player 2 is set to be immobile and engagement distance is set high to avoid contact between players.

**States**: 1D vector of player 1 body part positions w.r.t player 1's groin. The `z` coordinate
is replaced with absoluate `z` to inform agent how far from the ground it is. (`gym.spaces.box.Box`).

**Actions**: Joint states for player 1 (`gym.spaces.multi_discrete.MultiDiscrete`).

Reward is specified by the task (see below).

Settings:

Setting | Value
------- | -----
matchframes | 1000
turnframes | 5
engagement_distance | 1500

#### `Toribash-RunAway-v0`

Reward function: Positive reward for head body-part moving away from the center. See `torille.envs.solo_envs.reward_run_away`.

#### `Toribash-SelfDestruct-v0`

Reward function: Positive reward for damaging the player itself (not the opponent). See `torille.envs.solo_envs.reward_self_destruct`.

#### `Toribash-StaySafe-v0`

Reward function: Negative reward for damaging the player itself (not the opponent). See `torille.envs.solo_envs.reward_stay_safe`.


## Uke environments `torille.envs.uke_envs.UkeToriEnv`

Tasks where only one character is controlled by agent, but observations for both characters are provided.

Player 2 is set to be immobile or random, depending on the task.

**States**: 1D vector of player 1 and player 2 body part positions, w.r.t player 1's groin.
The `z` coordinate of player 1's groin is replaced with absolute `z` coordinate. (`gym.spaces.box.Box`)

**Actions**: Joint states for player 1 (`gym.spaces.multi_discrete.MultiDiscrete`)

Reward is specified by the task (see below).

Settings:

Setting | Value
------- | -----
matchframes | 1000
turnframes | 5

#### `Toribash-DestroyUke-v0`

Reward function: Positive reward for damaging immobile opponent. See `torille.envs.uke_envs.reward_destroy_uke`.

#### `Toribash-DestroyUke-v1`

Reward function: Positive reward for damaging immobile opponent and negative reward for receiving damage, summed together.
                 See `torille.envs.uke_envs.reward_destroy_uke_with_penalty`.

#### `Toribash-DestroyUke-v2`

Reward function: Positive reward for damaging opponent and negative reward for receiving damage, summed together.
                 **Opponent takes random actions each turn**. See `torille.envs.uke_envs.reward_destroy_uke_with_penalty`.


## Duo environments `torille.envs.duo_envs.DuoToriEnv`

Tasks where both characters are controlled by agent and observations are provided for both characters.

**States**: 1D vector of player 1 and player 2 body part positions, w.r.t player 1's **and** player 2's groin (i.e. 2x longer vector than in solo/uke environments).
The `z` coordinate of player's groins is replaced with the absolute `z` coordinate, so agents know their location w.r.t floor. (`gym.spaces.box.Box`)

**Actions**: Joint states for player 1 and player 2 (`gym.spaces.multi_discrete.MultiDiscrete`)

Reward is specified by the task (see below).

Settings:

Setting | Value
------- | -----
matchframes | 1000
turnframes | 5

#### `Toribash-DuoInjuryCombat-v0`

Reward function: Score from the point of view of player 1, in terms of injury:
                 Positive reward if opponent received damage,
                 negative if player 1 received damage (summed together).
                 See `torille.envs.duo_envs.reward_injury_player1_pov`.

#### `Toribash-DuoWinCombat-v0`

Reward function: Score from the point of view of player 1, in terms of winning:
                 +1 reward if player 1 won the game,
                 -1 reward if player 2 won the game and
                 0 reward if game was tie.
                 See `torille.envs.duo_envs.reward_win_player1_pov`.

#### `Toribash-Cuddles-v0`

Reward function: Positive reward relative to inverse of distance between two players (distance of center-of-masses).
                 Negative reward if either of players takes damage. These are summed together for final reward.
                 See `torille.envs.duo_envs.reward_cuddles`.

Setting | Value
------- | -----
turnframes | 2
