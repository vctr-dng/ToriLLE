### Unreleased

* Optional RGB frame capture of the game window (Linux/X11), aligned to turn
  boundaries: `capture_size=N` adds an `rgb` key (N x N x 3 uint8) to the dict
  observation, `render_mode="rgb_array"` makes `render()` return the latest
  frame, `capture_crop` is `"center"` or `"letterbox"`, `capture_display`
  picks the X display. Needs `draw_game=True`, which is now also a constructor
  option of the environments. In capture mode the controller script fixes the
  camera and switches off the HUD, the ghost and the joint flash. Off by
  default with no cost. See docs/envs.md and `scripts/capture_benchmark.py`.
  `render()` without a render mode now returns `None` instead of raising.
* Lower native per-turn overhead (turns/s on DestroyUke, random actions:
  40 -> 67 with `draw_game=False, turnframes=1`, 18 -> 31 with `draw_game=True`,
  19 -> 24 with `turnframes=5`; see docs/hacking.md). Python now wakes on an inotify
  event instead of sleeping 10 ms, and the Lua script picks up the action inside the
  freeze hook for `spin_ms` (default 5) before falling back to its per-frame poll.
  New `ToribashControl` / `NativeFileBridge` arguments `poll_interval`, `use_inotify`
  and `spin_ms`. Smaller poll intervals alone and fewer file operations were measured
  and did not matter beyond that.
* New two-player control API, `torille.envs.TwoPlayerToriEnv` (PettingZoo-style
  parallel API without the dependency): both players' actions come from
  outside, each agent (`player_0`, `player_1`) gets its own mirrored `self` /
  `opponent` observation, a zero-sum reward (`"injury"` or `"win"`, or a
  callable), done flags and the winner in `info`. See docs/envs.md and
  `examples/two_player_example.py`.
* New `observation_type="dict"` flag for all environments: a `Dict` observation
  with positions, velocities, joint states, grips, `match_frame` and
  `frames_next_turn` for both players (see docs/envs.md). The default
  observation is unchanged.
* `reset(seed=...)` and `seed()` seed `np_random`, which now drives the random
  opponent of `Toribash-DestroyUke-v2`. Documented (and tested) that the game
  itself is deterministic: same actions give identical trajectories.
* Require Python 3.14, manage the project with uv and migrate to Gymnasium.
* Launch native Toribash on Linux through Steam, controlled via a file bridge
  (`.torille/` in the game directory) instead of LuaSocket.
* Gym environments return `terminated`/`truncated` separately and report
  `winner`, `injuries` and `match_frame` in `info`.
* Native state messages are only accepted when complete, and a crashed game
  is detected immediately instead of after the connection timeout.
* Native controller script and bridge files are removed on `close()`, and also
  when Python exits without it (Ctrl-C, crashes, `sys.exit`) or the launch fails.
  Killing Python with SIGTERM still bypasses this cleanup.
* Settings validation reports the correct setting name and accepts numpy numbers.
* `torille.__version__` is read from the package metadata.
* `draw_game=False` now runs the native game without the fixed frame rate
  (about 2x faster than real time). The window is still shown.
* Fixed the native controller stalling after a reset when it read a consumed,
  empty action file as an action.
* Tests run with pytest; ruff and pytest are in the uv dev group. Tests that
  play real games are opt-in with `pytest -m native`.

### 31.8.2023 v1.0.3

* Update code to work with new Gym and numpy.
* Include Toribash binaries in the repo to make installation easier.

### 19.9.2019 v1.0.2

* Added state-machine to make sure actions/states are handled in right order.

### 5.4.2019 v1.0.1

* Added (experimental) code for going through replay files
* Added new experiment codes (stable_baselines and rlkit experiments)

### 18.2.2019 v1.0.0

* Added possibility to "remote control" in multiplayer games
  * Done via "manual_torille.py", where human player must manually
    launch the script in Toribash, which then connects to the remote
    controller
* Refactoring code to avoid one huge file
* Added match length, current frame and frames next turn to state
* Added possibility to change game mod
* Fixed off-by-one error with turnframes which prevented from using turnframes=1
* Updated pypi-package to use Toribash 5.4

### 8.11.2018 v0.9.9

* Fixed Gym environments changing actions list in-place during call to `step`
* Added info on game winner to ending state
* Added rotation matrix of players' groins to game state
* Added velocities of body parts to game state
* Added "get_normalized_locations" function to game state
* Added normalization to Gym environments
* Added new duo-combat environment with winning reward (rather than injury reward)
* Torille now attempts to make stderr.txt file read-only to avoid huge files (on Linux/Darwin)

* Added code of new experiments to `experiments` folder


### 20.8.2018 v0.9.5

* Added support for Mac/OSX (with Windows binaries, *durr*)

### 20.8.2018 v0.9.3 and v0.9.4

* Fixed Gym environments on Windows

### 28.7.2018 v0.9.2

* Restructured Gym environments and added some more
* Added limit for turnframes [2,matchframes]. Later turnframes=1 might be made possible
* Fixed "static" frames at the start of learning: First few states were always the same
* Added sanity check for available displays on Linux (for $DISPLAY="")
* FPS on Windows 10 improved by ~10%

### 27.7.2018 v0.9.1

* Removed resolution changes. This caused crashing on some Linux and Windows systems
* Fixed compatibility version checking

### 26.7.2018 v0.9.0

Initial public release.
