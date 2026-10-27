# Hacking & contributing to ToriLLE

This guide aims to offer quick trip to how to add and modify things in ToriLLE.

There are two parts that need modifying:

1. Toribash Lua script `remotecontrol.lua` (socket protocol) or, for native Linux Toribash,
   `torille/toribash/data/script/remotecontrol_native.lua` (file bridge)
2. The controller (e.g. the Python code `torille.py`). Not discussed here, as it is mostly trivial processing of the data.

## Toribash Lua script

Toribash codes reside in `toribash-codes` directory at root of this repository.

All interaction with Toribash is done via its Lua scripting capabilities.
Scripts are stored in `[toribash folder]/data/scripts` folder, and the main script used by ToriLLE is
`remotecontrol.lua`.
Additional Lua files remove UI elements and other startup things we do not want.
Lua library luasocket is used to handle communication over TCP/IP.

Modifying this side consists mainly of getting/setting the information you want with
Lua functions, and then using opened socket `s` to send/receive data to/from remote end.

Useful links:
* [Chat options/settings](http://forum.toribash.com/showthread.php?t=317900)
* [Raw list of (older) Toribash Lua functions](https://github.com/trittimo/ToriScriptAPI/blob/master/docs/toribash_docs.txt)
* Check `[toribash folder]/data/scripts/sdk` for examples on different Lua functions Toribash has.

## Native file bridge

The native Linux client cannot load LuaSocket, so `remotecontrol_native.lua` exchanges
messages with Python through files in `[toribash folder]/.torille/`. The payloads use the same
comma-separated formats as the socket [protocol](protocol/protocol.md).

File | Written by | Content
---- | ---------- | -------
`settings` | Python | Draw flag on the first line, comma-separated settings on the second
`state` | Lua | Current state, prefixed with `end:<winner>,` when the game is over
`action` | Python | Actions for both players
`reset` | Python | Request to start a new game with the current `settings`

Each reader consumes its file after reading it. Lua cannot rename files, so it rewrites `state`
in place. Python therefore only accepts a state once it ends in a newline.

## Per-turn overhead

Measured on 2026-10-27 with `scripts/turn_throughput.py` (`Toribash-DestroyUke-v0`, random actions,
`matchframes=1000`, 30 s per cell, `:0`), run interleaved before/after, twice each. The machine was shared with
other jobs (load 1.4-2.4), so treat differences under ~10% as noise.

| `draw_game` | `turnframes` | before (turns/s) | after (turns/s) | speedup |
|---|---|---|---|---|
| False | 1 | 40.4 / 39.3 | 65.8 / 68.0 | 1.68x |
| False | 5 | 18.3 / 20.3 | 23.7 / 24.6 | 1.25x |
| True | 1 | 17.8 / 17.6 | 30.7 / 30.9 | 1.74x |
| True | 5 | 8.6 / 8.5 | 10.3 / 10.3 | 1.20x |

"Before" is the old bridge (`use_inotify=False, poll_interval=0.01, spin_ms=0`, selectable with
`--legacy`). Note that one turn advances `turnframes + 1` game frames (`match_frame` goes 2, 4, 6 for
`turnframes=1`), so physics frames per second is `turns/s * (turnframes + 1)`. This quirk is older than this work and is
unchanged.

### Profile of a turn (before, `draw_game=False`, `turnframes=1`, about 21.5 ms)

Timestamps from Python plus inotify events on `.torille/` (`scripts/profile_turn.py`):

| Step | Time |
|---|---|
| Python writes the action (tmp file + rename) | 0.03 ms |
| Lua notices the action: it only looks in `pre_draw`, once per game frame (about 7.5 ms apart with `draw_game=False`, about 26 ms with `True`) | 2-6 ms |
| Lua applies the action, steps, game frame, `enter_freeze`, state written | about 15 ms |
| Python notices the state (10 ms sleep polling) | 4-5 ms (up to 10) |
| Python reads, deletes, parses the state | 0.06 ms |

So the fixed cost is not file operations (about 0.03 ms each) but two latencies: Python's sleep and the Lua
script only looking for actions once per frame. The remaining time is the game itself.

### Changes

* `ToribashControl(poll_interval=0.001, use_inotify=True, spin_ms=5.0)`, passed to `NativeFileBridge`.
  * `use_inotify`: block on a Linux inotify watch of `.torille/` (through `ctypes`, no dependency) instead of
    sleeping. Falls back to polling every `poll_interval` if inotify is unavailable. The watch exists before the first
    read, so a write between a read and the wait is not lost; the wait has a 50 ms timeout only to check that
    the game is alive.
  * `spin_ms`: written as a third line of the `settings` file. After writing the state in the `enter_freeze` hook,
    Lua keeps checking for the action for up to `spin_ms` milliseconds and steps immediately, instead of waiting for
    the next `pre_draw`. If Python answers later, the old per-frame path takes over, with identical results.
    The spin is not used in the `new_game` hook, because stepping from there advances a different number of frames.
* Not reachable from the Gym envs yet (the env layer is untouched); set attributes on
  `env.unwrapped.game` before the first `reset()`. The defaults are the fast ones.

### What was tried

Interleaved A/B in one game launch (`scripts/ab_turns.py`, 8 rounds of 100 turns, ms per turn, median):

| Variant | `draw_game=False` | `draw_game=True` |
|---|---|---|
| sleep 10 ms (old) | 25.1 | 56.0 |
| sleep 2 ms | 21.2 | 48.3 |
| sleep 0.5 ms | 19.9 | 48.2 |
| inotify | 20.8 | 47.9 |
| inotify + Lua spin (`turnframes=1`) | 13.5-14 | 32.1 |

* A shorter poll interval alone recovers most of the Python-side loss; inotify is no faster than 0.5 ms polling
  (the wake-up cost is similar). It is kept because it burns no CPU and has no interval to tune.
* The Lua spin is the larger win (saves most of the half-frame average pickup latency). Any window from 1 ms
  to 40 ms measured the same; 5 ms is the default so that a slow Python still falls back cheaply.
* Fewer file operations: not pursued. The write, read and delete cost about 0.03 ms each, against 20+ ms per turn.
* `draw_game=True` stays frame-limited (`fixedframerate=1`, about 26 ms per frame), and larger `turnframes`
  shrink the relative gain because per-frame time dominates.

## RGB capture notes

The capture path is `torille/capture.py` (window lookup and `XShmGetImage` through ctypes, crop and resize in numpy) and
`ToribashControl.grab_frame()`. The Lua side only changes when the settings file has a fourth line `1` (capture mode): then
`remotecontrol_native.lua` disables the HUD and the joint flash, and calls `place_camera()` (free camera, angle, look-at, position, ghost off)
from `pre_draw` and `new_game`. Details of the camera API, found by experiment on build 20840792:

* `set_camera_pos` / `set_camera_lookat` in free mode (`set_camera_mode(3)`) are not enough: the rendered direction follows
  `set_camera_angle`, and the game moves the look-at point (reading `get_camera_info()` shows it) after the hook that set it. Setting everything
  in `pre_draw` each frame is stable (readback unchanged over many turns).
* The `camera` hook did not help (frames went blank), and applying the camera once in `new_game` is overridden after the first turn.
* Ghost: a looping preview that keeps animating while the game waits. `set_ghost(0)` stops it.
* `jointflash` (an option saved in the user's config) makes the joint colour fade in over about 0.7 s after a turn.
  `trails`, `jointobjopacity`, `effects`, `motionblur` and `blood` did not remove that transient.
* The pause between a state and the picture of that state was measured with small region grabs right after `get_state()`: stale for up to
  about 15 ms after a turn and 25 ms after a reset. Publishing the state one frame later from Lua made no measurable difference, so a settle wait
  in Python is used instead.

## Tests

`uv run pytest` runs the unit tests. `uv run pytest -m native` also plays real games, which
needs Steam, Toribash and a `DISPLAY`. Only one Toribash instance can run at a time: `tests/conftest.py` takes
`/tmp/toribash-game.lock` for native tests, so do not wrap pytest in `flock` yourself (it would deadlock).

## Communication

See `protocol/` for detailed explanations on the messages between controller and Toribash Lua scripts.

Note that all messages should end with `\ņ` character both ways. TCP/IP may break single message into
multiple packets occasionally, so you may have to manually read till `\n` character is read (had to do with Python)

Be careful when coding these Lua scripts: Toribash may not print any errors on invalid calls and such,
so be sure to double-check your code before you call it a day!

Data is sent and received as comma-separated lists, ending to `\n`. No variable-names/keys were included
to save on the bandwidth.
