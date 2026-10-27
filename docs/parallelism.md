# Parallel Toribash instances (M1 spike)

Date: 2026-10-27. Machine: i9-12900H (20 threads), 31 GB RAM, Intel Iris Xe + RTX A2000 Laptop (driver 595.99), Linux 7.0, Steam client on `:0`.

## Verdict

**N > 1 works, without a second Steam account.** The "one instance" limit comes from ToriLLE's launch path (`steam -applaunch` plus one shared game directory), not from the game or its licence.

Conditions:

- The user's normal Steam client must stay running and logged in. The game needs it only for Steamworks IPC (`SteamAPI_Init`). Without a running Steam the game exits with `SteamAPI_Init() failed`.
- Each instance needs its own copy of the game directory (597 MB here, with its own `.torille/` bridge and `.launchlock`).
- Each instance is started directly: `cd <copy> && DISPLAY=<d> SteamAppId=248570 SteamGameId=248570 ./toribash_steam`.
- Tested up to N=8 on one Xvfb screen. N=8 used about 45% CPU in total, so the limit was not reached. Roughly N=12-16 before CPU saturates is an extrapolation, not a measurement.

This launches a Steam game outside the Steam launcher, using the same user's own licence, on the same machine, with the same client. I did not read the Toribash terms; the user should decide whether that is acceptable.

## Evidence

| # | Question | Result |
|---|---|---|
| 1 | Does Steam refuse a second `steam -applaunch 248570`? | Effectively yes. With a game up (Steam-launched or a direct copy), a further `steam -applaunch 248570` started no new process (checked after 10-20 s). ToriLLE also refuses itself: `_launch_native_game` raises if a process with the same executable path exists. So `-applaunch` gives exactly one instance. |
| 2 | Why does `./toribash_steam` "hand off to Steam"? | The binary links `libsteam_api.so` and imports `SteamAPI_RestartAppIfNecessary`. Without `SteamAppId`/`SteamGameId` in the environment it relaunches through Steam. With `SteamAppId=248570` set, it starts in place. No LD_PRELOAD or `steam_appid.txt` was needed. |
| 3 | Does a direct launch work against a copy? | Yes. `SteamAPI_Init(): Loaded .../linux64/steamclient.so OK`, the window appears, ToriLLE drives it (`scripts/parallelism_spike.py`). The real game directory was untouched. |
| 4 | Is Steam needed at all? | Yes. With a fake `HOME` (no Steam pipe): `SteamAPI_IsSteamRunning() did not locate a running instance of Steam`, `SteamAPI_Init() failed`, exit 1. |
| 5 | Can a Steam-launched game and a direct copy run together? | Yes, both stayed alive. |
| 6 | Does Xvfb rendering work? | Yes. On Xvfb the game reports `OpenGL 4.5 (Compatibility Profile) Mesa 25.2.8` (llvmpipe, software, on the CPU); on `:0` it reports Mesa 4.6 (Intel iGPU, not the NVIDIA GPU). Shaders compile and the turn rate is the same as on `:0`. The FAQ's Xvfb-with-NVIDIA warning does not apply because Xvfb never uses NVIDIA GLX. `glxinfo` is not installed; I used the game's own GL banner. |
| 7 | Is a second window on `:0` technically fine? | Yes (two and four copies on `:0` ran fine), but the windows land on the user's desktop, so Xvfb is the better default. |

## Throughput

Random actions, `Toribash-DestroyUke-v0`, `draw_game=True`, `turnframes=5`, `matchframes=1000`, 20-25 s per cell. Per-instance turns/s; all instances on the given display.

The machine was **not quiet**: other agents ran PPO and baseline jobs (load average 5-12 while I measured), so absolute numbers vary by about 2x between runs. Compare within a series, not across.

Series 1 (load about 5):

| Instances | Display | per-instance turns/s | aggregate |
|---|---|---|---|
| 1 | `:0` | 4.1 | 4.1 |
| 1 | Xvfb | 6.1 | 6.1 |
| 2 | `:0` + Xvfb | 5.8 / 5.4 | 11.2 |
| 2 | Xvfb | 5.3 / 5.5 | 10.9 |
| 2 | `:0` | 5.7 / 5.7 | 11.4 |

Series 2 (quieter):

| Instances | Display | per-instance turns/s | aggregate |
|---|---|---|---|
| 4 | Xvfb | 9.8 (all four) | 39.2 |
| 8 | Xvfb | 8.2-9.1 | 69.9 |
| 4 | `:0` | 9.6-10.2 | 39.5 |
| 1 | Xvfb | 10.2 | 10.2 |
| 2 | Xvfb | 7.0 / 6.9 | 13.8 |
| 1 | `:0` | 5.7 | 5.7 |

Series 3 (load average 9-12):

| Instances | Display | per-instance turns/s | aggregate |
|---|---|---|---|
| 1 | Xvfb | 5.5 | 5.5 |
| 2 | Xvfb | 5.8 / 6.0 | 11.8 |
| 4 | Xvfb | 7.7-8.0 | 31.3 |
| 8 | Xvfb | 4.2-4.7 | 35.7 |

Reading: aggregate throughput scales nearly linearly to N=4 and is still 4-7x one instance at N=8, even with other jobs on the machine. Per-instance rate drops mainly when the machine is loaded by other work. CPU sampled once during N=8: about 45% user in total. GPU utilisation samples (30-60%) are dominated by the other agents and the desktop; Xvfb instances use software GL, not the NVIDIA GPU. A clean quiet-machine sweep (N=1/2/4/8/12, `draw_game=False`) is still to do.

## Commands

```bash
# one copy per instance, outside the repo (never commit)
cp -a ~/.local/share/Steam/steamapps/common/Toribash ~/ai-fighters/.spike-copies/copyA
Xvfb :99 -screen 0 1280x720x24 &
# drive N workers (one env each) for 25 s; wrap in the shared game lock
flock /tmp/toribash-game.lock uv run python scripts/parallelism_spike.py drive 25 \
    ~/ai-fighters/.spike-copies/copyA:99 ~/ai-fighters/.spike-copies/copyB:99
```

`scripts/parallelism_spike.py` monkeypatches `ToribashControl._launch_native_game` to Popen the copy's executable with `DISPLAY` and `SteamAppId`. ToriLLE's `executable=` argument already puts the bridge, `.launchlock` and `stderr.txt` in the copy's directory, so no library change was needed for the spike.

## What a library change would look like (not done)

`ToribashControl` needs a direct-launch mode (for example `display=` or `direct_launch=True`, or implied by a non-default `executable`): skip `steam -applaunch`, Popen the binary with `SteamAppId`, `SteamGameId` and the chosen `DISPLAY`, and keep the `Popen` handle for cleanup. Add a small pool helper that creates and removes the copies (about 600 MB each). `_native_game_pids` keys on the realpath of the executable, so it already works per copy.

## Not tried

- A second Steam account, or a second Steam client under another `HOME`/Xvfb (not needed; no spare account).
- Running the user's own Steam under Xvfb.
- Whether direct-launched copies appear in the user's Steam status or affect Steam cloud sync (`steam_autocloud.vdf` is in each copy).
- N > 8, `draw_game=False` at N > 1, `turnframes=1`, hours-long stability, memory per instance (not recorded; 31 GB total).
- Reading the Toribash licence terms.
- GL on the NVIDIA GPU (needs a real X server on it; software GL was fast enough).

## What the user must decide or provide

- Accept the direct-launch route (Steam client used for IPC only). No second account is required.
- Keep the desktop Steam client running during parallel runs.
- Pick N (suggestion: start at 4, measure on a quiet machine, go higher if CPU allows) and allow 0.6 GB of disk per instance.

If the direct-launch route is rejected, N=1, with these consequences:

- About 10 turns/s at best (about 36000 turns/hour), so dataset size per hour of wall time is fixed by that.
- Larger `turnframes` and fewer, shorter episodes are the only levers; budget datasets in the low hundreds of thousands of turns.
- M9 self-play is limited to short runs; no population-based training or large league.
- Each experiment competes with every other agent for the one game, so schedule the heavy runs.
- A second Steam account with a Toribash licence would be the only other route to N>1.

## Follow-up: direct launch and instance pool in the library

Implemented after the spike, opt-in (`launch_mode="direct"`); the default Steam launch is unchanged. API and details are in
[envs.md](envs.md#parallel-instances-opt-in-direct-launch). Summary:

```python
from torille.envs import TwoPlayerToriEnv
from torille.pool import InstancePool

# Steam client running and logged in; display="xvfb" is the default: one Xvfb per instance, nothing on your desktop.
with InstancePool(4) as pool:
    envs = [TwoPlayerToriEnv(draw_game=True, capture_size=128, **kw) for kw in pool]
    ...  # one thread/process per env: env.reset(), env.step(...), env.close()
```

* **Requirements.** Linux, the Steam client running and logged in (IPC only; without it the game exits and ToriLLE raises a
  `RuntimeError` naming Steam), the game installed, `Xvfb` installed, free disk of at least 3x the copies.
* **Run everything on Xvfb.** `InstancePool(n)` defaults to `display="xvfb"`: each instance gets its own 1280x720x24 Xvfb on a free display
  number (searched from `:100`), so windows never appear on the desktop and the WM-class window lookup of the capture is unambiguous.
  Passing `display=":0"` is possible but puts every game window on that display; do not use it for unattended runs. To be sure, run with
  `DISPLAY` unset: `env -u DISPLAY uv run python your_script.py` (direct mode needs no `$DISPLAY`; it sets `DISPLAY` for the game from
  `display=`). The native test (`tests/test_native_parallel.py`) and `scripts/pool_benchmark.py` refuse `:0` and `$DISPLAY`.
* **Disk.** 0.65 GB per instance (measured: 640 MB of game files, 0.65 / 1.29 / 2.58 GB for N = 1 / 2 / 4, plain copies; this filesystem
  is ext4 so reflinks are not available). Copies are made with `shutil` (about 1.5 s per instance with a warm cache), removed on exit.
  Hardlinks are not used because it was not shown that the game never rewrites those files in place.
* **Lock.** Direct instances use `.launchlock` and the "already running" check of their own directory and never take
  `/tmp/toribash-game.lock`.

### Throughput on Xvfb with capture

`scripts/pool_benchmark.py 1 2 4 --seconds 25` (`TwoPlayerToriEnv`, random actions, `draw_game=True`, `capture_size=128`, `turnframes=5`,
`matchframes=1000`, Xvfb with llvmpipe, one worker process per instance, all measuring the same window), 2026-10-27, i9-12900H:

| Instances | per-instance turns/s | aggregate turns/s | load average before -> after |
|---|---|---|---|
| 1 | 3.0 | 3.0 | 1.9 -> 2.1 |
| 2 | 1.6 / 1.7 | 3.3 | 2.1 -> 6.9 |
| 4 | 1.07 (all four) | 4.3 | 6.9 -> 21 |

These numbers are **not** a measure of what the library can do. The machine was shared with other jobs while I measured and the load
average was 2 at the start and up to 21 during the N=4 cell (the games' software rendering uses many threads each). Further cells, all
also under load 8-20: with `LP_NUM_THREADS=2` (capture) N=1/2/4 gave 2.0 / 2.6 / 4.5 aggregate, and without capture N=1/4 gave 2.6 / 4.5
aggregate, so on that busy machine neither limiting llvmpipe threads nor switching capture off changed the picture, which indicates
CPU contention. The quiet-machine spike (no capture, no pool) reached 10 turns/s per instance at N=1 and 39 turns/s aggregate at N=4;
that run should be repeated with capture and the pool on an idle machine. I did not do so, because the user asked me to stop. Do not quote the table
above as the scaling of the method.

### Not tested

* The Steam-launched (default) path on the final code: its native tests were not re-run after the last change (the games opened on the user's
  desktop and the run was stopped). It is covered by unit tests with fakes (`tests/test_platform_launcher.py`, `tests/test_cleanup.py`).
  Earlier on the same code, `pytest -m native` (Steam mode) passed `test_native_capture.py` and `test_native_integration.py`
  (14 tests) in one run; an earlier run had failed `test_bridge_does_not_desync_over_hundreds_of_turns[False-1]` (one turn off under heavy
  load) which passed when rerun alone. That run was interrupted before `test_native_parallel.py` and `test_native_two_player.py`;
  `test_native_parallel.py` was then run alone, on Xvfb with `DISPLAY` unset, and passed (2 instances, capture, 13 s).
* N > 4 with the pool, `draw_game=False` with the pool, hours-long runs, memory per instance, `display=":0"` with the pool, a Steam client
  that restarts during a run, Python killed with SIGKILL (games and Xvfb are left behind; the pool cannot clean up), concurrent
  Steam-mode and direct-mode capture sharing `jointflash` restoration, reflink copies (no reflink filesystem here).
* Whether starting the game outside the Steam launcher is acceptable under the Toribash / Steam terms: **not reviewed**. This uses the
  user's own licence and Steam client on one machine; the user should decide.
