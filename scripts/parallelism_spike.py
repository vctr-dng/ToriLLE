"""M1 parallelism spike: run Toribash from game-directory copies without `steam -applaunch`.

Each worker drives one ToriLLE env against its own copy of the game directory,
launching `toribash_steam` directly with SteamAppId set (the Steam client on :0
only provides the Steamworks IPC; the game renders on whatever DISPLAY we give it).

Worker:  python scripts/parallelism_spike.py worker COPY_DIR DISPLAY SECONDS
Driver:  python scripts/parallelism_spike.py drive SECONDS COPY:DISPLAY [COPY:DISPLAY ...]

Wrap the driver in `flock /tmp/toribash-game.lock` when other agents use the game.
Copies live outside the repo; never commit them.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym

import torille.envs  # noqa: F401
from torille import torille as tl


class _Proc:
    def __init__(self, popen):
        self.p = popen
        self.pid = popen.pid

    def is_alive(self):
        return self.p.poll() is None

    def kill(self):
        self.p.kill()
        self.p.wait()


def worker(copy_dir, display, seconds):
    exe = str(Path(copy_dir) / "toribash_steam")
    env = {
        **os.environ,
        "DISPLAY": display,
        "SteamAppId": "248570",
        "SteamGameId": "248570",
    }

    def launch(self):
        return _Proc(
            subprocess.Popen(
                (exe,),
                cwd=copy_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

    tl.ToribashControl._launch_native_game = launch
    turnframes = 5
    env_ = gym.make(
        "Toribash-DestroyUke-v0",
        executable=exe,
        matchframes=1000,
        turnframes=turnframes,
    )
    env_.unwrapped.set_draw_game(True)
    t0 = time.perf_counter()
    try:
        env_.reset()
        launch_s = time.perf_counter() - t0
        steps = 0
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            _, _, term, trunc, _ = env_.step(env_.action_space.sample())
            steps += 1
            if term or trunc:
                env_.reset()
        elapsed = time.perf_counter() - start
    finally:
        env_.close()
    result = {
        "copy": copy_dir,
        "display": display,
        "launch_s": round(launch_s, 2),
        "turns": steps,
        "turns_per_s": round(steps / elapsed, 2),
        "frames_per_s": round(steps * turnframes / elapsed, 1),
    }
    print(json.dumps(result))


def drive(seconds, specs):
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, "worker", *s.split(":"), str(seconds)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for s in specs
    ]
    for p in procs:
        out, _ = p.communicate()
        lines = out.strip().splitlines()
        print(lines[-1] if lines else f"worker failed rc={p.returncode}")


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    else:
        drive(float(sys.argv[2]), sys.argv[3:])
