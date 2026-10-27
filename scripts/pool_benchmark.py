"""Aggregate turns/s of N direct-launch instances (needs the Steam client running).

    uv run python scripts/pool_benchmark.py 1 2 4 [--seconds 25] [--size 128] [--no-capture]

For each N: make an `InstancePool(N)` (copies of the game, one Xvfb each), start one worker
process per instance, wait until all games run, then let all of them play random
`TwoPlayerToriEnv` episodes (`draw_game=True`, `turnframes=5`) for the same time window and
sum the turns. Prints one JSON line per N with the load average before and after, since the
machine is usually shared. Direct instances use their own copies and do not take
`/tmp/toribash-game.lock`.
"""

import argparse
import json
import os
import subprocess
import sys
import time

from torille.envs.two_player import AGENTS, TwoPlayerToriEnv
from torille.pool import InstancePool


def worker(kwargs, seconds, size):
    env = TwoPlayerToriEnv(
        draw_game=True,
        capture_size=size,
        matchframes=1000,
        turnframes=5,
        **kwargs,
    )
    try:
        env.reset(seed=0)
        print("ready", flush=True)
        sys.stdin.readline()  # all workers start measuring together
        turns = 0
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            actions = {a: env.action_space(a).sample() for a in AGENTS}
            _, _, _, truncations, _ = env.step(actions)
            turns += 1
            if all(truncations.values()):
                env.reset()
        elapsed = time.perf_counter() - start
        print(json.dumps({"turns": turns, "seconds": elapsed}), flush=True)
    finally:
        env.close()


def measure(n, seconds, size):
    load_before = os.getloadavg()[0]
    with InstancePool(n) as pool:
        workers = [
            subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "worker",
                    json.dumps(kwargs),
                    str(seconds),
                    str(size) if size else "0",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            for kwargs in pool
        ]
        try:
            for w in workers:
                if w.stdout.readline().strip() != "ready":
                    raise RuntimeError("a worker failed to start its game")
            for w in workers:
                w.stdin.write("go\n")
                w.stdin.flush()
            results = [json.loads(w.stdout.readline()) for w in workers]
        finally:
            for w in workers:
                w.stdin.close()
                try:
                    w.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    w.kill()
        disk = pool.disk_bytes
        method = pool.copy_method
    rates = [r["turns"] / r["seconds"] for r in results]
    print(
        json.dumps(
            {
                "instances": n,
                "capture_size": size or None,
                "per_instance_turns_per_s": [round(r, 2) for r in rates],
                "aggregate_turns_per_s": round(sum(rates), 2),
                "load_before": round(load_before, 2),
                "load_after": round(os.getloadavg()[0], 2),
                "disk_gb": round(disk / 1e9, 2),
                "copy_method": method,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(json.loads(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]) or None)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("instances", type=int, nargs="+")
        parser.add_argument("--seconds", type=float, default=25)
        parser.add_argument("--size", type=int, default=128)
        parser.add_argument("--no-capture", action="store_true")
        args = parser.parse_args()
        for count in args.instances:
            measure(count, args.seconds, None if args.no_capture else args.size)
