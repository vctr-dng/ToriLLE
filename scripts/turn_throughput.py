"""Turns/s benchmark for the native bridge (same method as the M0 benchmark).

    flock /tmp/toribash-game.lock uv run python scripts/turn_throughput.py \
        --seconds 30 --out results.json [--legacy]

Random actions in Toribash-DestroyUke-v0 for a fixed wall-clock budget per
(draw_game, turnframes) cell. Needs a display and Steam; launches the game.
"""

import argparse
import itertools
import json
import os
import time

import gymnasium as gym

import torille.envs  # noqa: F401


def bench(draw_game, turnframes, matchframes, seconds, legacy):
    env = gym.make(
        "Toribash-DestroyUke-v0", matchframes=matchframes, turnframes=turnframes
    )
    env.unwrapped.set_draw_game(draw_game)
    if legacy:
        # Behaviour before the turn-overhead work: sleep-polling every 10 ms and
        # no in-hook action pickup in the Lua script.
        game = env.unwrapped.game
        game.use_inotify, game.poll_interval, game.spin_ms = False, 0.01, 0.0
    result = {
        "draw_game": draw_game,
        "turnframes": turnframes,
        "matchframes": matchframes,
        "legacy_bridge": legacy,
    }
    try:
        launch_start = time.perf_counter()
        env.reset()
        result["launch_s"] = time.perf_counter() - launch_start

        steps = episodes = 0
        reset_times = []
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            if terminated or truncated:
                episodes += 1
                reset_start = time.perf_counter()
                env.reset()
                reset_times.append(time.perf_counter() - reset_start)
        elapsed = time.perf_counter() - start
    finally:
        env.close()

    result.update(
        elapsed_s=elapsed,
        turns=steps,
        episodes=episodes,
        turns_per_s=steps / elapsed,
        frames_per_s=steps * turnframes / elapsed,
        mean_reset_s=sum(reset_times) / len(reset_times) if reset_times else None,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--matchframes", type=int, default=1000)
    parser.add_argument("--turnframes", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--draw", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--legacy", action="store_true", help="old bridge settings")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = []
    for draw, turnframes in itertools.product(args.draw, args.turnframes):
        result = bench(
            bool(draw), turnframes, args.matchframes, args.seconds, args.legacy
        )
        result["display"] = os.environ.get("DISPLAY")
        print(json.dumps(result), flush=True)
        results.append(result)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
