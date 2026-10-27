"""Interleaved A/B of bridge wait modes inside one game launch.

Machine noise (other jobs, thermal) drifts between launches by more than the
effects we measure, so modes alternate in blocks of --block turns and are
compared per mode over many rounds.

    flock /tmp/toribash-game.lock uv run python scripts/ab_turns.py --draw 0 --turnframes 1
"""

import argparse
import statistics
import time

import gymnasium as gym

import torille.envs  # noqa: F401

MODES = {
    "sleep10ms": (False, 0.01),
    "sleep2ms": (False, 0.002),
    "sleep0.5ms": (False, 0.0005),
    "inotify": (True, 0.01),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--turnframes", type=int, default=1)
    parser.add_argument("--spin", type=float, default=0)
    parser.add_argument("--block", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    args = parser.parse_args()

    env = gym.make(
        "Toribash-DestroyUke-v2", matchframes=10**6, turnframes=args.turnframes
    )
    env.unwrapped.set_draw_game(bool(args.draw))
    game = env.unwrapped.game
    game.spin_ms = args.spin
    per_mode = {m: [] for m in args.modes}
    try:
        env.reset()
        for _ in range(30):
            env.step(env.action_space.sample())
        for _ in range(args.rounds):
            for mode in args.modes:
                bridge = game.native_bridge
                bridge.use_inotify, bridge.poll_interval = MODES[mode]
                start = time.perf_counter()
                for _ in range(args.block):
                    env.step(env.action_space.sample())
                per_mode[mode].append(1000 * (time.perf_counter() - start) / args.block)
    finally:
        env.close()
    print(
        f"spin={args.spin} draw={args.draw} turnframes={args.turnframes} block={args.block} rounds={args.rounds}"
    )
    for mode, values in per_mode.items():
        print(
            f"  {mode:12s} median {statistics.median(values):6.2f} ms/turn  "
            f"min {min(values):6.2f} max {max(values):6.2f}  turns/s {1000 / statistics.median(values):5.1f}"
        )


if __name__ == "__main__":
    main()
