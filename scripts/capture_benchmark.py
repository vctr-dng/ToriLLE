"""Cost of RGB capture per turn, and turns/s with capture on and off.

    flock /tmp/toribash-game.lock uv run python scripts/capture_benchmark.py \
        [--seconds 20] [--turnframes 5] [--size 256] [--out results.json]

Launches the game once per cell. Needs Steam, Toribash and a display. Reports:

* the time of each capture stage (X11 grab, crop/resize) and of the same grab
  done with a fresh `ffmpeg -f x11grab` process per turn,
* the latency to read one frame from a persistent ffmpeg pipe (for comparison,
  not used by the package),
* turns/s of random play with capture off, on (`rgb` observation) and with
  `render()`.
"""

import argparse
import json
import statistics
import subprocess
import time

import gymnasium as gym

import torille.envs  # noqa: F401
from torille import capture


def stats(values):
    values = sorted(values)
    return {
        "mean_ms": 1000 * statistics.fmean(values),
        "median_ms": 1000 * statistics.median(values),
        "p95_ms": 1000 * values[int(0.95 * (len(values) - 1))],
    }


def timed(function, repeats):
    out = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        out.append(time.perf_counter() - start)
    return stats(out)


def stage_costs(size, repeats=30):
    """Run with the game on screen (after a reset)."""
    env = gym.make(
        "Toribash-DestroyUke-v2",
        observation_type="dict",
        capture_size=size,
        draw_game=True,
        turnframes=5,
    )
    env.reset()
    for _ in range(3):
        env.step(env.action_space.sample())
    game = env.unwrapped.game
    grabber = game._frame_capture._grabber
    width, height = grabber.size()
    x, y, side = capture.square_crop_box(width, height)
    result = {"window": [width, height]}
    result["xgetimage_square"] = timed(lambda: grabber.grab(x, y, side, side), repeats)
    result["xgetimage_full_window"] = timed(lambda: grabber.grab(), repeats)
    image = grabber.grab(x, y, side, side)
    result["resize_square_to_size"] = timed(
        lambda: capture.process_frame(image, size, "center"), repeats
    )
    result["grab_frame_total"] = timed(game.grab_frame, repeats)
    ffmpeg_repeats = 10
    result["ffmpeg_fresh_process_square"] = timed(
        lambda: capture.ffmpeg_grab(x, y, side, side), ffmpeg_repeats
    )
    # Persistent ffmpeg: one process streams the square, we read one frame
    # per "turn" (sleeping like the game does between actions).
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "x11grab",
        "-draw_mouse",
        "0",
        "-framerate",
        "30",
        "-video_size",
        f"{side}x{side}",
        "-i",
        f":0+{x},{y}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    frame_bytes = side * side * 3
    process.stdout.read(frame_bytes)

    def read_frame():
        time.sleep(0.05)
        process.stdout.read(frame_bytes)

    start = time.perf_counter()
    for _ in range(10):
        read_frame()
    result["ffmpeg_persistent_read_incl_50ms_sleep_ms"] = (
        1000 * (time.perf_counter() - start) / 10
    )
    process.kill()
    env.close()
    return result


def throughput(mode, size, turnframes, seconds):
    options = {"draw_game": True, "turnframes": turnframes, "matchframes": 1000}
    if mode == "rgb_obs":
        options.update(observation_type="dict", capture_size=size)
    elif mode == "render":
        options.update(render_mode="rgb_array", capture_size=None)
    env = gym.make("Toribash-DestroyUke-v2", **options)
    try:
        env.reset()
        steps = 0
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if mode == "render":
                frame = env.render()
                assert frame.shape[-1] == 3
            steps += 1
            if terminated or truncated:
                env.reset()
        elapsed = time.perf_counter() - start
    finally:
        env.close()
    return {"mode": mode, "turns": steps, "turns_per_s": steps / elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--turnframes", type=int, default=5)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = {"size": args.size, "turnframes": args.turnframes}
    result["stages"] = stage_costs(args.size)
    print(json.dumps(result["stages"], indent=1), flush=True)
    result["throughput"] = []
    for mode in ("off", "rgb_obs", "render", "off", "rgb_obs"):
        row = throughput(mode, args.size, args.turnframes, args.seconds)
        print(row, flush=True)
        result["throughput"].append(row)
    if args.out:
        with open(args.out, "w") as file:
            json.dump(result, file, indent=1)


if __name__ == "__main__":
    main()
