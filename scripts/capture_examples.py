"""Write a strip of three captured frames to docs/images/rgb_capture_example.png.

    flock /tmp/toribash-game.lock uv run python scripts/capture_examples.py

Plays `Toribash-DestroyUke-v0` with seeded random actions and keeps the frames
of turns 0, 10 and 30 (256x256 each, side by side). Needs Steam, Toribash and a
display.
"""

import struct
import sys
import zlib
from pathlib import Path

import gymnasium as gym
import numpy as np

import torille.envs  # noqa: F401

TURNS = (0, 10, 30)
OUT = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "images"
    / "rgb_capture_example.png"
)


def write_png(path, image):
    height, width, _ = image.shape
    raw = b"".join(b"\0" + image[row].tobytes() for row in range(height))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def main():
    env = gym.make(
        "Toribash-DestroyUke-v0",
        observation_type="dict",
        capture_size=256,
        draw_game=True,
        turnframes=5,
        matchframes=1000,
    )
    env.action_space.seed(0)
    frames = {}
    try:
        observation, _ = env.reset(seed=0)
        frames[0] = observation["rgb"]
        for turn in range(1, max(TURNS) + 1):
            observation, _, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            frames[turn] = observation["rgb"]
            if terminated or truncated:
                break
    finally:
        env.close()
    strip = np.concatenate([frames[min(turn, max(frames))] for turn in TURNS], axis=1)
    write_png(OUT, strip)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
