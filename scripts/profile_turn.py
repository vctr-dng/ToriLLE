"""Per-turn timeline of the native bridge, from inotify events on .torille/.

    flock /tmp/toribash-game.lock uv run python scripts/profile_turn.py \
        --draw 0 --turnframes 1 --turns 300

Python records perf_counter stamps around its own steps; a thread reads inotify
events (via ctypes) on the bridge directory to stamp when Lua opens, truncates
and writes the files. Printed values are medians in milliseconds.
"""

import argparse
import ctypes
import os
import statistics
import struct
import threading
import time

import gymnasium as gym

import torille.envs  # noqa: F401

IN_MODIFY, IN_OPEN, IN_CLOSE_WRITE, IN_MOVED_TO, IN_CREATE, IN_DELETE = (
    0x2,
    0x20,
    0x8,
    0x80,
    0x100,
    0x200,
)
MASK = IN_MODIFY | IN_OPEN | IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE | IN_DELETE
NAMES = {
    IN_MODIFY: "modify",
    IN_OPEN: "open",
    IN_CLOSE_WRITE: "close_write",
    IN_MOVED_TO: "moved_to",
    IN_CREATE: "create",
    IN_DELETE: "delete",
}


def watch(directory, events, stop):
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK)
    libc.inotify_add_watch(fd, str(directory).encode(), MASK)
    import select

    while not stop.is_set():
        if not select.select([fd], [], [], 0.05)[0]:
            continue
        data = os.read(fd, 65536)
        stamp = time.perf_counter()
        pos = 0
        while pos < len(data):
            _, mask, _, length = struct.unpack_from("iIII", data, pos)
            name = data[pos + 16 : pos + 16 + length].rstrip(b"\0").decode()
            pos += 16 + length
            events.append((stamp, name, NAMES.get(mask & ~0x40000000, hex(mask))))
    os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--turnframes", type=int, default=1)
    parser.add_argument("--turns", type=int, default=300)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument(
        "--dump", type=int, default=0, help="print raw events of N turns"
    )
    args = parser.parse_args()

    env = gym.make(
        "Toribash-DestroyUke-v2", matchframes=100000, turnframes=args.turnframes
    )
    env.unwrapped.set_draw_game(bool(args.draw))
    game = env.unwrapped.game
    if args.legacy:
        game.use_inotify, game.poll_interval, game.spin_ms = False, 0.01, 0.0
    events, stop = [], threading.Event()
    marks = []
    try:
        env.reset()
        thread = threading.Thread(
            target=watch, args=(game.native_bridge.directory, events, stop)
        )
        thread.start()
        time.sleep(0.2)
        for _ in range(args.turns):
            act = env.action_space.sample()
            t0 = time.perf_counter()
            env.step(act)
            marks.append((t0, time.perf_counter(), len(events)))
        time.sleep(0.1)
    finally:
        stop.set()
        env.close()

    # Slice events per turn using step start/end stamps.
    rows = []
    for index, (t0, t1, _) in enumerate(marks[5:], 5):
        nxt = marks[index + 1][0] if index + 1 < len(marks) else t1 + 1
        evs = [(s - t0, n, e) for s, n, e in events if t0 - 0.0005 <= s < nxt]
        rows.append((t0, t1, evs))
        if index - 5 < args.dump:
            print(f"turn {index}: step total {1000 * (t1 - t0):.2f} ms")
            for s, n, e in evs:
                print(f"   {1000 * s:8.2f} {n:12s} {e}")

    def first(evs, name, kind, after=-1.0):
        for s, n, e in evs:
            if n == name and e == kind and s > after:
                return s
        return None

    cols = {
        "step_total": [],
        "py_write->lua_open_action": [],
        "lua_open->state_write_start": [],
        "state_write_start->state_close": [],
        "state_close->py_return": [],
    }
    for t0, t1, evs in rows:
        wr = first(evs, "action", "moved_to")
        op = first(evs, "action", "open", wr if wr is not None else -1)
        sm = first(evs, "state", "modify", op if op is not None else -1)
        sc = first(evs, "state", "close_write", sm if sm is not None else -1)
        cols["step_total"].append(1000 * (t1 - t0))
        if None in (wr, op, sm, sc):
            continue
        end = 1000 * (t1 - t0)
        cols["py_write->lua_open_action"].append(1000 * (op - wr))
        cols["lua_open->state_write_start"].append(1000 * (sm - op))
        cols["state_write_start->state_close"].append(1000 * (sc - sm))
        cols["state_close->py_return"].append(end - 1000 * sc)
    print(
        f"draw={args.draw} turnframes={args.turnframes} poll={game.poll_interval if hasattr(game, 'poll_interval') else 'n/a'} n={len(rows)}"
    )
    for key, values in cols.items():
        if values:
            print(
                f"  {key:34s} median {statistics.median(values):7.2f}  mean {statistics.mean(values):7.2f}  (n={len(values)})"
            )


if __name__ == "__main__":
    main()
