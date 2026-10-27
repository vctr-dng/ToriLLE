<p align="center">
  <img alt="Mating-ritual of two random agents" src="https://github.com/Miffyli/ToriLLE/raw/master/images/toribash.gif">
  <a href="http://www.youtube.com/watch?feature=player_embedded&v=_oxx28PbfdI
    " target="_blank"><img src="http://img.youtube.com/vi/_oxx28PbfdI/0.jpg"
    alt="PPO ripping a Uke a new one, and feeling the scene" width="250" height="240" border="0" /></a>
  <a href="https://www.youtube.com/watch?v=oWxVb4YcU1w
    " target="_blank"><img src="http://img.youtube.com/vi/oWxVb4YcU1w/0.jpg"
    alt="Another PPO attacking Uke in more random situations" width="250" height="240" border="0" /></a>
</p>

# ToriLLE
Toribash Learning Environment. Extra "L" to make words more memorable for Finns ("Torille" = "To the marketplace").

ToriLLE provides learning agents an interface to video-game [Toribash](http://www.toribash.com/), a humanoid fighting game.
Toribash provides environment for MuJoCo-like humanoid control, specifically aimed for competitive gameplay. This makes
ToriLLE suitable for e.g. self-play experiments.

ToriLLE comes with a Python interface and pre-made OpenAI Gym environment with various tasks. Following white-paper includes baseline experiments and benchmarks conducted using ToriLLE: [https://arxiv.org/abs/1807.10110](https://arxiv.org/abs/1807.10110)

## Requirements
Tested with Python 3.14 on Linux and Windows.

* A native Toribash installation
  * Linux: Steam installs `toribash_steam` under `$XDG_DATA_HOME/Steam/steamapps/common/Toribash/`. When `XDG_DATA_HOME` is unset, ToriLLE uses `~/.local/share`.
    ToriLLE must be able to write to this directory while a controller is running.
  * Windows: ToriLLE uses the packaged `toribash.exe` by default.
* An active display for Toribash's OpenGL renderer. See the [FAQ](docs/faq.md) for headless Linux setups.

## Quickstart
Install the project and its locked dependencies with uv:
```
uv sync
```

Random agent:
```python
from torille import ToribashControl
from torille.utils import create_random_actions

# Show gameplay
toribash = ToribashControl(draw_game=True)
toribash.init()

# Random agent
while 1:
    state, t = toribash.get_state()
    if t:
        break
    toribash.make_actions(create_random_actions())
toribash.close()
```

Gymnasium environment:
```python
import gymnasium as gym
import torille.envs

env = gym.make("Toribash-DestroyUke-v0")
# Show gameplay
env.set_draw_game(True)

initial_state = env.reset()
t = False
# Random agent
while not t:
    s, r, t, _ = env.step(env.action_space.sample())
env.close()
```

## Manual installation

:warning: Following instructions are old and may not work with current versions of Toribash. :warning:

You can install ToriLLE without PyPI/pip with the following:

* Install Toribash from Steam.
* Copy contents of `toribash-codes` to Toribash installation directory. Overwrite files
  * **Note:** This will prevent using that specific installation as a regular game. Remove/rename `profile.tbs` file
               to revert most of the changes and use game normally again.
  * **Note2:** Starting with Toribash 5.4 / Steam version of Toribash, settings file is stored in under user's directory at
               `Saved Games/Toribash/custom.cfg`. Toribash loads this file by default if it finds it, which may cause
               ToriLLE to run in wrong settings.
* On Linux, ToriLLE discovers Steam's native `toribash_steam` executable automatically. To use a different native installation, pass its path explicitly:
  ```python
  toribash = ToribashControl(executable="/path/to/toribash_steam")
  ```
  ToriLLE temporarily adds its native controller script to `profile.tbs` and restores the original profile when `close()` is called.

## Playing in multiplayer

:warning: Multiplayer compatibility depends on the current native Toribash build and may require additional configuration. :warning:

Want to try your agents against human players in multiplayer? Check how [manual remote control](docs/manual_torille.md) works.


## Documentation

Examples in `examples` provide quickstart to how to use ToriLLE, and also show how to apply settings or
record replays.

For references see:

* [Python library](docs/torille.md)
* [Gym environment](docs/envs.md)

For troubleshooting, see the [FAQ](docs/faq.md).

If you wish to modify ToriLLE or use other language to control Toribash instance, see [hacking](docs/hacking.md) and [documentation on protocol](docs/protocol).

## Repository structure
- `./torille/`: Python codes for the learning environment (inc. Gym environment)
  - `./torille/toribash`: This includes stripped version of the Toribash game when installed from PyPi
- `./toribash-codes/`: Files required for Toribash to make this learning environment work
- `./examples/`: Python examples on how to use this library
- `./docs/`: Detailed documentation of the inner workings
- `./experiments/`: Codes used to run experiments in the white paper
- `./images/`: Contains GIFs used here

## Related work and useful links

* Options/chat rules: http://forum.toribash.com/showthread.php?t=317900
* Lua functions: https://github.com/trittimo/ToriScriptAPI/blob/master/docs/toribash_docs.txt
* Bodypart list: http://forum.toribash.com/showthread.php?t=9391
* Previous experiments at machine learning with Toribash:
  * http://forum.toribash.com/showthread.php?t=170100
  * http://forum.toribash.com/showthread.php?t=167355
  * http://forum.toribash.com/showthread.php?t=25263
  * https://www.researchgate.net/profile/Jonathan_Byrne/publication/228848637_Optimising_offensive_moves_in_toribash_using_a_genetic_algorithm/links/0046351420d5001396000000.pdf

## Citing

We wouldn't mind a citation if you find ToriLLE useful in your work. It also helps us to see what people have been up to!

```
@inproceedings{kanervisto2019torille,
  title={ToriLLE: Learning Environment for Hand-to-Hand Combat},
  author={Kanervisto, Anssi and Hautam{\"a}ki, Ville},
  booktitle={2019 IEEE Conference on Games (CoG)},
  year={2019},
}
```

## Special thanks & Acknowledgements
- hampa and Dranix for invaluable help with configuring Toribash and lua scripts (also for developing the game in the first place!)
- Siim Põder (user "windo" on GitHub) for original [toribash-evolver](https://github.com/windo/toribash-evolver) code
- box (Toribash user) for comments during inception of this project

## License
Code original to ToriLLE is licensed under GNU GPL 3.0. Toribash is property of Nabistudios. Toribash binary in this package is shared with the permission of main developer "hampa".
