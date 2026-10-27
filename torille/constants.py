#  constants.py
#  File holding all constants for Torille
#
#  Author: Anssi "Miffyli" Kanervisto, 2018
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
import os
import sys
from pathlib import Path

# Port where Toribashes connect to
PORT = 7788
# Global timeout (in seconds) for connections
TIMEOUT = 300
# Buffer size (for recv)
BUFFER_SIZE = 8096
# Line ending character
MESSAGE_END = b"\n"

# "Limb" or "Bodypart"
NUM_LIMBS = 21
NUM_JOINTS = 20
# Add hand_grips
NUM_CONTROLLABLES = NUM_JOINTS + 2
NUM_JOINT_STATES = 4
# Number of setting variables
NUM_SETTINGS = 19

# Bodypart x,y,z + Bodypart x,y,z velocities +
# groin rotation + Joint states + hand grips + injuries + selected player
# + frame info (x3)
STATE_LENGTH = (NUM_LIMBS * 3 * 2) * 2 + 16 * 2 + NUM_JOINTS * 2 + 4 + 2 + 1 + 3

# Path to Toribash supplied with the wheel package on Windows.
PACKAGE_TORIBASH_EXE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "toribash", "toribash.exe"
)


def get_platform_executable(platform=None):
    """Return the default native Toribash executable for the current platform."""
    platform = platform or sys.platform
    if platform == "linux":
        data_home = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )
        return str(
            data_home / "Steam" / "steamapps" / "common" / "Toribash" / "toribash_steam"
        )
    if platform == "win32":
        return PACKAGE_TORIBASH_EXE
    raise RuntimeError(
        "Toribash is only supported on Linux and Windows; "
        f"detected platform: {platform}"
    )
