#  utils.py
#  Misc. utility functions for Torille
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
import random as r
import warnings
from stat import S_IREAD, S_IRGRP, S_IROTH

from . import constants


def create_random_actions():
    """Return random actions for ToribashControl"""
    ret = [[], []]
    for plridx in range(2):
        for jointidx in range(constants.NUM_CONTROLLABLES):
            ret[plridx].append(r.randint(1, 4))
    return ret


def set_file_readonly(filepath):
    """
    Attempt to set given file read-only, and
    return True on success.

    Parameters:
        filepath: Path to file to be set read-only
    Returns:
        success: True if file was set read only, otherwise False
    """
    if os.path.isfile(filepath):
        # Set to read only (for user, group and all)
        try:
            os.chmod(filepath, S_IREAD | S_IRGRP | S_IROTH)
        except PermissionError:
            return False
        return True
    else:
        return False


def check_linux_sanity(display=None):
    """
    A helper function that checks Linux environment
    for requirements, and warns/throws accordingly

    Parameters:
        display: X display to check instead of the `DISPLAY` variable.
    """

    # Check that we have a valid display to render into.
    # We rather avoid running over SSH
    display = display or os.getenv("DISPLAY")
    if display is None:
        raise RuntimeError(
            "No display detected. "
            + "Toribash won't launch without active display. "
            + "If you have a monitor attached, set environment variable "
            + "DISPLAY to point at it (e.g. `export DISPLAY=:0`)"
        )
    if display[0] != ":":
        warnings.warn(
            "Looks like you have X-forwarding enabled. "
            + "This makes Toribash very slow and sad. "
            + "Consider using virtual screen buffer like Xvfb. "
            + "More info at the Github page https://github.com/Miffyli/ToriLLE"
        )
