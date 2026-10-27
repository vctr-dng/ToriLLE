from importlib.metadata import version

from . import envs as envs
from .manual_torille import *
from .torille import *

__version__ = version("torille")
