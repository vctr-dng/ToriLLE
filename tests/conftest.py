import pytest
from filelock import FileLock

# One Toribash instance at a time on this machine: native tests hold the same
# lock file as `flock /tmp/toribash-game.lock <command>`, so they wait for any
# other job that plays a game (and are not to be wrapped in flock themselves).
GAME_LOCK = "/tmp/toribash-game.lock"


@pytest.fixture(scope="session", autouse=True)
def _toribash_game_lock(request):
    if not any(item.get_closest_marker("native") for item in request.session.items):
        yield
        return
    with FileLock(GAME_LOCK, timeout=3600):
        yield
