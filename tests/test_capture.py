"""RGB capture: geometry, resizing, spaces and option checks. No game is started."""

import gymnasium as gym
import numpy as np
import pytest

import torille.envs  # noqa: F401
from torille import capture, constants
from torille.envs.observations import dict_observation_space
from torille.envs.two_player import TwoPlayerToriEnv, observation_space
from torille.torille import ToribashState


def make_state(match_frame=15):
    values = np.zeros(constants.STATE_LENGTH)
    for start in (0, 165):
        values[start + 126 : start + 142] = np.eye(4).ravel()
        values[start + 142 : start + 162] = 3
    values[330:334] = [0, 1000, match_frame, 5]
    return ToribashState(list(values))


def test_square_crop_box_is_centred_and_square():
    assert capture.square_crop_box(2560, 1440) == (560, 0, 1440)
    assert capture.square_crop_box(1000, 1000) == (0, 0, 1000)
    assert capture.square_crop_box(600, 1001) == (0, 200, 600)


def test_letterbox_keeps_the_aspect_ratio():
    assert capture.letterbox_geometry(2560, 1440, 256) == (256, 144, 0, 56)
    assert capture.letterbox_geometry(1000, 2000, 200) == (100, 200, 50, 0)
    assert capture.letterbox_geometry(300, 300, 256) == (256, 256, 0, 0)


def test_resize_keeps_constant_images_and_integer_means():
    flat = np.full((90, 160, 3), 77, dtype=np.uint8)
    assert (capture.resize_area(flat, 30, 40) == 77).all()
    # Exact 2x2 box average
    image = np.array([[[0], [10]], [[20], [30]]], dtype=np.uint8)
    assert capture.resize_area(image, 1, 1)[0, 0, 0] == 15
    # Non-uniform content: bins of a 6x1 ramp
    ramp = np.arange(6, dtype=np.uint8).reshape(1, 6, 1) * 10
    assert capture.resize_area(ramp, 1, 3)[0, :, 0].tolist() == [5, 25, 45]
    # Growing repeats pixels
    assert capture.resize_area(ramp, 2, 12).shape == (2, 12, 1)
    # Identity
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (12, 9, 3), dtype=np.uint8)
    np.testing.assert_array_equal(capture.resize_area(noise, 12, 9), noise)


def test_process_frame_center_crop():
    image = np.zeros((90, 160, 3), dtype=np.uint8)
    # Left and right thirds-ish are outside the centred 90x90 square.
    image[:, :35] = 255
    image[:, 125:] = 255
    frame = capture.process_frame(image, 30, "center")
    assert frame.shape == (30, 30, 3)
    assert frame.dtype == np.uint8
    assert (frame == 0).all()


def test_process_frame_letterbox_has_black_bars_and_the_whole_image():
    image = np.full((90, 160, 3), 200, dtype=np.uint8)
    frame = capture.process_frame(image, 32, "letterbox")
    assert frame.shape == (32, 32, 3)
    out_w, out_h, x, y = capture.letterbox_geometry(160, 90, 32)
    assert (frame[:y] == 0).all()
    assert (frame[y + out_h :] == 0).all()
    assert (frame[y : y + out_h, x : x + out_w] == 200).all()


def test_process_frame_rejects_unknown_crop():
    with pytest.raises(ValueError, match="capture_crop"):
        capture.process_frame(np.zeros((4, 4, 3), np.uint8), 2, "stretch")


def test_flat_frame_detection():
    assert capture.is_flat_frame(np.zeros((8, 8, 3), np.uint8))
    assert capture.is_flat_frame(np.full((8, 8, 3), 255, np.uint8))
    noisy = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    assert not capture.is_flat_frame(noisy)


def test_validate_capture_options():
    capture.validate_capture_options(True, 224, "center")
    with pytest.raises(ValueError, match="draw_game=True"):
        capture.validate_capture_options(False, 224, "center")
    with pytest.raises(ValueError, match="capture_size"):
        capture.validate_capture_options(True, 4, "center")
    with pytest.raises(ValueError, match="capture_crop"):
        capture.validate_capture_options(True, 224, "zoom")


def test_dict_space_gets_an_rgb_key_only_with_capture():
    assert "rgb" not in dict_observation_space(100).spaces
    space = dict_observation_space(100, 224)
    assert space["rgb"].shape == (224, 224, 3)
    assert space["rgb"].dtype == np.uint8
    assert (space["rgb"].low == 0).all()
    assert (space["rgb"].high == 255).all()
    assert "rgb" in observation_space(100, 64).spaces


def test_default_envs_have_no_capture():
    env = gym.make("Toribash-DestroyUke-v2", observation_type="dict")
    assert "rgb" not in env.observation_space.spaces
    assert env.unwrapped.game.capture_size is None
    assert env.unwrapped.render() is None


def test_capture_options_on_gym_env():
    env = gym.make(
        "Toribash-DestroyUke-v2",
        observation_type="dict",
        capture_size=96,
        draw_game=True,
    )
    assert env.observation_space["rgb"].shape == (96, 96, 3)
    assert env.unwrapped.game.draw_game
    # render_mode alone: render() only, the observation is unchanged
    render_only = gym.make("Toribash-DestroyUke-v2", render_mode="rgb_array")
    assert render_only.unwrapped.capture_size == 256
    assert not render_only.unwrapped.capture_rgb_observation
    with pytest.raises(ValueError, match="observation_type='dict'"):
        gym.make("Toribash-DestroyUke-v2", capture_size=64)
    with pytest.raises(ValueError, match="render_mode"):
        gym.make("Toribash-DestroyUke-v2", render_mode="bogus")


def test_capture_without_draw_game_is_an_error_at_init():
    env = gym.make("Toribash-DestroyUke-v2", render_mode="rgb_array")
    with pytest.raises(ValueError, match="draw_game=True"):
        env.reset()
    two = TwoPlayerToriEnv(capture_size=64)
    with pytest.raises(ValueError, match="draw_game=True"):
        two.reset()


def test_observation_and_render_use_one_grab_per_state():
    env = gym.make(
        "Toribash-DestroyUke-v2",
        observation_type="dict",
        capture_size=8,
        render_mode="rgb_array",
        draw_game=True,
    ).unwrapped
    grabs = []

    def fake_grab():
        grabs.append(1)
        return np.full((8, 8, 3), len(grabs), dtype=np.uint8)

    env.game.grab_frame = fake_grab
    first = make_state(match_frame=6)
    observation = env._observe(first)
    assert observation["rgb"][0, 0, 0] == 1
    assert env.observation_space.contains(observation)
    env.old_state = first
    frame = env.render()
    assert len(grabs) == 1
    assert frame[0, 0, 0] == 1
    # The returned arrays are copies
    frame[:] = 0
    assert env.render()[0, 0, 0] == 1
    # A new state is a new grab
    second = make_state(match_frame=12)
    assert env._observe(second)["rgb"][0, 0, 0] == 2
    assert len(grabs) == 2


def test_two_player_observations_carry_the_same_frame():
    env = TwoPlayerToriEnv(draw_game=True, capture_size=8, render_mode="rgb_array")
    env.game.grab_frame = lambda: np.full((8, 8, 3), 5, dtype=np.uint8)
    observations = env._observations(make_state())
    for agent in env.possible_agents:
        assert env.observation_space(agent).contains(observations[agent])
    np.testing.assert_array_equal(
        observations["player_0"]["rgb"], observations["player_1"]["rgb"]
    )
    assert observations["player_0"]["rgb"] is not observations["player_1"]["rgb"]
    assert env.render().shape == (8, 8, 3)


def test_changed_game_options_are_restored_in_the_config_files(tmp_path):
    from torille.torille import _restore_options, _snapshot_options

    custom = tmp_path / "custom.cfg"
    temp = tmp_path / "temp.cfg"
    custom.write_text('trails "1"\njointflash "1"\nwidth "1280"\n')
    snapshot = _snapshot_options(
        [custom, temp, tmp_path / "missing.cfg"], ["jointflash"]
    )
    assert snapshot == {custom: {"jointflash": "1"}}

    # The game ran with the option off and a user change elsewhere in the file
    custom.write_text('trails "0"\njointflash "0"\nwidth "1280"\n')
    _restore_options(snapshot)
    assert custom.read_text() == 'trails "0"\njointflash "1"\nwidth "1280"\n'
