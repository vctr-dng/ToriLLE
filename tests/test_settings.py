import numpy as np
import pytest

from torille.torille import ToribashSettings


def test_default_settings_are_valid():
    ToribashSettings().validate_settings()


def test_numpy_numbers_are_accepted():
    settings = ToribashSettings(matchframes=np.int64(500), gravity_z=np.float32(-9.81))

    settings.validate_settings()


def test_wrong_type_error_names_the_offending_setting():
    settings = ToribashSettings(matchframes="500")

    with pytest.raises(ValueError, match="Setting matchframes"):
        settings.validate_settings()


def test_last_numeric_setting_is_validated():
    settings = ToribashSettings(dojo_size="big")

    with pytest.raises(ValueError, match="Setting dojo_size"):
        settings.validate_settings()


def test_turnframes_cannot_exceed_matchframes():
    settings = ToribashSettings(matchframes=10, turnframes=11)

    with pytest.raises(ValueError, match="turnframes"):
        settings.validate_settings()
