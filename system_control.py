"""Windows volume and brightness control."""

import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities

_volume_interface = None


def _get_volume_interface():
    global _volume_interface
    if _volume_interface is None:
        _volume_interface = AudioUtilities.GetSpeakers().EndpointVolume
    return _volume_interface


def set_volume(level):
    """level: 0.0-1.0"""
    try:
        _get_volume_interface().SetMasterVolumeLevelScalar(max(0.0, min(1.0, level)), None)
    except OSError:
        pass


def get_volume():
    try:
        return _get_volume_interface().GetMasterVolumeLevelScalar()
    except OSError:
        return None


def set_brightness(level):
    """level: 0.0-1.0"""
    try:
        sbc.set_brightness(int(max(0.0, min(1.0, level)) * 100))
    except (sbc.exceptions.ScreenBrightnessError, OSError):
        pass


def get_brightness():
    try:
        return sbc.get_brightness()[0] / 100.0
    except (sbc.exceptions.ScreenBrightnessError, OSError, IndexError):
        return None
