"""Shared fixtures for all tests."""

import os
import sys
import tempfile

import numpy as np
import pytest
import soundfile as sf

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 22050
DURATION = 3.0  # seconds


def _sine(freq=440.0, sr=SR, duration=DURATION, amplitude=0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _stereo(freq=440.0, sr=SR, duration=DURATION) -> np.ndarray:
    mono = _sine(freq, sr, duration)
    return np.stack([mono, mono], axis=1)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mono_wav(tmp_dir) -> str:
    """A short mono WAV file (440 Hz sine, 3 s)."""
    path = os.path.join(tmp_dir, "mono.wav")
    sf.write(path, _sine(), SR)
    return path


@pytest.fixture
def stereo_wav(tmp_dir) -> str:
    """A short stereo WAV file (440 Hz sine, 3 s)."""
    path = os.path.join(tmp_dir, "stereo.wav")
    sf.write(path, _stereo(), SR)
    return path


@pytest.fixture
def vocals_wav(tmp_dir) -> str:
    """Simulated isolated vocals (220 Hz sine = typical male voice F0)."""
    path = os.path.join(tmp_dir, "vocals.wav")
    sf.write(path, _sine(freq=220.0, duration=10.0), SR)
    return path


@pytest.fixture
def instrumental_wav(tmp_dir) -> str:
    """Simulated instrumental (stereo, 44100 Hz)."""
    path = os.path.join(tmp_dir, "no_vocals.wav")
    sr = 44100
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    sf.write(path, np.stack([mono, mono], axis=1), sr)
    return path
