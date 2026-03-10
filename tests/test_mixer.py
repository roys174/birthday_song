"""Tests for mixer.py — vocal + instrumental mixing."""

import os

import numpy as np
import pytest
import soundfile as sf

from mixer import mix_tracks


SR_INST = 44100
DURATION = 3.0


def _write_wav(path, data, sr):
    sf.write(path, data, sr)
    return path


class TestMixTracks:
    def test_basic_mix_produces_output(self, tmp_dir, instrumental_wav, vocals_wav):
        """Mix should produce a WAV file at the given path."""
        # Write a 44100 Hz mono vocals file
        out_path = os.path.join(tmp_dir, "out.wav")
        voc_sr = 44100
        voc = (0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, 3, voc_sr * 3))).astype(np.float32)
        voc_path = os.path.join(tmp_dir, "voc44.wav")
        sf.write(voc_path, voc, voc_sr)

        result = mix_tracks(instrumental_wav, voc_path, out_path)
        assert os.path.exists(result)
        data, sr = sf.read(result)
        assert data.shape[0] > 0

    def test_output_is_stereo(self, tmp_dir, instrumental_wav):
        """Output should be stereo (matching instrumental channels)."""
        voc_sr = 44100
        voc = (0.3 * np.ones(voc_sr * 3)).astype(np.float32)
        voc_path = os.path.join(tmp_dir, "voc.wav")
        sf.write(voc_path, voc, voc_sr)

        out_path = os.path.join(tmp_dir, "out.wav")
        mix_tracks(instrumental_wav, voc_path, out_path)
        data, _ = sf.read(out_path)
        assert data.ndim == 2
        assert data.shape[1] == 2

    def test_no_clipping(self, tmp_dir, instrumental_wav):
        """Mixed output should not clip (values within [-1, 1])."""
        voc_sr = 44100
        # Very loud vocals
        voc = (0.9 * np.ones(voc_sr * 3)).astype(np.float32)
        voc_path = os.path.join(tmp_dir, "voc_loud.wav")
        sf.write(voc_path, voc, voc_sr)

        out_path = os.path.join(tmp_dir, "out.wav")
        mix_tracks(instrumental_wav, voc_path, out_path)
        data, _ = sf.read(out_path)
        assert np.max(np.abs(data)) <= 1.0

    def test_volume_multipliers(self, tmp_dir, instrumental_wav):
        """Vocals volume=0 should produce output equal to instrumental only."""
        voc_sr = 44100
        voc = (0.5 * np.ones(voc_sr * 3)).astype(np.float32)
        voc_path = os.path.join(tmp_dir, "voc.wav")
        sf.write(voc_path, voc, voc_sr)

        out_path = os.path.join(tmp_dir, "out.wav")
        mix_tracks(instrumental_wav, voc_path, out_path, vocals_volume=0.0, instrumental_volume=1.0)
        mix_data, _ = sf.read(out_path)

        inst_data, _ = sf.read(instrumental_wav, always_2d=True)
        # Should be identical to instrumental (after normalization)
        np.testing.assert_allclose(mix_data, inst_data, atol=1e-5)

    def test_instrumental_volume_zero(self, tmp_dir, instrumental_wav):
        """Instrumental volume=0 should leave only vocals in the mix."""
        voc_sr = 44100
        amplitude = 0.4
        voc = (amplitude * np.ones(voc_sr * 3)).astype(np.float32)
        voc_path = os.path.join(tmp_dir, "voc.wav")
        sf.write(voc_path, voc, voc_sr)

        out_path = os.path.join(tmp_dir, "out.wav")
        mix_tracks(instrumental_wav, voc_path, out_path, vocals_volume=1.0, instrumental_volume=0.0)
        data, _ = sf.read(out_path)
        # All values should equal amplitude (stereo, same in both channels)
        np.testing.assert_allclose(data[:, 0], amplitude, atol=1e-4)

    def test_different_length_tracks_padded(self, tmp_dir, instrumental_wav):
        """Shorter track should be zero-padded to match the longer one."""
        # Vocals shorter than instrumental (3 s instrumental, 1 s vocals)
        voc_sr = 44100
        voc = (0.3 * np.ones(voc_sr)).astype(np.float32)  # 1 second
        voc_path = os.path.join(tmp_dir, "voc_short.wav")
        sf.write(voc_path, voc, voc_sr)

        out_path = os.path.join(tmp_dir, "out.wav")
        mix_tracks(instrumental_wav, voc_path, out_path)
        data, _ = sf.read(out_path)
        inst, _ = sf.read(instrumental_wav)
        # Output length should match the longer track
        assert len(data) >= len(inst)

    def test_resamples_vocals_if_needed(self, tmp_dir, instrumental_wav):
        """Vocals at a different sample rate should be resampled correctly."""
        voc_sr = 22050  # different from instrumental's 44100
        voc = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 3, voc_sr * 3))).astype(np.float32)
        voc_path = os.path.join(tmp_dir, "voc22.wav")
        sf.write(voc_path, voc, voc_sr)

        out_path = os.path.join(tmp_dir, "out.wav")
        result = mix_tracks(instrumental_wav, voc_path, out_path)
        data, sr = sf.read(result)
        assert sr == 44100
        assert data.shape[0] > 0

    def test_creates_output_directory(self, tmp_dir, instrumental_wav):
        """mix_tracks should create nested output directories as needed."""
        voc_sr = 44100
        voc = np.zeros(voc_sr * 2, dtype=np.float32)
        voc_path = os.path.join(tmp_dir, "voc.wav")
        sf.write(voc_path, voc, voc_sr)

        nested = os.path.join(tmp_dir, "deep", "nested", "out.wav")
        mix_tracks(instrumental_wav, voc_path, nested)
        assert os.path.exists(nested)

    def test_returns_output_path(self, tmp_dir, instrumental_wav):
        voc_sr = 44100
        voc = np.zeros(voc_sr, dtype=np.float32)
        voc_path = os.path.join(tmp_dir, "voc.wav")
        sf.write(voc_path, voc, voc_sr)
        out_path = os.path.join(tmp_dir, "out.wav")
        result = mix_tracks(instrumental_wav, voc_path, out_path)
        assert result == out_path
