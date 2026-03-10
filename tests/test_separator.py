"""Tests for separator.py — Demucs source separation."""

import os

import pytest

from separator import _find_demucs_output, separate_audio


# ──────────────────────────────────────────────────────────────────────────────
# _find_demucs_output
# ──────────────────────────────────────────────────────────────────────────────

class TestFindDemucsOutput:
    def test_finds_correct_dir(self, tmp_dir):
        model_dir = os.path.join(tmp_dir, "htdemucs")
        track_dir = os.path.join(model_dir, "my_song")
        os.makedirs(track_dir)
        result = _find_demucs_output(tmp_dir, "my_song")
        assert result == track_dir

    def test_finds_nested_model_dir(self, tmp_dir):
        model_dir = os.path.join(tmp_dir, "htdemucs_ft")
        track_dir = os.path.join(model_dir, "track_name")
        os.makedirs(track_dir)
        result = _find_demucs_output(tmp_dir, "track_name")
        assert result == track_dir

    def test_missing_track_raises(self, tmp_dir):
        model_dir = os.path.join(tmp_dir, "htdemucs")
        os.makedirs(model_dir)
        with pytest.raises(RuntimeError, match="Could not find demucs output"):
            _find_demucs_output(tmp_dir, "nonexistent_song")

    def test_empty_output_dir_raises(self, tmp_dir):
        with pytest.raises(RuntimeError):
            _find_demucs_output(tmp_dir, "any_song")


# ──────────────────────────────────────────────────────────────────────────────
# separate_audio (mocked demucs subprocess)
# ──────────────────────────────────────────────────────────────────────────────

class TestSeparateAudio:
    def _setup_demucs_output(self, out_dir, audio_name):
        """Simulate the file structure Demucs produces."""
        import numpy as np
        import soundfile as sf

        track_dir = os.path.join(out_dir, "htdemucs", audio_name)
        os.makedirs(track_dir)

        sr = 44100
        silence = np.zeros((sr * 3, 2), dtype=np.float32)
        sf.write(os.path.join(track_dir, "vocals.wav"), silence, sr)
        sf.write(os.path.join(track_dir, "no_vocals.wav"), silence, sr)
        return track_dir

    def test_returns_vocal_and_instrumental_paths(self, mocker, mono_wav, tmp_dir):
        out_dir = os.path.join(tmp_dir, "sep")
        audio_name = "mono"  # stem of mono_wav

        self._setup_demucs_output(out_dir, audio_name)

        run_mock = mocker.MagicMock()
        run_mock.returncode = 0
        mocker.patch("subprocess.run", return_value=run_mock)

        vocals, instrumental = separate_audio(mono_wav, out_dir)
        assert vocals.endswith("vocals.wav")
        assert instrumental.endswith("no_vocals.wav")
        assert os.path.exists(vocals)
        assert os.path.exists(instrumental)

    def test_calls_demucs_with_two_stems(self, mocker, mono_wav, tmp_dir):
        out_dir = os.path.join(tmp_dir, "sep")
        self._setup_demucs_output(out_dir, "mono")

        run_mock = mocker.MagicMock()
        run_mock.returncode = 0
        mocker.patch("subprocess.run", return_value=run_mock)

        separate_audio(mono_wav, out_dir)

        import subprocess
        args = subprocess.run.call_args[0][0]
        assert "--two-stems" in args
        assert "vocals" in args

    def test_demucs_failure_raises(self, mocker, mono_wav, tmp_dir):
        run_mock = mocker.MagicMock()
        run_mock.returncode = 1
        run_mock.stderr = "CUDA out of memory"
        mocker.patch("subprocess.run", return_value=run_mock)

        with pytest.raises(RuntimeError, match="Demucs failed"):
            separate_audio(mono_wav, tmp_dir)

    def test_creates_output_directory(self, mocker, mono_wav, tmp_dir):
        new_out = os.path.join(tmp_dir, "deep", "sep")
        self._setup_demucs_output(new_out, "mono")

        run_mock = mocker.MagicMock()
        run_mock.returncode = 0
        mocker.patch("subprocess.run", return_value=run_mock)

        separate_audio(mono_wav, new_out)
        assert os.path.isdir(new_out)

    def test_missing_vocals_file_raises(self, mocker, mono_wav, tmp_dir):
        out_dir = os.path.join(tmp_dir, "sep")
        # Create partial demucs output (missing vocals.wav)
        track_dir = os.path.join(out_dir, "htdemucs", "mono")
        os.makedirs(track_dir)
        open(os.path.join(track_dir, "no_vocals.wav"), "w").close()

        run_mock = mocker.MagicMock()
        run_mock.returncode = 0
        mocker.patch("subprocess.run", return_value=run_mock)

        with pytest.raises(RuntimeError, match="Vocals not found"):
            separate_audio(mono_wav, out_dir)
