"""Tests for synthesizer.py — backend dispatcher."""

import numpy as np
import pytest
from aligner import LyricSegment
from synthesizer import clone_and_synthesize, BACKENDS


class TestBackendDispatcher:
    def _segs(self):
        return [LyricSegment("hello", 0.0, 2.0)]

    def test_backends_tuple(self):
        assert "elevenlabs" in BACKENDS
        assert "local" in BACKENDS

    def test_unknown_backend_raises(self, tmp_dir):
        with pytest.raises(ValueError, match="Unknown backend"):
            clone_and_synthesize(
                vocals_path="fake.wav",
                timed_new_lyrics=self._segs(),
                output_path=f"{tmp_dir}/out.wav",
                language="en",
                backend="nonexistent",
            )

    def test_elevenlabs_without_key_raises(self, tmp_dir):
        with pytest.raises(ValueError, match="API key"):
            clone_and_synthesize(
                vocals_path="fake.wav",
                timed_new_lyrics=self._segs(),
                output_path=f"{tmp_dir}/out.wav",
                language="en",
                backend="elevenlabs",
                api_key=None,
            )

    def test_elevenlabs_empty_key_raises(self, tmp_dir):
        with pytest.raises(ValueError, match="API key"):
            clone_and_synthesize(
                vocals_path="fake.wav",
                timed_new_lyrics=self._segs(),
                output_path=f"{tmp_dir}/out.wav",
                language="en",
                backend="elevenlabs",
                api_key="",
            )

    def test_routes_to_elevenlabs(self, mocker, tmp_dir):
        mock = mocker.patch(
            "synth_elevenlabs.clone_and_synthesize",
            return_value=f"{tmp_dir}/out.wav",
        )
        clone_and_synthesize(
            vocals_path="fake.wav",
            timed_new_lyrics=self._segs(),
            output_path=f"{tmp_dir}/out.wav",
            language="en",
            backend="elevenlabs",
            api_key="sk-test",
        )
        mock.assert_called_once()

    def test_routes_to_local(self, mocker, tmp_dir):
        mock = mocker.patch(
            "synth_local.clone_and_synthesize",
            return_value=f"{tmp_dir}/out.wav",
        )
        clone_and_synthesize(
            vocals_path="fake.wav",
            timed_new_lyrics=self._segs(),
            output_path=f"{tmp_dir}/out.wav",
            language="he",
            backend="local",
        )
        mock.assert_called_once()

    def test_local_ignores_api_key(self, mocker, tmp_dir):
        """Local backend should work even if api_key is None."""
        mock = mocker.patch(
            "synth_local.clone_and_synthesize",
            return_value=f"{tmp_dir}/out.wav",
        )
        clone_and_synthesize(
            vocals_path="fake.wav",
            timed_new_lyrics=self._segs(),
            output_path=f"{tmp_dir}/out.wav",
            language="en",
            backend="local",
            api_key=None,
        )
        mock.assert_called_once()

    def test_passes_language_to_backend(self, mocker, tmp_dir):
        mock = mocker.patch(
            "synth_local.clone_and_synthesize",
            return_value=f"{tmp_dir}/out.wav",
        )
        clone_and_synthesize(
            vocals_path="fake.wav",
            timed_new_lyrics=self._segs(),
            output_path=f"{tmp_dir}/out.wav",
            language="he",
            backend="local",
        )
        _, kwargs = mock.call_args
        assert kwargs.get("language") == "he"
