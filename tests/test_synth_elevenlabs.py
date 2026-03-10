"""Tests for synth_elevenlabs.py — ElevenLabs synthesis backend."""

import os

import numpy as np
import pytest
import soundfile as sf

from aligner import LyricSegment

SR = 44100


def _pcm_bytes(n_samples=SR, amplitude=0.3) -> bytes:
    """Generate raw PCM int16 bytes (simulates ElevenLabs pcm_44100 output)."""
    audio = (amplitude * np.ones(n_samples, dtype=np.float32))
    return (audio * 32768).astype(np.int16).tobytes()


# ──────────────────────────────────────────────────────────────────────────────
# _assemble_segments
# ──────────────────────────────────────────────────────────────────────────────

class TestAssembleSegments:
    def _seg(self, text, start, end):
        return LyricSegment(text, start, end)

    def _audio(self, n=SR):
        return np.ones(n, dtype=np.float32) * 0.3

    def test_empty_returns_silence(self):
        from synth_elevenlabs import _assemble_segments
        result = _assemble_segments([])
        assert isinstance(result, np.ndarray)
        assert len(result) == SR  # fallback

    def test_single_segment_placed_correctly(self):
        from synth_elevenlabs import _assemble_segments
        seg = self._seg("hello", 1.0, 2.0)
        audio = self._audio(SR)  # 1 second at 44100
        result = _assemble_segments([(audio, SR, seg)])
        # Before start: silence
        assert np.max(np.abs(result[: int(0.9 * SR)])) < 0.01

    def test_no_clipping(self):
        from synth_elevenlabs import _assemble_segments
        # Two overlapping loud segments
        seg1 = self._seg("a", 0.0, 2.0)
        seg2 = self._seg("b", 0.0, 2.0)
        audio = np.ones(SR * 2, dtype=np.float32) * 0.9
        result = _assemble_segments([(audio, SR, seg1), (audio, SR, seg2)])
        assert np.max(np.abs(result)) <= 1.0

    def test_skips_zero_length_audio(self):
        from synth_elevenlabs import _assemble_segments
        seg = self._seg("hi", 0.0, 1.0)
        result = _assemble_segments([(np.array([], dtype=np.float32), SR, seg)])
        assert isinstance(result, np.ndarray)

    def test_multiple_segments_ordered(self):
        from synth_elevenlabs import _assemble_segments
        segs = [self._seg("a", 0.0, 1.0), self._seg("b", 2.0, 3.0)]
        audios = [(self._audio(SR), SR, segs[0]), (self._audio(SR), SR, segs[1])]
        result = _assemble_segments(audios)
        assert len(result) >= int(3.0 * SR)


# ──────────────────────────────────────────────────────────────────────────────
# _synthesize_line
# ──────────────────────────────────────────────────────────────────────────────

class TestSynthesizeLine:
    def _make_client(self, mocker):
        client = mocker.MagicMock()
        client.text_to_speech.convert.return_value = iter([_pcm_bytes()])
        return client

    def test_returns_float32_array(self, mocker):
        from synth_elevenlabs import _synthesize_line
        client = self._make_client(mocker)
        audio, sr = _synthesize_line(client, "voice123", "hello", "en")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 44100

    def test_values_normalized(self, mocker):
        """PCM bytes should be decoded to [-1, 1] range."""
        from synth_elevenlabs import _synthesize_line
        client = self._make_client(mocker)
        audio, _ = _synthesize_line(client, "voice123", "hello", "en")
        assert np.max(np.abs(audio)) <= 1.0

    def test_calls_tts_convert(self, mocker):
        from synth_elevenlabs import _synthesize_line
        client = self._make_client(mocker)
        _synthesize_line(client, "voice123", "שלום", "he")
        client.text_to_speech.convert.assert_called_once()

    def test_uses_multilingual_model(self, mocker):
        from synth_elevenlabs import _synthesize_line, ELEVENLABS_MULTILINGUAL_MODEL
        client = self._make_client(mocker)
        _synthesize_line(client, "voice123", "hello", "en")
        call_kwargs = client.text_to_speech.convert.call_args
        assert call_kwargs.kwargs.get("model_id") == ELEVENLABS_MULTILINGUAL_MODEL


# ──────────────────────────────────────────────────────────────────────────────
# _clone_voice
# ──────────────────────────────────────────────────────────────────────────────

class TestCloneVoice:
    def test_returns_voice_id(self, mocker, mono_wav):
        from synth_elevenlabs import _clone_voice
        client = mocker.MagicMock()
        client.voices.ivc.create.return_value = mocker.MagicMock(voice_id="abc123")
        voice_id = _clone_voice(client, mono_wav)
        assert voice_id == "abc123"

    def test_uploads_file(self, mocker, mono_wav):
        from synth_elevenlabs import _clone_voice
        client = mocker.MagicMock()
        client.voices.ivc.create.return_value = mocker.MagicMock(voice_id="xyz")
        _clone_voice(client, mono_wav)
        client.voices.ivc.create.assert_called_once()
        _, kwargs = client.voices.ivc.create.call_args
        # Should include a file in the files argument
        files = kwargs.get("files", [])
        assert len(files) > 0


# ──────────────────────────────────────────────────────────────────────────────
# clone_and_synthesize (ElevenLabs — full flow, mocked client)
# ──────────────────────────────────────────────────────────────────────────────

class TestElevenLabsCloneAndSynthesize:
    def _setup_mocks(self, mocker):
        client = mocker.MagicMock()
        client.voices.ivc.create.return_value = mocker.MagicMock(voice_id="voice_xyz")
        client.text_to_speech.convert.return_value = iter([_pcm_bytes()])
        mocker.patch("synth_elevenlabs.ElevenLabs", return_value=client)
        return client

    def test_output_file_created(self, mocker, mono_wav, tmp_dir):
        from synth_elevenlabs import clone_and_synthesize
        # Patch the elevenlabs imports inside the module
        mocker.patch("synth_elevenlabs._clone_voice", return_value="voice_xyz")
        mocker.patch(
            "synth_elevenlabs._synthesize_line",
            return_value=(np.zeros(SR, dtype=np.float32), SR),
        )
        mocker.patch(
            "synth_elevenlabs._assemble_segments",
            return_value=np.zeros(SR * 3, dtype=np.float32),
        )

        # Patch the ElevenLabs import
        mock_client = mocker.MagicMock()
        mocker.patch("synth_elevenlabs.ElevenLabs", return_value=mock_client)

        segs = [LyricSegment("hello", 0.0, 2.0), LyricSegment("world", 2.5, 5.0)]
        out = os.path.join(tmp_dir, "el_out.wav")
        clone_and_synthesize(mono_wav, segs, out, language="en", api_key="fake_key")
        assert os.path.exists(out)

    def test_cleans_up_voice(self, mocker, mono_wav, tmp_dir):
        """The cloned voice should be deleted after synthesis."""
        from synth_elevenlabs import clone_and_synthesize

        mock_client = mocker.MagicMock()
        mocker.patch("synth_elevenlabs.ElevenLabs", return_value=mock_client)
        mocker.patch("synth_elevenlabs._clone_voice", return_value="temp_voice")
        mocker.patch(
            "synth_elevenlabs._synthesize_line",
            return_value=(np.zeros(SR, dtype=np.float32), SR),
        )
        mocker.patch(
            "synth_elevenlabs._assemble_segments",
            return_value=np.zeros(SR * 3, dtype=np.float32),
        )

        segs = [LyricSegment("hi", 0.0, 1.0)]
        out = os.path.join(tmp_dir, "out.wav")
        clone_and_synthesize(mono_wav, segs, out, language="en", api_key="key")
        mock_client.voices.delete.assert_called_once_with("temp_voice")

    def test_synthesizes_each_segment(self, mocker, mono_wav, tmp_dir):
        from synth_elevenlabs import clone_and_synthesize

        mock_client = mocker.MagicMock()
        mocker.patch("synth_elevenlabs.ElevenLabs", return_value=mock_client)
        mocker.patch("synth_elevenlabs._clone_voice", return_value="v1")

        synth_mock = mocker.patch(
            "synth_elevenlabs._synthesize_line",
            return_value=(np.zeros(SR, dtype=np.float32), SR),
        )
        mocker.patch(
            "synth_elevenlabs._assemble_segments",
            return_value=np.zeros(SR * 5, dtype=np.float32),
        )

        segs = [
            LyricSegment("hello", 0.0, 2.0),
            LyricSegment("world", 2.5, 5.0),
            LyricSegment("foo", 5.5, 7.0),
        ]
        out = os.path.join(tmp_dir, "out.wav")
        clone_and_synthesize(mono_wav, segs, out, language="en", api_key="key")
        assert synth_mock.call_count == 3

    def test_empty_segments_produce_silence(self, mocker, mono_wav, tmp_dir):
        """Empty-text segments should produce silence, not call synthesize_line."""
        from synth_elevenlabs import clone_and_synthesize

        mocker.patch("synth_elevenlabs.ElevenLabs", return_value=mocker.MagicMock())
        mocker.patch("synth_elevenlabs._clone_voice", return_value="v1")
        synth_mock = mocker.patch("synth_elevenlabs._synthesize_line")
        mocker.patch(
            "synth_elevenlabs._assemble_segments",
            return_value=np.zeros(SR, dtype=np.float32),
        )

        segs = [LyricSegment("", 0.0, 2.0), LyricSegment("", 2.0, 4.0)]
        out = os.path.join(tmp_dir, "out.wav")
        clone_and_synthesize(mono_wav, segs, out, language="en", api_key="key")
        synth_mock.assert_not_called()
