"""Tests for synth_local.py — free local synthesis backend."""

import asyncio
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from aligner import LyricSegment


SR = 22050
OUTPUT_SR = 44100


# ──────────────────────────────────────────────────────────────────────────────
# _median_f0
# ──────────────────────────────────────────────────────────────────────────────

class TestMedianF0:
    def test_sine_wave_f0(self, tmp_dir):
        """A pure sine wave should have F0 close to its frequency."""
        from synth_local import _median_f0
        freq = 220.0
        t = np.linspace(0, 5, int(SR * 5), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        path = os.path.join(tmp_dir, "sine.wav")
        sf.write(path, audio, SR)
        f0 = _median_f0(path)
        assert pytest.approx(freq, rel=0.1) == f0  # within 10%

    def test_silence_returns_default(self, tmp_dir):
        """Silence (no voiced F0) should return the default fallback value."""
        from synth_local import _median_f0
        audio = np.zeros(SR * 3, dtype=np.float32)
        path = os.path.join(tmp_dir, "silence.wav")
        sf.write(path, audio, SR)
        f0 = _median_f0(path)
        assert f0 == pytest.approx(200.0)

    def test_male_range(self, tmp_dir):
        """220 Hz → classified as male (< 180 Hz threshold)."""
        from synth_local import _median_f0
        t = np.linspace(0, 5, int(SR * 5), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 120.0 * t)).astype(np.float32)
        path = os.path.join(tmp_dir, "male.wav")
        sf.write(path, audio, SR)
        f0 = _median_f0(path)
        assert f0 < 180

    def test_female_range(self, tmp_dir):
        """300 Hz → classified as female (>= 180 Hz threshold)."""
        from synth_local import _median_f0
        t = np.linspace(0, 5, int(SR * 5), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
        path = os.path.join(tmp_dir, "female.wav")
        sf.write(path, audio, SR)
        f0 = _median_f0(path)
        assert f0 >= 180


# ──────────────────────────────────────────────────────────────────────────────
# _match_pitch
# ──────────────────────────────────────────────────────────────────────────────

class TestMatchPitch:
    def _sine(self, freq, duration=3.0):
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def test_returns_same_length_list(self):
        from synth_local import _match_pitch
        audios = [self._sine(440), self._sine(330), self._sine(220)]
        result = _match_pitch(audios, target_f0=300.0)
        assert len(result) == len(audios)

    def test_output_is_float32(self):
        from synth_local import _match_pitch
        audios = [self._sine(440)]
        result = _match_pitch(audios, target_f0=300.0)
        assert result[0].dtype == np.float32

    def test_short_audio_passthrough(self):
        """Audio shorter than 512 samples should be returned unchanged."""
        from synth_local import _match_pitch
        short = np.zeros(100, dtype=np.float32)
        result = _match_pitch([short], target_f0=300.0)
        np.testing.assert_array_equal(result[0], short)

    def test_silence_passthrough(self):
        """Silence (no F0) should be returned unchanged."""
        from synth_local import _match_pitch
        silence = np.zeros(SR * 2, dtype=np.float32)
        result = _match_pitch([silence], target_f0=300.0)
        assert len(result) == 1  # doesn't crash

    def test_mixed_inputs(self):
        """Mix of valid and short audio."""
        from synth_local import _match_pitch
        audios = [
            self._sine(440),   # normal
            np.zeros(50, dtype=np.float32),  # too short
            self._sine(220),   # normal
        ]
        result = _match_pitch(audios, target_f0=300.0)
        assert len(result) == 3
        np.testing.assert_array_equal(result[1], audios[1])  # short unchanged


# ──────────────────────────────────────────────────────────────────────────────
# _assemble
# ──────────────────────────────────────────────────────────────────────────────

class TestAssemble:
    def _seg(self, text, start, end):
        return LyricSegment(text, start, end)

    def _sine22(self, duration=1.0):
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_output_at_correct_position(self):
        """Audio placed at segment.start should appear at the right sample."""
        from synth_local import _assemble
        seg = self._seg("hi", 1.0, 2.0)
        audio = self._sine22(1.0)
        result = _assemble([audio], [seg])

        start_sample = int(1.0 * OUTPUT_SR)
        # The region before start should be (near) zero
        assert np.max(np.abs(result[:start_sample])) < 0.01

    def test_output_length_covers_last_segment(self):
        from synth_local import _assemble
        segs = [self._seg("a", 0.0, 2.0), self._seg("b", 3.0, 5.0)]
        audios = [self._sine22(2.0), self._sine22(2.0)]
        result = _assemble(audios, segs)
        expected_min_length = int(5.0 * OUTPUT_SR)
        assert len(result) >= expected_min_length

    def test_normalization_prevents_clipping(self):
        from synth_local import _assemble
        # Overlapping loud segments
        segs = [self._seg("a", 0.0, 2.0), self._seg("b", 0.0, 2.0)]
        loud = (0.8 * np.ones(int(SR * 2), dtype=np.float32))
        result = _assemble([loud, loud], segs)
        assert np.max(np.abs(result)) <= 1.0

    def test_empty_audio_skipped(self):
        from synth_local import _assemble
        segs = [self._seg("a", 0.0, 2.0)]
        result = _assemble([np.array([], dtype=np.float32)], segs)
        assert len(result) > 0  # doesn't crash, returns some output

    def test_empty_input_returns_array(self):
        from synth_local import _assemble
        result = _assemble([], [])
        assert isinstance(result, np.ndarray)

    def test_multiple_segments_no_overlap(self):
        from synth_local import _assemble
        segs = [self._seg("a", 0.0, 1.0), self._seg("b", 2.0, 3.0)]
        audios = [self._sine22(1.0), self._sine22(1.0)]
        result = _assemble(audios, segs)
        # The gap between segments [1.0, 2.0] should be ~silence
        gap_start = int(1.0 * OUTPUT_SR)
        gap_end = int(2.0 * OUTPUT_SR)
        assert np.max(np.abs(result[gap_start:gap_end])) < 0.05


# ──────────────────────────────────────────────────────────────────────────────
# _synthesize_edge_tts (mocked network call)
# ──────────────────────────────────────────────────────────────────────────────

class TestSynthesizeEdgeTts:
    def test_returns_list_of_arrays(self, mocker, tmp_dir):
        """Should return one numpy array per segment."""
        from synth_local import _synthesize_edge_tts
        from aligner import LyricSegment
        import soundfile as sf

        # Stub edge_tts.Communicate.save so it writes a valid wav
        async def fake_save(path):
            silence = np.zeros(SR, dtype=np.float32)
            sf.write(path.replace(".mp3", ".tmp.wav"), silence, SR)
            # edge_tts saves to path (mp3 here), librosa.load handles it
            import shutil
            shutil.copy(path.replace(".mp3", ".tmp.wav"), path)

        mocker.patch("edge_tts.Communicate", return_value=mocker.MagicMock(
            save=lambda p: fake_save(p)
        ))

        # Patch asyncio.run to actually call the coroutine synchronously
        original_run = asyncio.run

        def sync_run(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        mocker.patch("synth_local.asyncio.run", side_effect=sync_run)

        segs = [LyricSegment("hello", 0.0, 2.0), LyricSegment("world", 2.5, 5.0)]
        result = _synthesize_edge_tts(segs, language="en", is_male=False)
        assert len(result) == 2
        assert all(isinstance(a, np.ndarray) for a in result)

    def test_empty_segment_returns_silence(self, mocker):
        """Empty text segment should yield a silence array without calling edge-tts."""
        from synth_local import _synthesize_edge_tts
        from aligner import LyricSegment

        mocker.patch("edge_tts.Communicate")  # should never be called

        segs = [LyricSegment("", 0.0, 2.0)]
        result = _synthesize_edge_tts(segs, language="he", is_male=True)
        assert len(result) == 1
        assert np.all(result[0] == 0)

    def test_hebrew_uses_he_voice(self, mocker):
        """Hebrew language should select an he-IL voice."""
        from synth_local import _synthesize_edge_tts, EDGE_TTS_VOICES
        from aligner import LyricSegment

        captured_voice = []

        async def fake_save(path):
            import soundfile as sf
            sf.write(path, np.zeros(SR, dtype=np.float32), SR)

        class FakeCommunicate:
            def __init__(self, text, voice):
                captured_voice.append(voice)
                self.save = lambda p: fake_save(p)

        mocker.patch("edge_tts.Communicate", FakeCommunicate)
        mocker.patch("synth_local.asyncio.run", side_effect=lambda c: asyncio.get_event_loop().run_until_complete(c))

        segs = [LyricSegment("שלום", 0.0, 1.0)]
        _synthesize_edge_tts(segs, language="he", is_male=False)
        if captured_voice:
            assert "he-IL" in captured_voice[0]


# ──────────────────────────────────────────────────────────────────────────────
# clone_and_synthesize (integration, mocked heavy models)
# ──────────────────────────────────────────────────────────────────────────────

class TestLocalBackendIntegration:
    def test_english_calls_chatterbox(self, mocker, vocals_wav, tmp_dir):
        """English language should route to Chatterbox TTS."""
        from synth_local import clone_and_synthesize
        from aligner import LyricSegment

        mock_synth = mocker.patch(
            "synth_local._synthesize_chatterbox",
            return_value=[np.zeros(int(SR * 2), dtype=np.float32)],
        )
        mocker.patch(
            "synth_local._assemble",
            return_value=np.zeros(int(44100 * 3), dtype=np.float32),
        )

        segs = [LyricSegment("hello world", 0.0, 2.0)]
        out = os.path.join(tmp_dir, "out.wav")
        clone_and_synthesize(vocals_wav, segs, out, language="en")
        mock_synth.assert_called_once()

    def test_hebrew_calls_edge_tts_and_pitch_match(self, mocker, vocals_wav, tmp_dir):
        """Hebrew should use edge-tts then pitch matching."""
        from synth_local import clone_and_synthesize
        from aligner import LyricSegment

        fake_audio = [np.zeros(int(SR * 2), dtype=np.float32)]
        mocker.patch("synth_local._synthesize_edge_tts", return_value=fake_audio)
        mock_pitch = mocker.patch("synth_local._match_pitch", return_value=fake_audio)
        mocker.patch(
            "synth_local._assemble",
            return_value=np.zeros(int(44100 * 3), dtype=np.float32),
        )

        segs = [LyricSegment("שלום", 0.0, 2.0)]
        out = os.path.join(tmp_dir, "out.wav")
        clone_and_synthesize(vocals_wav, segs, out, language="he")
        mock_pitch.assert_called_once()

    def test_output_file_created(self, mocker, vocals_wav, tmp_dir):
        """Output WAV file should be created."""
        from synth_local import clone_and_synthesize
        from aligner import LyricSegment

        mocker.patch(
            "synth_local._synthesize_edge_tts",
            return_value=[np.zeros(int(SR * 2), dtype=np.float32)],
        )
        mocker.patch(
            "synth_local._match_pitch",
            return_value=[np.zeros(int(SR * 2), dtype=np.float32)],
        )

        segs = [LyricSegment("שלום", 0.0, 2.0)]
        out = os.path.join(tmp_dir, "synth_out.wav")
        clone_and_synthesize(vocals_wav, segs, out, language="he")
        assert os.path.exists(out)
