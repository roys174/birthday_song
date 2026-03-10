"""Tests for aligner.py — timing extraction and lyric mapping."""

import pytest
from aligner import (
    LyricSegment,
    _split_lyrics,
    _map_to_lyrics_lines,
    map_new_lyrics,
    extract_timings,
)


# ──────────────────────────────────────────────────────────────────────────────
# LyricSegment
# ──────────────────────────────────────────────────────────────────────────────

class TestLyricSegment:
    def test_duration(self):
        seg = LyricSegment("hello", start=1.0, end=4.0)
        assert seg.duration == pytest.approx(3.0)

    def test_zero_duration(self):
        seg = LyricSegment("", start=5.0, end=5.0)
        assert seg.duration == pytest.approx(0.0)

    def test_fields(self):
        seg = LyricSegment("test text", 2.5, 7.0)
        assert seg.text == "test text"
        assert seg.start == 2.5
        assert seg.end == 7.0


# ──────────────────────────────────────────────────────────────────────────────
# _split_lyrics
# ──────────────────────────────────────────────────────────────────────────────

class TestSplitLyrics:
    def test_basic(self):
        lines = _split_lyrics("line one\nline two\nline three")
        assert lines == ["line one", "line two", "line three"]

    def test_strips_whitespace(self):
        lines = _split_lyrics("  hello  \n  world  ")
        assert lines == ["hello", "world"]

    def test_skips_blank_lines(self):
        lines = _split_lyrics("a\n\n\nb\n\nc")
        assert lines == ["a", "b", "c"]

    def test_empty_string(self):
        assert _split_lyrics("") == []

    def test_only_whitespace(self):
        assert _split_lyrics("   \n\n   ") == []

    def test_single_line(self):
        assert _split_lyrics("only one line") == ["only one line"]

    def test_windows_line_endings(self):
        lines = _split_lyrics("line1\r\nline2\r\nline3")
        assert lines == ["line1", "line2", "line3"]

    def test_hebrew_text(self):
        lyrics = "שלום עולם\nיום הולדת שמח"
        lines = _split_lyrics(lyrics)
        assert lines == ["שלום עולם", "יום הולדת שמח"]


# ──────────────────────────────────────────────────────────────────────────────
# _map_to_lyrics_lines
# ──────────────────────────────────────────────────────────────────────────────

class TestMapToLyricsLines:
    def test_exact_match(self):
        segments = [
            {"start": 0.0, "end": 2.0},
            {"start": 2.5, "end": 5.0},
            {"start": 5.5, "end": 8.0},
        ]
        lines = ["line one", "line two", "line three"]
        result = _map_to_lyrics_lines(segments, lines)
        assert len(result) == 3
        assert result[0].text == "line one"
        assert result[0].start == pytest.approx(0.0)
        assert result[0].end == pytest.approx(2.0)
        assert result[1].text == "line two"
        assert result[2].text == "line three"

    def test_no_segments_fallback(self):
        lines = ["a", "b", "c"]
        result = _map_to_lyrics_lines([], lines)
        assert len(result) == 3
        # Each line gets 3-second slots starting at 0
        assert result[0].start == pytest.approx(0.0)
        assert result[0].end == pytest.approx(3.0)
        assert result[1].start == pytest.approx(3.0)

    def test_mismatched_count_distributes(self):
        # More segments than lines → distributes proportionally
        segments = [{"start": i, "end": i + 1} for i in range(10)]
        lines = ["a", "b"]
        result = _map_to_lyrics_lines(segments, lines)
        assert len(result) == 2

    def test_fewer_segments_than_lines(self):
        segments = [{"start": 0.0, "end": 5.0}]
        lines = ["a", "b", "c"]
        result = _map_to_lyrics_lines(segments, lines)
        assert len(result) == 3
        for seg in result:
            assert seg.duration > 0

    def test_preserves_text(self):
        segs = [{"start": 0.0, "end": 3.0}]
        lines = ["hello world"]
        result = _map_to_lyrics_lines(segs, lines)
        assert result[0].text == "hello world"


# ──────────────────────────────────────────────────────────────────────────────
# map_new_lyrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMapNewLyrics:
    def _make_timed(self, texts):
        return [
            LyricSegment(t, i * 3.0, i * 3.0 + 2.5)
            for i, t in enumerate(texts)
        ]

    def test_same_line_count(self):
        orig = self._make_timed(["hello", "world", "foo"])
        result = map_new_lyrics(orig, "shalom\nolam\nbar")
        assert len(result) == 3
        assert result[0].text == "shalom"
        assert result[0].start == pytest.approx(0.0)
        assert result[0].end == pytest.approx(2.5)
        assert result[1].text == "olam"
        assert result[2].text == "bar"

    def test_fewer_new_lines(self):
        orig = self._make_timed(["a", "b", "c", "d"])
        result = map_new_lyrics(orig, "x\ny")
        assert len(result) == 2
        assert result[0].text == "x"
        assert result[1].text == "y"

    def test_extra_new_lines(self):
        orig = self._make_timed(["a"])
        result = map_new_lyrics(orig, "x\ny\nz")
        assert len(result) == 3
        # Extra lines placed after the last segment
        assert result[1].start >= orig[-1].end
        assert result[2].start >= result[1].end

    def test_hebrew_new_lyrics(self):
        orig = self._make_timed(["Happy birthday to you"])
        result = map_new_lyrics(orig, "יום הולדת שמח לך")
        assert result[0].text == "יום הולדת שמח לך"
        assert result[0].start == pytest.approx(orig[0].start)
        assert result[0].end == pytest.approx(orig[0].end)

    def test_timing_preserved_exactly(self):
        orig = [
            LyricSegment("one", 1.2, 3.4),
            LyricSegment("two", 4.0, 6.8),
        ]
        result = map_new_lyrics(orig, "alpha\nbeta")
        assert result[0].start == pytest.approx(1.2)
        assert result[0].end == pytest.approx(3.4)
        assert result[1].start == pytest.approx(4.0)
        assert result[1].end == pytest.approx(6.8)

    def test_empty_lines_skipped(self):
        orig = self._make_timed(["a", "b"])
        result = map_new_lyrics(orig, "x\n\n\ny")
        assert len(result) == 2

    def test_whitespace_only_lines_skipped(self):
        orig = self._make_timed(["a"])
        result = map_new_lyrics(orig, "  x  ")
        assert result[0].text == "x"


# ──────────────────────────────────────────────────────────────────────────────
# extract_timings (mocked — avoids loading WhisperX model in unit tests)
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractTimings:
    def test_returns_lyric_segments(self, mocker, vocals_wav):
        fake_segments = [
            {"start": 0.0, "end": 2.0, "text": "happy birthday to you"},
            {"start": 2.5, "end": 5.0, "text": "happy birthday to you"},
        ]
        mock_model = mocker.MagicMock()
        mock_model.transcribe.return_value = {"segments": fake_segments}

        mocker.patch("whisperx.load_model", return_value=mock_model)
        mocker.patch("whisperx.load_audio", return_value=None)
        mocker.patch(
            "whisperx.load_align_model",
            side_effect=Exception("alignment unavailable"),
        )

        lyrics = "happy birthday to you\nhappy birthday to you"
        result = extract_timings(vocals_wav, lyrics, language="en")

        assert len(result) == 2
        assert all(isinstance(s, LyricSegment) for s in result)
        assert result[0].start == pytest.approx(0.0)
        assert result[0].end == pytest.approx(2.0)

    def test_handles_no_segments(self, mocker, vocals_wav):
        mock_model = mocker.MagicMock()
        mock_model.transcribe.return_value = {"segments": []}
        mocker.patch("whisperx.load_model", return_value=mock_model)
        mocker.patch("whisperx.load_audio", return_value=None)
        mocker.patch(
            "whisperx.load_align_model",
            side_effect=Exception("alignment unavailable"),
        )

        result = extract_timings(vocals_wav, "line one\nline two", language="en")
        # Fallback: 3 s per line
        assert len(result) == 2
        assert result[0].duration == pytest.approx(3.0)
