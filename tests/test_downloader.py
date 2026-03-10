"""Tests for downloader.py — audio download and validation."""

import os
import shutil

import pytest

from downloader import download_audio, _is_youtube_url


# ──────────────────────────────────────────────────────────────────────────────
# _is_youtube_url
# ──────────────────────────────────────────────────────────────────────────────

class TestIsYoutubeUrl:
    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "http://youtube.com/watch?v=abc",
        "www.youtube.com/watch?v=abc",
    ])
    def test_youtube_urls(self, url):
        assert _is_youtube_url(url) is True

    @pytest.mark.parametrize("source", [
        "/path/to/song.mp3",
        "song.wav",
        "https://vimeo.com/123456",
        "https://soundcloud.com/artist/track",
        "",
        "not_a_url",
    ])
    def test_non_youtube(self, source):
        assert _is_youtube_url(source) is False


# ──────────────────────────────────────────────────────────────────────────────
# download_audio — local file passthrough
# ──────────────────────────────────────────────────────────────────────────────

class TestDownloadAudioLocalFile:
    def test_copies_mp3(self, mono_wav, tmp_dir):
        """MP3 file should be copied to output dir."""
        import soundfile as sf
        import numpy as np

        mp3_path = mono_wav.replace(".wav", ".mp3")
        shutil.copy(mono_wav, mp3_path)

        out_dir = os.path.join(tmp_dir, "out")
        result = download_audio(mp3_path, out_dir)
        assert os.path.exists(result)
        assert result.endswith(".mp3")

    def test_converts_wav_to_mp3(self, mocker, mono_wav, tmp_dir):
        """WAV file should be converted to MP3 via ffmpeg."""
        out_dir = os.path.join(tmp_dir, "out")

        def fake_ffmpeg(cmd, **kwargs):
            # Simulate ffmpeg by copying the source to dest
            src = cmd[cmd.index("-i") + 1]
            dst = cmd[-1]
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            m = mocker.MagicMock()
            m.returncode = 0
            return m

        mocker.patch("subprocess.run", side_effect=fake_ffmpeg)
        result = download_audio(mono_wav, out_dir)
        assert result.endswith(".mp3")

    def test_missing_file_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            download_audio("/nonexistent/path/song.mp3", tmp_dir)

    def test_unsupported_format_raises(self, tmp_dir):
        fake = os.path.join(tmp_dir, "song.xyz")
        open(fake, "w").close()
        with pytest.raises(ValueError, match="Unsupported audio format"):
            download_audio(fake, tmp_dir)

    def test_creates_output_directory(self, mocker, tmp_dir):
        """Output directory should be created if it doesn't exist."""
        mp3_src = os.path.join(tmp_dir, "src.mp3")
        open(mp3_src, "wb").close()

        new_out = os.path.join(tmp_dir, "brand", "new", "dir")
        result = download_audio(mp3_src, new_out)
        assert os.path.isdir(new_out)


# ──────────────────────────────────────────────────────────────────────────────
# download_audio — YouTube path (mocked yt-dlp)
# ──────────────────────────────────────────────────────────────────────────────

class TestDownloadAudioYouTube:
    def test_calls_yt_dlp(self, mocker, tmp_dir):
        url = "https://www.youtube.com/watch?v=test123"
        out_dir = os.path.join(tmp_dir, "yt_out")

        run_mock = mocker.MagicMock()
        run_mock.returncode = 0
        mocker.patch("subprocess.run", return_value=run_mock)

        # yt-dlp writes the file; simulate that
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "input_audio.mp3")
        open(out_path, "wb").close()

        result = download_audio(url, out_dir)
        import subprocess
        subprocess.run.assert_called_once()
        args = subprocess.run.call_args[0][0]
        assert "yt-dlp" in args
        assert url in args

    def test_yt_dlp_failure_raises(self, mocker, tmp_dir):
        url = "https://www.youtube.com/watch?v=bad"
        run_mock = mocker.MagicMock()
        run_mock.returncode = 1
        run_mock.stderr = "Download failed"
        mocker.patch("subprocess.run", return_value=run_mock)

        with pytest.raises(RuntimeError, match="yt-dlp failed"):
            download_audio(url, tmp_dir)

    @pytest.mark.parametrize("url,expected", [
        ("https://www.youtube.com/watch?v=abc", True),
        ("https://youtu.be/abc",                True),
        ("https://youtube.com/watch?v=abc",     True),
    ])
    def test_youtube_url_detection(self, url, expected):
        assert _is_youtube_url(url) == expected
