"""Download audio from YouTube or pass through local MP3 files."""

import os
import subprocess
import tempfile
from pathlib import Path


def download_audio(source: str, output_dir: str) -> str:
    """
    Download audio from YouTube URL or copy/validate local MP3.

    Args:
        source: YouTube URL or path to local audio file
        output_dir: Directory to save the audio file

    Returns:
        Path to the downloaded/validated audio file (MP3 or WAV)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "input_audio.mp3")

    if _is_youtube_url(source):
        _download_youtube(source, output_path)
    else:
        # Local file - validate and copy/convert
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")
        if source_path.suffix.lower() not in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"):
            raise ValueError(f"Unsupported audio format: {source_path.suffix}")
        if source_path.suffix.lower() != ".mp3":
            # Convert to MP3 using ffmpeg
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(source_path), output_path],
                check=True,
                capture_output=True,
            )
        else:
            import shutil
            shutil.copy2(str(source_path), output_path)

    return output_path


def _is_youtube_url(source: str) -> bool:
    return any(
        domain in source
        for domain in ["youtube.com", "youtu.be", "www.youtube.com"]
    )


def _download_youtube(url: str, output_path: str) -> None:
    """Download audio from YouTube using yt-dlp."""
    cmd = [
        "yt-dlp",
        "--format", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--output", output_path,
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")
