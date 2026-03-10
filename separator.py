"""Separate vocals from instrumental using Demucs."""

import os
import subprocess
from pathlib import Path


def separate_audio(audio_path: str, output_dir: str) -> tuple[str, str]:
    """
    Separate vocals and instrumental tracks using Demucs (htdemucs model).

    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save separated tracks

    Returns:
        Tuple of (vocals_path, instrumental_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_name = Path(audio_path).stem

    # Run demucs - htdemucs separates into: drums, bass, other, vocals
    cmd = [
        "python3", "-m", "demucs",
        "--two-stems", "vocals",   # Only split into vocals + no_vocals
        "--out", output_dir,
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{result.stderr}")

    # Demucs output structure: output_dir/htdemucs/<audio_name>/vocals.wav
    # Find the output directory
    demucs_out = _find_demucs_output(output_dir, audio_name)

    vocals_path = os.path.join(demucs_out, "vocals.wav")
    instrumental_path = os.path.join(demucs_out, "no_vocals.wav")

    if not os.path.exists(vocals_path):
        raise RuntimeError(f"Vocals not found at expected path: {vocals_path}")
    if not os.path.exists(instrumental_path):
        raise RuntimeError(f"Instrumental not found at expected path: {instrumental_path}")

    return vocals_path, instrumental_path


def _find_demucs_output(output_dir: str, audio_name: str) -> str:
    """Find the demucs output directory for the given audio file."""
    # Demucs creates: output_dir/<model_name>/<audio_name>/
    for model_dir in Path(output_dir).iterdir():
        if model_dir.is_dir():
            track_dir = model_dir / audio_name
            if track_dir.exists():
                return str(track_dir)
    raise RuntimeError(
        f"Could not find demucs output for '{audio_name}' in {output_dir}"
    )
