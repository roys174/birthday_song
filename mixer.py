"""Mix synthesized vocals with the original instrumental track."""

import os
import numpy as np
import soundfile as sf
import librosa


def mix_tracks(
    instrumental_path: str,
    vocals_path: str,
    output_path: str,
    vocals_volume: float = 1.0,
    instrumental_volume: float = 1.0,
) -> str:
    """
    Mix synthesized vocals with the original instrumental.

    Args:
        instrumental_path: Path to instrumental WAV (from Demucs)
        vocals_path: Path to synthesized vocals WAV
        output_path: Where to save the final mix
        vocals_volume: Volume multiplier for vocals (default 1.0)
        instrumental_volume: Volume multiplier for instrumental (default 1.0)

    Returns:
        Path to the final mixed audio file
    """
    print("Mixing vocals with instrumental...")

    # Load both tracks
    instrumental, inst_sr = sf.read(instrumental_path, always_2d=True)
    vocals, voc_sr = sf.read(vocals_path)

    # Ensure mono vocals → stereo if needed
    if vocals.ndim == 1:
        vocals = np.stack([vocals, vocals], axis=1)

    # Resample to match if needed (use instrumental sample rate as reference)
    target_sr = inst_sr
    if voc_sr != target_sr:
        vocals = librosa.resample(
            vocals.T, orig_sr=voc_sr, target_sr=target_sr
        ).T

    # Pad shorter track with silence to match lengths
    inst_len = len(instrumental)
    voc_len = len(vocals)

    if inst_len > voc_len:
        pad = inst_len - voc_len
        vocals = np.pad(vocals, ((0, pad), (0, 0)))
    elif voc_len > inst_len:
        pad = voc_len - inst_len
        instrumental = np.pad(instrumental, ((0, pad), (0, 0)))

    # Mix
    mixed = instrumental_volume * instrumental + vocals_volume * vocals

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed * (0.95 / max_val)

    # Save as WAV
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, mixed, target_sr)
    print(f"Final mix saved to: {output_path}")

    return output_path
