"""
Main processing pipeline: ties together all stages.
"""

import os
import tempfile

from downloader import download_audio
from separator import separate_audio
from aligner import extract_timings, map_new_lyrics
from synthesizer import clone_and_synthesize, BACKENDS
from mixer import mix_tracks


def run_pipeline(
    source: str,
    original_lyrics: str,
    new_lyrics: str,
    output_path: str,
    original_language: str = "en",
    new_language: str = "en",
    backend: str = "elevenlabs",
    api_key: str | None = None,
    hf_token: str | None = None,
    work_dir: str | None = None,
    vocals_volume: float = 1.0,
    instrumental_volume: float = 1.0,
) -> str:
    """
    Full pipeline: download → separate → align → synthesize → mix.

    Args:
        source: YouTube URL or path to MP3/audio file.
        original_lyrics: Lyrics of the original song.
        new_lyrics: New lyrics (aligned line-by-line to original).
        output_path: Where to save the final output WAV.
        original_language: Language of original lyrics ("en" or "he").
        new_language: Language of new lyrics ("en" or "he").
        backend: Synthesis backend — "elevenlabs" (paid) or "local" (free).
        api_key: ElevenLabs API key (required when backend="elevenlabs").
        hf_token: Optional HuggingFace token for WhisperX alignment.
        work_dir: Directory for intermediate files (temp dir if None).
        vocals_volume: Volume multiplier for synthesized vocals in final mix.
        instrumental_volume: Volume multiplier for instrumental in final mix.

    Returns:
        Path to the final output audio file.
    """
    use_temp = work_dir is None
    if use_temp:
        work_dir = tempfile.mkdtemp(prefix="birthday_song_")

    try:
        print(f"=== Stage 1: Downloading/loading audio ===")
        audio_path = download_audio(source, os.path.join(work_dir, "input"))

        print(f"\n=== Stage 2: Separating vocals from instrumental ===")
        vocals_path, instrumental_path = separate_audio(
            audio_path, os.path.join(work_dir, "separated")
        )

        print(f"\n=== Stage 3: Extracting timing from original vocals ===")
        timed_original = extract_timings(
            vocals_path=vocals_path,
            original_lyrics=original_lyrics,
            language=original_language,
            hf_token=hf_token,
        )
        print(f"Found {len(timed_original)} timed segments")
        for seg in timed_original:
            print(f"  [{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text[:60]}")

        print(f"\n=== Stage 4: Mapping new lyrics to original timing ===")
        timed_new = map_new_lyrics(timed_original, new_lyrics)
        for seg in timed_new:
            print(f"  [{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text[:60]}")

        print(f"\n=== Stage 5: Synthesizing new lyrics (backend={backend}) ===")
        synth_vocals_path = os.path.join(work_dir, "synthesized_vocals.wav")
        clone_and_synthesize(
            vocals_path=vocals_path,
            timed_new_lyrics=timed_new,
            output_path=synth_vocals_path,
            language=new_language,
            backend=backend,
            api_key=api_key,
        )

        print(f"\n=== Stage 6: Mixing vocals with instrumental ===")
        final_path = mix_tracks(
            instrumental_path=instrumental_path,
            vocals_path=synth_vocals_path,
            output_path=output_path,
            vocals_volume=vocals_volume,
            instrumental_volume=instrumental_volume,
        )

        print(f"\n=== Done! Output: {final_path} ===")
        return final_path

    finally:
        if use_temp:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
