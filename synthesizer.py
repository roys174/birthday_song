"""
Dispatcher: selects the synthesis backend based on the `backend` argument.

Supported backends:
  "elevenlabs" — ElevenLabs API (paid, best quality, supports Hebrew natively)
  "local"      — Coqui XTTS v2 + edge-tts (free, runs locally, no API key needed)
"""

from aligner import LyricSegment

BACKENDS = ("elevenlabs", "local")


def clone_and_synthesize(
    vocals_path: str,
    timed_new_lyrics: list[LyricSegment],
    output_path: str,
    language: str = "en",
    backend: str = "elevenlabs",
    api_key: str | None = None,
) -> str:
    """
    Synthesize new lyrics in the singer's voice.

    Args:
        vocals_path: Path to isolated vocals WAV (from Demucs).
        timed_new_lyrics: New lyrics with target timings.
        output_path: Where to save the synthesized vocals WAV.
        language: Language code of new lyrics ("en" or "he").
        backend: "elevenlabs" or "local".
        api_key: ElevenLabs API key (required when backend="elevenlabs").

    Returns:
        Path to synthesized vocals WAV.
    """
    if backend == "elevenlabs":
        if not api_key:
            raise ValueError(
                "ElevenLabs API key is required for the 'elevenlabs' backend. "
                "Use --backend local for the free alternative."
            )
        from synth_elevenlabs import clone_and_synthesize as _synth
        return _synth(
            vocals_path=vocals_path,
            timed_new_lyrics=timed_new_lyrics,
            output_path=output_path,
            language=language,
            api_key=api_key,
        )

    elif backend == "local":
        from synth_local import clone_and_synthesize as _synth
        return _synth(
            vocals_path=vocals_path,
            timed_new_lyrics=timed_new_lyrics,
            output_path=output_path,
            language=language,
        )

    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {BACKENDS}")
