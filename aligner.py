"""
Extract timing information from original vocals using WhisperX,
then map new lyrics to those timings.
"""

import os
import re
from dataclasses import dataclass


@dataclass
class LyricSegment:
    """A segment of lyrics with timing."""
    text: str
    start: float  # seconds
    end: float    # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


def extract_timings(
    vocals_path: str,
    original_lyrics: str,
    language: str = "en",
    hf_token: str | None = None,
) -> list[LyricSegment]:
    """
    Use WhisperX to transcribe with timestamps, then align to provided lyrics.

    Args:
        vocals_path: Path to isolated vocals WAV
        original_lyrics: Original lyrics text (used to guide alignment)
        language: Language code ("en" or "he")
        hf_token: HuggingFace token for alignment models

    Returns:
        List of LyricSegment with start/end times per line
    """
    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Loading WhisperX model on {device}...")
    model = whisperx.load_model(
        "large-v2",
        device=device,
        compute_type=compute_type,
        language=language,
    )

    print("Transcribing vocals...")
    audio = whisperx.load_audio(vocals_path)
    result = model.transcribe(audio, batch_size=16, language=language)

    print("Aligning transcription...")
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        segments = result["segments"]
    except Exception as e:
        print(f"Alignment failed ({e}), using transcription segments directly")
        segments = result.get("segments", [])

    # Map transcribed segments to lines of original lyrics
    original_lines = _split_lyrics(original_lyrics)
    timed_segments = _map_to_lyrics_lines(segments, original_lines)

    return timed_segments


def map_new_lyrics(
    timed_original: list[LyricSegment],
    new_lyrics: str,
) -> list[LyricSegment]:
    """
    Map new lyrics lines to the timing of original lyrics lines.

    Assumes new_lyrics has the same number of lines (or close to) as original.
    Each line i of new_lyrics gets the timing of line i of original.

    Args:
        timed_original: Timed segments from original lyrics
        new_lyrics: New lyrics text (same structure as original)

    Returns:
        List of LyricSegment for new lyrics with original timings
    """
    new_lines = _split_lyrics(new_lyrics)

    if len(new_lines) != len(timed_original):
        print(
            f"Warning: new lyrics has {len(new_lines)} lines, "
            f"original has {len(timed_original)} lines. "
            "Mapping as best as possible."
        )

    result = []
    for i, new_line in enumerate(new_lines):
        if i < len(timed_original):
            orig = timed_original[i]
            result.append(LyricSegment(
                text=new_line,
                start=orig.start,
                end=orig.end,
            ))
        else:
            # Extra lines: sequence them one after another from the last known end
            prev_end = result[-1].end if result else 0.0
            result.append(LyricSegment(
                text=new_line,
                start=prev_end,
                end=prev_end + 3.0,
            ))

    return result


def _split_lyrics(lyrics: str) -> list[str]:
    """Split lyrics into non-empty lines, stripping whitespace."""
    return [line.strip() for line in lyrics.splitlines() if line.strip()]


def _map_to_lyrics_lines(
    segments: list[dict],
    lyrics_lines: list[str],
) -> list[LyricSegment]:
    """
    Map WhisperX segments to lyrics lines by distributing segments
    proportionally across lines.
    """
    if not segments:
        # Fallback: assume 3 seconds per line starting at 0
        result = []
        t = 0.0
        for line in lyrics_lines:
            result.append(LyricSegment(text=line, start=t, end=t + 3.0))
            t += 3.0
        return result

    # If segments roughly match lines, do 1:1 mapping
    if len(segments) == len(lyrics_lines):
        return [
            LyricSegment(
                text=line,
                start=seg.get("start", 0.0),
                end=seg.get("end", seg.get("start", 0.0) + 3.0),
            )
            for line, seg in zip(lyrics_lines, segments)
        ]

    # Otherwise, distribute segments across lines proportionally
    total_duration = segments[-1].get("end", 0) - segments[0].get("start", 0)
    start_time = segments[0].get("start", 0)
    per_line = total_duration / max(len(lyrics_lines), 1)

    result = []
    for i, line in enumerate(lyrics_lines):
        t = start_time + i * per_line
        result.append(LyricSegment(text=line, start=t, end=t + per_line))

    return result
