"""
Free local backend:
- English (and other Chatterbox-supported languages): Chatterbox TTS
  Zero-shot voice cloning from a reference audio sample.
- Hebrew: edge-tts (Microsoft Edge TTS, free, native he-IL voices)
  + pitch-shift to match the singer's vocal range.

Chatterbox TTS: https://github.com/resemble-ai/chatterbox (MIT license)
edge-tts:       https://github.com/rany2/edge-tts (free, unofficial Edge API)
"""

import asyncio
import os
import tempfile

import numpy as np
import librosa
import soundfile as sf
import torch
from aligner import LyricSegment

# Languages handled natively by Chatterbox (English-focused model)
CHATTERBOX_LANGUAGES = {"en"}

# edge-tts voices by language + detected singer gender
EDGE_TTS_VOICES = {
    "he": {"male": "he-IL-AvriNeural",  "female": "he-IL-HilaNeural"},
    "en": {"male": "en-US-GuyNeural",   "female": "en-US-JennyNeural"},
}
EDGE_TTS_FALLBACK = {"male": "en-US-GuyNeural", "female": "en-US-JennyNeural"}

TARGET_SR = 22050
OUTPUT_SR = 44100

# Detect best available device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def clone_and_synthesize(
    vocals_path: str,
    timed_new_lyrics: list[LyricSegment],
    output_path: str,
    language: str,
    api_key: str | None = None,  # unused, kept for interface compatibility
) -> str:
    """
    Synthesize new lyrics using free local models.

    - For English: Chatterbox TTS with zero-shot voice cloning.
    - For Hebrew (and other unsupported languages): edge-tts + pitch matching
      to approximate the singer's vocal range.
    """
    singer_f0 = _median_f0(vocals_path)
    is_male = singer_f0 < 180
    print(f"Singer: {'male' if is_male else 'female'} (median F0: {singer_f0:.1f} Hz), device: {DEVICE}")

    if language in CHATTERBOX_LANGUAGES:
        print(f"Using Chatterbox TTS (zero-shot voice cloning) for language '{language}'")
        raw_audios = _synthesize_chatterbox(vocals_path, timed_new_lyrics)
    else:
        print(f"Language '{language}' — using edge-tts + pitch matching")
        raw_audios = _synthesize_edge_tts(timed_new_lyrics, language, is_male)
        raw_audios = _match_pitch(raw_audios, singer_f0)

    final = _assemble(raw_audios, timed_new_lyrics)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, final, OUTPUT_SR)
    print(f"Saved: {output_path}")
    return output_path


# --------------------------------------------------------------------------- #
# Chatterbox TTS (English, zero-shot voice cloning)
# --------------------------------------------------------------------------- #

def _synthesize_chatterbox(
    vocals_path: str,
    segments: list[LyricSegment],
) -> list[np.ndarray]:
    from chatterbox.tts import ChatterboxTTS

    print("Loading Chatterbox TTS model (downloads on first run)...")
    model = ChatterboxTTS.from_pretrained(DEVICE)

    results = []
    for i, seg in enumerate(segments):
        if not seg.text.strip():
            results.append(np.zeros(int(seg.duration * TARGET_SR)))
            continue
        print(f"  Chatterbox [{i+1}/{len(segments)}]: {seg.text[:60]}")
        wav = model.generate(seg.text, audio_prompt_path=vocals_path)
        # wav is a torch tensor (1, samples) at model.sr
        audio = wav.squeeze().cpu().numpy().astype(np.float32)
        sr = model.sr
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        results.append(audio)

    return results


# --------------------------------------------------------------------------- #
# edge-tts (Hebrew and other unsupported languages)
# --------------------------------------------------------------------------- #

def _synthesize_edge_tts(
    segments: list[LyricSegment],
    language: str,
    is_male: bool,
) -> list[np.ndarray]:
    gender = "male" if is_male else "female"
    voices = EDGE_TTS_VOICES.get(language, EDGE_TTS_FALLBACK)
    voice = voices[gender]
    print(f"edge-tts voice: {voice}")

    results = []
    for i, seg in enumerate(segments):
        if not seg.text.strip():
            results.append(np.zeros(int(seg.duration * TARGET_SR)))
            continue
        print(f"  edge-tts [{i+1}/{len(segments)}]: {seg.text[:60]}")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            asyncio.run(_edge_save(seg.text, voice, tmp_path))
            audio, _ = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
            results.append(audio.astype(np.float32))
        finally:
            os.unlink(tmp_path)

    return results


async def _edge_save(text: str, voice: str, path: str) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)


# --------------------------------------------------------------------------- #
# Pitch analysis & matching
# --------------------------------------------------------------------------- #

def _median_f0(vocals_path: str) -> float:
    """Return median F0 of the vocalist (first 60 s of isolated vocals)."""
    audio, sr = librosa.load(vocals_path, sr=TARGET_SR, mono=True, duration=60)
    f0, voiced, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    voiced_f0 = f0[voiced] if voiced is not None else f0
    if voiced_f0 is not None:
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
    return float(np.median(voiced_f0)) if voiced_f0 is not None and len(voiced_f0) > 0 else 200.0


def _match_pitch(audios: list[np.ndarray], target_f0: float) -> list[np.ndarray]:
    """Pitch-shift each segment so its median F0 matches the singer's."""
    results = []
    for audio in audios:
        if len(audio) < 512:
            results.append(audio)
            continue
        try:
            f0, voiced, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=TARGET_SR,
            )
            voiced_f0 = f0[voiced] if voiced is not None else f0
            if voiced_f0 is not None:
                voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
            if voiced_f0 is None or len(voiced_f0) == 0:
                results.append(audio)
                continue
            src_f0 = float(np.median(voiced_f0))
            if src_f0 < 1.0:
                results.append(audio)
                continue
            semitones = float(np.clip(12 * np.log2(target_f0 / src_f0), -12, 12))
            results.append(librosa.effects.pitch_shift(audio, sr=TARGET_SR, n_steps=semitones))
        except Exception as e:
            print(f"  Pitch shift failed ({e}), using original")
            results.append(audio)
    return results


# --------------------------------------------------------------------------- #
# Assembly: time-stretch each segment to its target duration, place in output
# --------------------------------------------------------------------------- #

def _assemble(audios: list[np.ndarray], segments: list[LyricSegment]) -> np.ndarray:
    import pyrubberband as pyrb

    last = segments[-1] if segments else None
    total_samples = int((last.end + 1.0) * OUTPUT_SR) if last else OUTPUT_SR
    output = np.zeros(total_samples)

    for audio, seg in zip(audios, segments):
        if len(audio) == 0:
            continue
        target_dur = seg.duration
        current_dur = len(audio) / TARGET_SR
        if current_dur < 0.01:
            continue

        ratio = max(0.3, min(3.0, target_dur / current_dur))
        try:
            stretched = pyrb.time_stretch(audio, TARGET_SR, ratio)
        except Exception:
            stretched = audio

        if TARGET_SR != OUTPUT_SR:
            stretched = librosa.resample(stretched, orig_sr=TARGET_SR, target_sr=OUTPUT_SR)

        start = int(seg.start * OUTPUT_SR)
        end = start + len(stretched)
        if end > len(output):
            output = np.pad(output, (0, end - len(output)))
        output[start:end] += stretched

    mx = np.max(np.abs(output))
    if mx > 0.95:
        output *= 0.95 / mx

    return output
