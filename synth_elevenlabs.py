"""ElevenLabs backend: instant voice cloning + multilingual TTS."""

import os
import tempfile

import numpy as np
import soundfile as sf
import librosa
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from aligner import LyricSegment

ELEVENLABS_MULTILINGUAL_MODEL = "eleven_multilingual_v2"


def clone_and_synthesize(
    vocals_path: str,
    timed_new_lyrics: list[LyricSegment],
    output_path: str,
    language: str,
    api_key: str,
) -> str:
    client = ElevenLabs(api_key=api_key)

    print("Cloning singer's voice with ElevenLabs...")
    voice_id = _clone_voice(client, vocals_path)
    print(f"Voice cloned: {voice_id}")

    segments_audio = []
    for i, segment in enumerate(timed_new_lyrics):
        if not segment.text.strip():
            segments_audio.append((np.zeros(int(segment.duration * 44100)), 44100, segment))
            continue
        print(f"  [{i+1}/{len(timed_new_lyrics)}]: {segment.text[:60]}")
        audio_data, sr = _synthesize_line(client, voice_id, segment.text, language)
        segments_audio.append((audio_data, sr, segment))

    final_audio = _assemble_segments(segments_audio)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, final_audio, 44100)
    print(f"Saved: {output_path}")

    try:
        client.voices.delete(voice_id)
    except Exception:
        pass

    return output_path


def _clone_voice(client, vocals_path: str) -> str:
    with open(vocals_path, "rb") as f:
        voice = client.voices.ivc.create(
            name="BirthdaySongSinger",
            files=[("vocals.wav", f, "audio/wav")],
            description="Auto-cloned singer voice for lyric replacement",
        )
    return voice.voice_id


def _synthesize_line(client, voice_id: str, text: str, language: str) -> tuple[np.ndarray, int]:
    audio_stream = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=ELEVENLABS_MULTILINGUAL_MODEL,
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.85,
            style=0.3,
            use_speaker_boost=True,
        ),
        output_format="pcm_44100",
    )
    audio_bytes = b"".join(audio_stream)
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_data, 44100


def _assemble_segments(segments_audio: list[tuple[np.ndarray, int, LyricSegment]]) -> np.ndarray:
    import pyrubberband as pyrb

    if not segments_audio:
        return np.zeros(44100)

    last = segments_audio[-1][2]
    total_samples = int((last.end + 1.0) * 44100)
    output = np.zeros(total_samples)

    for audio_data, sample_rate, segment in segments_audio:
        if len(audio_data) == 0:
            continue
        target_duration = segment.duration
        current_duration = len(audio_data) / sample_rate
        if current_duration < 0.01:
            continue

        ratio = target_duration / current_duration
        if ratio > 2.0:
            # Segment is much longer than speech — place at natural pace
            stretched = audio_data
        elif ratio < 0.5:
            # Need to compress a lot — trim instead
            stretched = audio_data[:int(target_duration * sample_rate)]
        else:
            try:
                stretched = pyrb.time_stretch(audio_data, sample_rate, ratio)
            except Exception as e:
                print(f"  Time-stretch failed ({e}), using original")
                stretched = audio_data

        if sample_rate != 44100:
            stretched = librosa.resample(stretched, orig_sr=sample_rate, target_sr=44100)

        start = int(segment.start * 44100)
        end = start + len(stretched)
        if end > len(output):
            output = np.pad(output, (0, end - len(output)))
        output[start:end] += stretched

    mx = np.max(np.abs(output))
    if mx > 0.95:
        output *= 0.95 / mx

    return output
