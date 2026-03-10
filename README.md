# Lyric Replacement App

Replace a song's lyrics while keeping the **same singer's voice** and **original music**.
Supports **English** and **Hebrew** input/output.

## How it works

1. **Download** the song (YouTube URL or local MP3/WAV)
2. **Separate** vocals from instrumental using [Demucs](https://github.com/facebookresearch/demucs)
3. **Extract timing** of each lyric line from the original vocals using [WhisperX](https://github.com/m-bain/whisperX)
4. **Clone the singer's voice** using [ElevenLabs Instant Voice Cloning](https://elevenlabs.io)
5. **Synthesize** the new lyrics with the cloned voice, time-stretched to match original timing
6. **Mix** the synthesized vocals back with the original instrumental

## Setup

### 1. Install system dependencies

```bash
brew install ffmpeg rubberband
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your ElevenLabs API key
```

Get your ElevenLabs API key at https://elevenlabs.io (free tier available).

Optional: A HuggingFace token improves WhisperX alignment quality.

## Usage

### Web UI (recommended)

```bash
python app.py
```

Opens a Gradio interface in your browser.

### CLI

```bash
# English original → Hebrew new lyrics
python main.py \
  --source "https://www.youtube.com/watch?v=..." \
  --original-lyrics original.txt \
  --new-lyrics new_hebrew.txt \
  --original-lang en \
  --new-lang he \
  --output result.wav

# Local MP3 file
python main.py \
  --source song.mp3 \
  --original-lyrics original.txt \
  --new-lyrics new_lyrics.txt \
  --output result.wav
```

### Lyrics format

One line per row, same number of lines in original and new lyrics:

**original.txt:**
```
Happy birthday to you
Happy birthday to you
Happy birthday dear friend
Happy birthday to you
```

**new_hebrew.txt:**
```
יום הולדת שמח לך
יום הולדת שמח לך
יום הולדת יקיר ידידי
יום הולדת שמח לך
```

## Notes

- **ElevenLabs Instant Voice Cloning** works best with 1+ minutes of clean vocal audio. Shorter songs may have lower cloning quality.
- **Timing alignment**: The app maps new lyrics line-by-line to original timing. If the new lyrics have a very different syllable count, some lines may sound rushed or stretched.
- Processing time: ~3-10 minutes per song depending on length and your hardware.
- GPU (CUDA) significantly speeds up Demucs and WhisperX.

## Architecture

```
main.py / app.py     ← CLI & Gradio UI
    │
pipeline.py          ← Orchestrates all stages
    ├── downloader.py    ← yt-dlp (YouTube) or local file
    ├── separator.py     ← Demucs (vocals + instrumental)
    ├── aligner.py       ← WhisperX (timing extraction & mapping)
    ├── synthesizer.py   ← ElevenLabs (voice clone + TTS + time-stretch)
    └── mixer.py         ← Mix vocals + instrumental
```
