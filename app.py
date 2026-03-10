"""
Gradio web interface for the birthday song / lyric replacement app.
"""

import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from pipeline import run_pipeline

LANGUAGE_CODES = {
    "English": "en",
    "Hebrew (עברית)": "he",
}

BACKEND_LABELS = {
    "ElevenLabs (paid, best quality)": "elevenlabs",
    "Local / Free (XTTS v2 + edge-tts)": "local",
}

EXAMPLE_ORIGINAL_EN = """\
Happy birthday to you
Happy birthday to you
Happy birthday dear friend
Happy birthday to you"""

EXAMPLE_NEW_HE = """\
יום הולדת שמח לך
יום הולדת שמח לך
יום הולדת יקיר ידידי
יום הולדת שמח לך"""


def process_song(
    source_type: str,
    youtube_url: str,
    audio_file,
    original_lyrics: str,
    new_lyrics: str,
    original_language: str,
    new_language: str,
    backend_label: str,
    api_key: str,
    hf_token: str,
    vocals_volume: float,
    instrumental_volume: float,
) -> tuple[str | None, str]:
    backend = BACKEND_LABELS.get(backend_label, "elevenlabs")

    # Validate API key for ElevenLabs
    if backend == "elevenlabs":
        if not api_key.strip():
            api_key = os.getenv("ELEVENLABS_API_KEY", "")
        if not api_key.strip():
            return None, "Error: ElevenLabs API key is required for the ElevenLabs backend."
    else:
        api_key = None  # not needed

    if not original_lyrics.strip():
        return None, "Error: Original lyrics are required."
    if not new_lyrics.strip():
        return None, "Error: New lyrics are required."

    if source_type == "YouTube URL":
        if not youtube_url.strip():
            return None, "Error: Please provide a YouTube URL."
        source = youtube_url.strip()
    else:
        if audio_file is None:
            return None, "Error: Please upload an audio file."
        source = audio_file

    orig_lang = LANGUAGE_CODES.get(original_language, "en")
    new_lang = LANGUAGE_CODES.get(new_language, "en")

    output_dir = tempfile.mkdtemp(prefix="birthday_output_")
    output_path = os.path.join(output_dir, "output.wav")

    try:
        run_pipeline(
            source=source,
            original_lyrics=original_lyrics,
            new_lyrics=new_lyrics,
            output_path=output_path,
            original_language=orig_lang,
            new_language=new_lang,
            backend=backend,
            api_key=api_key or None,
            hf_token=hf_token.strip() or None,
            vocals_volume=vocals_volume,
            instrumental_volume=instrumental_volume,
        )
        return output_path, "Done! Your song is ready."
    except Exception as e:
        return None, f"Error: {e}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Lyric Replacement - Same Singer, New Words",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # Lyric Replacement App
            Replace song lyrics while keeping the same singer's voice and music.
            Supports **English** and **Hebrew** input/output.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Song")

                source_type = gr.Radio(
                    choices=["YouTube URL", "Upload MP3"],
                    value="YouTube URL",
                    label="Source type",
                )
                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                audio_file = gr.Audio(
                    label="Upload audio file (MP3, WAV, etc.)",
                    type="filepath",
                    visible=False,
                )

                def toggle_source(choice):
                    return (
                        gr.update(visible=choice == "YouTube URL"),
                        gr.update(visible=choice == "Upload MP3"),
                    )

                source_type.change(
                    toggle_source,
                    inputs=source_type,
                    outputs=[youtube_url, audio_file],
                )

                gr.Markdown("### Lyrics")

                with gr.Row():
                    orig_lang = gr.Dropdown(
                        choices=list(LANGUAGE_CODES.keys()),
                        value="English",
                        label="Original language",
                    )
                    new_lang = gr.Dropdown(
                        choices=list(LANGUAGE_CODES.keys()),
                        value="Hebrew (עברית)",
                        label="New lyrics language",
                    )

                original_lyrics = gr.Textbox(
                    label="Original lyrics (one line per row)",
                    lines=8,
                    placeholder="Enter original lyrics here...",
                    value=EXAMPLE_ORIGINAL_EN,
                )
                new_lyrics = gr.Textbox(
                    label="New lyrics (same number of lines, aligned to original)",
                    lines=8,
                    placeholder="Enter new lyrics here...",
                    value=EXAMPLE_NEW_HE,
                    rtl=True,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Settings")

                backend_radio = gr.Radio(
                    choices=list(BACKEND_LABELS.keys()),
                    value=list(BACKEND_LABELS.keys())[0],
                    label="Synthesis backend",
                )

                api_key_box = gr.Textbox(
                    label="ElevenLabs API Key",
                    type="password",
                    placeholder="Leave blank to use ELEVENLABS_API_KEY env var",
                    value=os.getenv("ELEVENLABS_API_KEY", ""),
                    visible=True,
                )

                local_note = gr.Markdown(
                    "_No API key needed. XTTS v2 (~2 GB) downloads on first run._",
                    visible=False,
                )

                def toggle_backend(choice):
                    is_el = BACKEND_LABELS.get(choice) == "elevenlabs"
                    return gr.update(visible=is_el), gr.update(visible=not is_el)

                backend_radio.change(
                    toggle_backend,
                    inputs=backend_radio,
                    outputs=[api_key_box, local_note],
                )

                hf_token = gr.Textbox(
                    label="HuggingFace Token (optional, improves timing alignment)",
                    type="password",
                    placeholder="hf_...",
                    value=os.getenv("HF_TOKEN", ""),
                )

                with gr.Row():
                    vocals_vol = gr.Slider(
                        minimum=0.0, maximum=2.0, value=1.0, step=0.05,
                        label="Vocals volume",
                    )
                    inst_vol = gr.Slider(
                        minimum=0.0, maximum=2.0, value=1.0, step=0.05,
                        label="Instrumental volume",
                    )

                gr.Markdown("### Output")

                run_btn = gr.Button("Generate Song", variant="primary", size="lg")
                status_msg = gr.Textbox(label="Status", interactive=False)
                output_audio = gr.Audio(label="Output song", type="filepath")

                run_btn.click(
                    fn=process_song,
                    inputs=[
                        source_type, youtube_url, audio_file,
                        original_lyrics, new_lyrics,
                        orig_lang, new_lang,
                        backend_radio, api_key_box, hf_token,
                        vocals_vol, inst_vol,
                    ],
                    outputs=[output_audio, status_msg],
                )

        gr.Markdown(
            """
            ---
            **Backend comparison:**
            | | ElevenLabs | Local / Free |
            |---|---|---|
            | Cost | Paid (API key) | Free |
            | Voice cloning quality | Excellent | Good (XTTS v2) |
            | Hebrew | Native | edge-tts + pitch match |
            | Runs offline | No | Yes |
            | First-run download | — | ~2 GB (XTTS v2 model) |
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False)
