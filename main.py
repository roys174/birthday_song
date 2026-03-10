"""CLI entry point for the lyric replacement pipeline."""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Replace song lyrics while keeping the same singer's voice.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Free local backend, English → Hebrew
  python main.py --source "https://youtube.com/watch?v=..." \\
                 --original-lyrics original.txt --new-lyrics new_he.txt \\
                 --original-lang en --new-lang he \\
                 --backend local --output output.wav

  # ElevenLabs backend, local MP3
  python main.py --source song.mp3 \\
                 --original-lyrics original.txt --new-lyrics new_en.txt \\
                 --backend elevenlabs --api-key YOUR_KEY --output output.wav
        """,
    )

    parser.add_argument("--source", required=True,
                        help="YouTube URL or path to local audio file")
    parser.add_argument("--original-lyrics", required=True,
                        help="Path to file with original lyrics (one line per row)")
    parser.add_argument("--new-lyrics", required=True,
                        help="Path to file with new lyrics (aligned to original)")
    parser.add_argument("--output", default="output.wav",
                        help="Output audio file path (default: output.wav)")
    parser.add_argument("--original-lang", default="en", choices=["en", "he"],
                        help="Language of original lyrics (default: en)")
    parser.add_argument("--new-lang", default="en", choices=["en", "he"],
                        help="Language of new lyrics (default: en)")
    parser.add_argument("--backend", default="elevenlabs", choices=["elevenlabs", "local"],
                        help="Synthesis backend: 'elevenlabs' (paid) or 'local' (free, default: elevenlabs)")
    parser.add_argument("--api-key", default=None,
                        help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var). Required for --backend elevenlabs")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token for WhisperX alignment (or set HF_TOKEN env var)")
    parser.add_argument("--vocals-volume", type=float, default=1.0,
                        help="Volume multiplier for synthesized vocals (default: 1.0)")
    parser.add_argument("--instrumental-volume", type=float, default=1.0,
                        help="Volume multiplier for instrumental (default: 1.0)")
    parser.add_argument("--work-dir", default=None,
                        help="Directory for intermediate files (temp dir if not specified)")
    parser.add_argument("--ui", action="store_true",
                        help="Launch the Gradio web UI instead of CLI processing")

    args = parser.parse_args()

    if args.ui:
        from app import build_ui
        demo = build_ui()
        demo.launch(share=False)
        return

    # Resolve API key (only required for ElevenLabs)
    api_key = None
    if args.backend == "elevenlabs":
        api_key = args.api_key or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            print(
                "Error: ElevenLabs API key is required for --backend elevenlabs.\n"
                "       Use --api-key or set ELEVENLABS_API_KEY, or switch to --backend local."
            )
            sys.exit(1)

    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    try:
        with open(args.original_lyrics, encoding="utf-8") as f:
            original_lyrics = f.read()
        with open(args.new_lyrics, encoding="utf-8") as f:
            new_lyrics = f.read()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    from pipeline import run_pipeline

    run_pipeline(
        source=args.source,
        original_lyrics=original_lyrics,
        new_lyrics=new_lyrics,
        output_path=args.output,
        original_language=args.original_lang,
        new_language=args.new_lang,
        backend=args.backend,
        api_key=api_key,
        hf_token=hf_token,
        work_dir=args.work_dir,
        vocals_volume=args.vocals_volume,
        instrumental_volume=args.instrumental_volume,
    )


if __name__ == "__main__":
    main()
