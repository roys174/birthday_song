"""Tests for pipeline.py — end-to-end orchestration (all stages mocked)."""

import os
import numpy as np
import pytest
import soundfile as sf

from pipeline import run_pipeline


ORIG_LYRICS = "Happy birthday to you\nHappy birthday to you"
NEW_LYRICS_EN = "Hello world to you\nHello world to you"
NEW_LYRICS_HE = "יום הולדת שמח\nיום הולדת שמח"


def _mock_all_stages(mocker, tmp_dir):
    """Patch all external stages so pipeline runs without real models."""
    # Stage 1: downloader
    dl_path = os.path.join(tmp_dir, "input.mp3")
    open(dl_path, "wb").close()
    mocker.patch("pipeline.download_audio", return_value=dl_path)

    # Stage 2: separator
    vocals_path = os.path.join(tmp_dir, "vocals.wav")
    inst_path = os.path.join(tmp_dir, "no_vocals.wav")
    sr = 44100
    silence = np.zeros((sr * 3, 2), dtype=np.float32)
    sf.write(vocals_path, np.zeros(sr * 3, dtype=np.float32), 22050)
    sf.write(inst_path, silence, sr)
    mocker.patch("pipeline.separate_audio", return_value=(vocals_path, inst_path))

    # Stage 3: aligner
    from aligner import LyricSegment
    timed_orig = [
        LyricSegment("Happy birthday to you", 0.0, 3.0),
        LyricSegment("Happy birthday to you", 3.5, 6.5),
    ]
    mocker.patch("pipeline.extract_timings", return_value=timed_orig)

    # Stage 4: map_new_lyrics — call through (pure function, no mock needed)

    # Stage 5: synthesizer
    synth_path = os.path.join(tmp_dir, "synth.wav")
    sf.write(synth_path, np.zeros(sr * 7, dtype=np.float32), sr)
    mocker.patch("pipeline.clone_and_synthesize", return_value=synth_path)

    # Stage 6: mixer
    final_path = os.path.join(tmp_dir, "final.wav")
    sf.write(final_path, silence, sr)
    mocker.patch("pipeline.mix_tracks", return_value=final_path)

    return {
        "vocals": vocals_path,
        "instrumental": inst_path,
        "synth": synth_path,
        "final": final_path,
    }


class TestPipeline:
    def test_runs_without_error(self, mocker, tmp_dir):
        paths = _mock_all_stages(mocker, tmp_dir)
        result = run_pipeline(
            source="song.mp3",
            original_lyrics=ORIG_LYRICS,
            new_lyrics=NEW_LYRICS_HE,
            output_path=paths["final"],
            backend="local",
        )
        assert result == paths["final"]

    def test_all_stages_called(self, mocker, tmp_dir):
        paths = _mock_all_stages(mocker, tmp_dir)
        run_pipeline(
            source="song.mp3",
            original_lyrics=ORIG_LYRICS,
            new_lyrics=NEW_LYRICS_EN,
            output_path=paths["final"],
            backend="local",
        )
        import pipeline as pl
        pl.download_audio.assert_called_once()
        pl.separate_audio.assert_called_once()
        pl.extract_timings.assert_called_once()
        pl.clone_and_synthesize.assert_called_once()
        pl.mix_tracks.assert_called_once()

    def test_backend_passed_to_synthesizer(self, mocker, tmp_dir):
        paths = _mock_all_stages(mocker, tmp_dir)
        run_pipeline(
            source="song.mp3",
            original_lyrics=ORIG_LYRICS,
            new_lyrics=NEW_LYRICS_EN,
            output_path=paths["final"],
            backend="elevenlabs",
            api_key="fake_key",
        )
        import pipeline as pl
        _, kwargs = pl.clone_and_synthesize.call_args
        assert kwargs.get("backend") == "elevenlabs"

    def test_api_key_passed_to_synthesizer(self, mocker, tmp_dir):
        paths = _mock_all_stages(mocker, tmp_dir)
        run_pipeline(
            source="song.mp3",
            original_lyrics=ORIG_LYRICS,
            new_lyrics=NEW_LYRICS_EN,
            output_path=paths["final"],
            backend="elevenlabs",
            api_key="my_secret_key",
        )
        import pipeline as pl
        _, kwargs = pl.clone_and_synthesize.call_args
        assert kwargs.get("api_key") == "my_secret_key"

    def test_language_params_forwarded(self, mocker, tmp_dir):
        paths = _mock_all_stages(mocker, tmp_dir)
        run_pipeline(
            source="song.mp3",
            original_lyrics=ORIG_LYRICS,
            new_lyrics=NEW_LYRICS_HE,
            output_path=paths["final"],
            original_language="en",
            new_language="he",
            backend="local",
        )
        import pipeline as pl
        _, kwargs = pl.extract_timings.call_args
        assert kwargs.get("language") == "en"
        _, kwargs = pl.clone_and_synthesize.call_args
        assert kwargs.get("language") == "he"

    def test_volume_params_forwarded(self, mocker, tmp_dir):
        paths = _mock_all_stages(mocker, tmp_dir)
        run_pipeline(
            source="song.mp3",
            original_lyrics=ORIG_LYRICS,
            new_lyrics=NEW_LYRICS_EN,
            output_path=paths["final"],
            backend="local",
            vocals_volume=0.7,
            instrumental_volume=1.3,
        )
        import pipeline as pl
        _, kwargs = pl.mix_tracks.call_args
        assert kwargs.get("vocals_volume") == pytest.approx(0.7)
        assert kwargs.get("instrumental_volume") == pytest.approx(1.3)

    def test_hf_token_forwarded(self, mocker, tmp_dir):
        paths = _mock_all_stages(mocker, tmp_dir)
        run_pipeline(
            source="song.mp3",
            original_lyrics=ORIG_LYRICS,
            new_lyrics=NEW_LYRICS_EN,
            output_path=paths["final"],
            backend="local",
            hf_token="hf_abc",
        )
        import pipeline as pl
        _, kwargs = pl.extract_timings.call_args
        assert kwargs.get("hf_token") == "hf_abc"

    def test_youtube_source_passes_url(self, mocker, tmp_dir):
        paths = _mock_all_stages(mocker, tmp_dir)
        url = "https://www.youtube.com/watch?v=test"
        run_pipeline(
            source=url,
            original_lyrics=ORIG_LYRICS,
            new_lyrics=NEW_LYRICS_EN,
            output_path=paths["final"],
            backend="local",
        )
        import pipeline as pl
        args, _ = pl.download_audio.call_args
        assert args[0] == url

    def test_temp_workdir_cleaned_up(self, mocker, tmp_dir):
        """When work_dir is None, temp dir should be cleaned up after."""
        import tempfile
        created_dirs = []

        original_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_dirs.append(d)
            return d

        mocker.patch("tempfile.mkdtemp", side_effect=tracking_mkdtemp)
        paths = _mock_all_stages(mocker, tmp_dir)

        run_pipeline(
            source="song.mp3",
            original_lyrics=ORIG_LYRICS,
            new_lyrics=NEW_LYRICS_EN,
            output_path=paths["final"],
            backend="local",
            work_dir=None,  # triggers temp dir
        )
        # Temp dirs created for this pipeline should no longer exist
        for d in created_dirs:
            if "birthday_song_" in d:
                assert not os.path.exists(d), f"Temp dir not cleaned up: {d}"

    def test_explicit_work_dir_not_deleted(self, mocker, tmp_dir):
        """When work_dir is explicitly provided, it should NOT be deleted."""
        paths = _mock_all_stages(mocker, tmp_dir)
        work = os.path.join(tmp_dir, "work")
        os.makedirs(work, exist_ok=True)

        run_pipeline(
            source="song.mp3",
            original_lyrics=ORIG_LYRICS,
            new_lyrics=NEW_LYRICS_EN,
            output_path=paths["final"],
            backend="local",
            work_dir=work,
        )
        assert os.path.isdir(work)
