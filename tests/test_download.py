"""Unit checks for ytalk.app download fallbacks (offline — yt-dlp is stubbed).

Usage:
    PYTHONPATH=src python tests/test_download.py
"""
import os
import tempfile

import yt_dlp

from ytalk import app
from ytalk.app import JS_RUNTIMES, PLAYER_CLIENT_FALLBACKS, download_audio, find_js_runtime

INFO = {"title": "T", "uploader": "C", "duration": 42}


class FakeYDL:
    """Stand-in for yt_dlp.YoutubeDL that records opts and scripts outcomes."""

    calls = []      # opts dicts, one per attempt
    outcomes = []   # per attempt: None to succeed, or an exception to raise
    output_dir = ""

    def __init__(self, opts):
        self.opts = opts
        FakeYDL.calls.append(opts)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        outcome = FakeYDL.outcomes[len(FakeYDL.calls) - 1]
        if outcome is not None:
            raise outcome
        open(os.path.join(FakeYDL.output_dir, "audio.mp3"), "w").close()
        return dict(INFO)


def run_download(outcomes, tmpdir):
    """Drive download_audio with a scripted sequence of per-attempt outcomes."""
    FakeYDL.calls = []
    FakeYDL.outcomes = outcomes
    FakeYDL.output_dir = tmpdir
    real = app.yt_dlp.YoutubeDL
    app.yt_dlp.YoutubeDL = FakeYDL
    try:
        return download_audio("https://youtu.be/x", tmpdir)
    finally:
        app.yt_dlp.YoutubeDL = real


def test_first_attempt_uses_defaults():
    """The cheap path: yt-dlp's own client choice, no extractor_args override."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio, meta = run_download([None], tmpdir)

    assert len(FakeYDL.calls) == 1
    assert "extractor_args" not in FakeYDL.calls[0]
    # The EJS solver scripts are what make the challenge-solving clients work.
    assert FakeYDL.calls[0]["remote_components"] == ["ejs:github"]
    assert audio.endswith("audio.mp3")
    assert meta == {
        "title": "T", "channel": "C", "duration": 42, "url": "https://youtu.be/x",
    }


def test_403_falls_back_to_challenge_solving_clients():
    """A 403 on the default client retries with clients that solve the JS challenge."""
    err = yt_dlp.utils.DownloadError("unable to download video data: HTTP Error 403: Forbidden")
    with tempfile.TemporaryDirectory() as tmpdir:
        audio, _ = run_download([err, None], tmpdir)
        assert os.path.exists(audio)

    assert len(FakeYDL.calls) == 2
    assert FakeYDL.calls[1]["extractor_args"] == {
        "youtube": {"player_client": ["tv_simply", "web_safari"]}
    }


def test_missing_js_runtime_is_reported():
    """When every client fails and no JS runtime exists, say so — 403 alone is a dead end."""
    err = yt_dlp.utils.DownloadError("HTTP Error 403: Forbidden")
    outcomes = [err] * len(PLAYER_CLIENT_FALLBACKS)

    import shutil
    orig = shutil.which
    shutil.which = lambda name: None if name in JS_RUNTIMES else orig(name)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                run_download(outcomes, tmpdir)
            except RuntimeError as e:
                assert "brew install deno" in str(e)
                assert "403" in str(e)  # the underlying yt-dlp error is preserved
            else:
                raise AssertionError("expected RuntimeError about the JS runtime")
    finally:
        shutil.which = orig

    # Every client in the ladder was tried before giving up.
    assert len(FakeYDL.calls) == len(PLAYER_CLIENT_FALLBACKS)


def test_exhausted_fallbacks_reraise_ytdlp_error():
    """With a runtime present the JS hint would mislead, so surface yt-dlp's own error."""
    err = yt_dlp.utils.DownloadError("video unavailable")
    outcomes = [err] * len(PLAYER_CLIENT_FALLBACKS)

    import shutil
    orig = shutil.which
    shutil.which = lambda name: "/usr/bin/deno" if name in JS_RUNTIMES else orig(name)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                run_download(outcomes, tmpdir)
            except yt_dlp.utils.DownloadError as e:
                assert "video unavailable" in str(e)
            else:
                raise AssertionError("expected the yt-dlp DownloadError to propagate")
    finally:
        shutil.which = orig


def test_partial_files_are_not_mistaken_for_audio():
    """A failed attempt leaves audio.*.part behind; picking it would break Whisper."""
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "audio.mp4.part"), "w").close()
        open(os.path.join(tmpdir, "audio.webm"), "w").close()

        FakeYDL.calls = []
        FakeYDL.outcomes = [None]
        FakeYDL.output_dir = tmpdir
        real = app.yt_dlp.YoutubeDL

        # Succeed without producing audio.mp3, so the directory scan has to choose.
        class NoMp3(FakeYDL):
            def extract_info(self, url, download=False):
                return dict(INFO)

        app.yt_dlp.YoutubeDL = NoMp3
        try:
            audio, _ = download_audio("https://youtu.be/x", tmpdir)
        finally:
            app.yt_dlp.YoutubeDL = real

        assert audio.endswith("audio.webm"), audio


def test_find_js_runtime():
    """Detection prefers the first name in JS_RUNTIMES that is on PATH."""
    import shutil
    orig = shutil.which

    shutil.which = lambda name: None
    try:
        assert find_js_runtime() is None
    finally:
        shutil.which = orig

    shutil.which = lambda name: "/bin/bun" if name == "bun" else None
    try:
        assert find_js_runtime() == "bun"
    finally:
        shutil.which = orig


def main():
    test_first_attempt_uses_defaults()
    test_403_falls_back_to_challenge_solving_clients()
    test_missing_js_runtime_is_reported()
    test_exhausted_fallbacks_reraise_ytdlp_error()
    test_partial_files_are_not_mistaken_for_audio()
    test_find_js_runtime()
    print("PASS: download fallbacks, JS-runtime detection, and audio-file selection")


if __name__ == "__main__":
    main()
