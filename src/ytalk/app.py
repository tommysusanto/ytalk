#!/usr/bin/env python3
"""
yTalk – Download a YouTube video, transcribe it with Whisper, and chat about it with Ollama.

Usage: python ytalk.py <youtube_url> [--model <ollama_model>]

Requirements:
    pip install openai-whisper requests
    brew install yt-dlp ffmpeg  (or pip install yt-dlp)
    ollama must be running with a model pulled
"""

import argparse
import base64
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import requests
import whisper
import yt_dlp
from rich.markup import escape as rich_escape
from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    Markdown,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)


LOG_PATH = Path.home() / ".cache" / "ytalk" / "ytalk.log"


def _init_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("ytalk")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    log.addHandler(handler)
    log.propagate = False
    try:
        from importlib.metadata import version as _pkg_version
        ytalk_version = _pkg_version("ytalk")
    except Exception:
        ytalk_version = "unknown"
    log.info(
        "ytalk session start: version=%s python=%s",
        ytalk_version,
        sys.version.split()[0],
    )
    return log


logger = _init_logger()


# JS runtimes yt-dlp can drive to solve YouTube's challenges, best first.
JS_RUNTIMES = ("deno", "node", "bun")

JS_RUNTIME_HINT = "No JavaScript runtime found. Install: brew install deno"

# YouTube only hands out media URLs to players that solve its JavaScript "n"
# challenge. yt-dlp's default client (android_vr) skips the challenge but its
# URLs now 403, so we retry with clients that do solve it. Which clients YouTube
# honours changes often, hence a ladder rather than a single choice. None means
# "whatever yt-dlp defaults to" -- cheapest, and right again whenever upstream
# catches up.
PLAYER_CLIENT_FALLBACKS = (None, "tv_simply,web_safari", "mweb,web")


def find_js_runtime() -> str | None:
    """Return the first JS runtime on PATH that yt-dlp can use, else None."""
    import shutil

    for runtime in JS_RUNTIMES:
        if shutil.which(runtime):
            return runtime
    return None


def download_audio(url: str, output_dir: str, progress_callback=None) -> tuple[str, dict]:
    """Download audio from YouTube using yt-dlp Python API.

    Returns (audio_file_path, metadata_dict).
    """
    output_path = os.path.join(output_dir, "audio.%(ext)s")

    def _progress_hook(d):
        if progress_callback and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            if total:
                pct = downloaded / total * 100
                progress_callback(f"Status: Downloading: {pct:.1f}%")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "0",
        }],
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": True,
        # quiet doesn't cover the progress bar, and writing it to stdout garbles
        # the TUI. _progress_hook feeds the status line instead.
        "noprogress": True,
        "progress_hooks": [_progress_hook],
        # Fetches (and caches) the EJS challenge-solver scripts that the JS
        # runtime runs. Without them every format 403s.
        "remote_components": ["ejs:github"],
    }
    print(f"Downloading audio from: {url}")

    info = None
    last_error = None
    for attempt, player_client in enumerate(PLAYER_CLIENT_FALLBACKS):
        opts = dict(ydl_opts)
        if player_client:
            opts["extractor_args"] = {"youtube": {"player_client": player_client.split(",")}}
            if progress_callback:
                progress_callback(f"Status: Retrying download (player: {player_client})...")
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            logger.warning(
                "download: attempt %d/%d failed (player_client=%s): %s",
                attempt + 1, len(PLAYER_CLIENT_FALLBACKS), player_client or "default", e,
            )

    if info is None:
        if find_js_runtime() is None:
            raise RuntimeError(
                f"{JS_RUNTIME_HINT} — YouTube requires one to unlock video downloads. "
                f"Last yt-dlp error: {last_error}"
            )
        raise last_error

    meta = {
        "title": info.get("title") or "Unknown title",
        "channel": info.get("uploader") or info.get("channel") or "Unknown channel",
        "duration": int(info.get("duration") or 0),
        "url": url,
    }

    audio_file = os.path.join(output_dir, "audio.mp3")
    if not os.path.exists(audio_file):
        for f in sorted(os.listdir(output_dir)):
            # .part/.ytdl leftovers from a failed attempt are not playable audio.
            if f.startswith("audio.") and not f.endswith((".part", ".ytdl")):
                audio_file = os.path.join(output_dir, f)
                break
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"No audio file found in {output_dir}")
    print(f"Audio saved to: {audio_file}")
    return audio_file, meta


def transcribe(audio_path: str, whisper_model: str = "base", progress_callback=None) -> tuple[str, list[dict]]:
    """Transcribe audio using OpenAI Whisper. Returns (text, segments)."""
    # Prevent tqdm from creating a multiprocessing RLock, which spawns a
    # resource-tracker subprocess and fails with "bad value(s) in fds_to_keep"
    # when called from a Textual worker thread.
    import tqdm.std as tqdm_std
    tqdm_std.TqdmDefaultWriteLock.mp_lock = threading.RLock()

    # Monkey-patch tqdm in whisper.transcribe to intercept progress updates.
    # whisper.transcribe does `import tqdm` then `tqdm.tqdm(...)`, so we
    # replace the class attribute on the tqdm module directly.
    import tqdm as _tqdm_pkg
    _OrigTqdm = _tqdm_pkg.tqdm

    if progress_callback:
        class _ProgressTqdm(_OrigTqdm):
            def update(self, n=1):
                super().update(n)
                if self.total:
                    pct = self.n / self.total * 100
                    progress_callback(f"Status: Transcribing: {pct:.1f}%")

        _tqdm_pkg.tqdm = _ProgressTqdm

    try:
        print(f"Loading Whisper model '{whisper_model}'...")
        model = whisper.load_model(whisper_model)
        print("Transcribing (this may take a while)...")
        result = model.transcribe(audio_path, verbose=False)
    finally:
        _tqdm_pkg.tqdm = _OrigTqdm

    text = result["text"].strip()
    segments = result.get("segments", []) or []
    print(f"Transcription complete ({len(text)} chars, {len(segments)} segments).")
    return text, segments


def format_timestamp(seconds: float) -> str:
    """Seconds -> compact YouTube timestamp: 'M:SS', or 'H:MM:SS' when over an hour."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_transcript_segments(
    segments: list[dict],
    pause_threshold: float = 1.5,
    max_paragraph_seconds: float = 30.0,
    timestamp_fmt=None,
) -> str:
    """Group Whisper segments into paragraphs on >pause_threshold pauses or every
    ~max_paragraph_seconds. Each paragraph is prefixed with a timestamp.

    By default the prefix is zero-padded ``[mm:ss]``. Pass ``timestamp_fmt`` (a
    callable taking seconds) for a different prefix — e.g. ``format_timestamp`` for
    hour-aware ``[H:MM:SS]`` markers."""
    if not segments:
        return ""
    paragraphs: list[tuple[float, str]] = []
    current_texts: list[str] = [segments[0]["text"].strip()]
    current_start = segments[0]["start"]
    last_end = segments[0]["end"]
    for seg in segments[1:]:
        gap = seg["start"] - last_end
        running = seg["start"] - current_start
        if gap > pause_threshold or running > max_paragraph_seconds:
            paragraphs.append((current_start, " ".join(current_texts)))
            current_texts = []
            current_start = seg["start"]
        current_texts.append(seg["text"].strip())
        last_end = seg["end"]
    if current_texts:
        paragraphs.append((current_start, " ".join(current_texts)))
    lines = []
    for start, text in paragraphs:
        if timestamp_fmt is not None:
            lines.append(f"[{timestamp_fmt(start)}] {text}")
        else:
            mm, ss = divmod(int(start), 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {text}")
    return "\n\n".join(lines)


MIN_CTX = 4096
DEFAULT_MAX_CTX = 32768

# Leave this fraction of the accelerator budget unused. The model weights and KV cache
# are not the only things on the GPU (compute buffers scale with context too), and
# overshooting on unified memory is not a soft failure -- see _gpu_memory_budget_bytes.
_MEM_SAFETY_FRACTION = 0.90

_mem_profile_cache: dict[str, tuple[int, int] | None] = {}


def _sysctl_int(name: str) -> int | None:
    """Read an integer sysctl, or None if it is missing/unreadable."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def _gpu_memory_budget_bytes() -> int | None:
    """Bytes the accelerator may hold at once, or None if undeterminable.

    On Apple Silicon the GPU shares system RAM, and macOS caps how much of it can be
    wired down (~75% by default, overridable via iogpu.wired_limit_mb). Wired pages
    cannot be swapped out, so once the model plus its KV cache exceed that cap the
    overflow lands in host RAM that the wired allocation has already starved -- the
    machine thrashes and locks up instead of Ollama failing cleanly. Sizing the
    context against this budget is what keeps a long transcript from taking the box down.
    """
    if sys.platform != "darwin":
        return None
    wired_mb = _sysctl_int("iogpu.wired_limit_mb")
    if wired_mb:  # 0 means "unset, use the default"
        return wired_mb * 1024 * 1024
    total = _sysctl_int("hw.memsize")
    if not total:
        return None
    return int(total * 0.75)


def _model_memory_profile(model: str) -> tuple[int, int] | None:
    """(weight_bytes, kv_cache_bytes_per_token) for `model`, or None if unavailable.

    KV bytes/token comes from the model's own architecture metadata:
    block_count * head_count_kv * (key_length + value_length) * 2 bytes (f16 K and V).
    """
    if model in _mem_profile_cache:
        return _mem_profile_cache[model]

    profile = None
    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=5).json()
        weights = next(
            (m.get("size") for m in tags.get("models", []) if m.get("name") == model), None
        )
        info = requests.post(
            "http://localhost:11434/api/show", json={"model": model}, timeout=5
        ).json().get("model_info", {})

        def _arch(suffix: str):
            return next((v for k, v in info.items() if k.endswith(suffix)), None)

        blocks = _arch(".block_count")
        kv_heads = _arch(".attention.head_count_kv")
        key_len = _arch(".attention.key_length")
        val_len = _arch(".attention.value_length")
        if key_len is None or val_len is None:
            # Models that omit explicit head dims: derive it from embedding width.
            embed, heads = _arch(".embedding_length"), _arch(".attention.head_count")
            if embed and heads:
                key_len = val_len = embed // heads
        if weights and blocks and kv_heads and key_len and val_len:
            profile = (int(weights), int(blocks) * int(kv_heads) * (int(key_len) + int(val_len)) * 2)
    except (requests.RequestException, ValueError, TypeError, KeyError):
        profile = None

    _mem_profile_cache[model] = profile
    return profile


def _memory_capped_ctx(model: str) -> int | None:
    """Largest num_ctx whose KV cache still fits alongside `model`'s weights.

    None when the budget or the model's shape can't be determined, in which case the
    caller falls back to sizing on prompt length alone.
    """
    budget = _gpu_memory_budget_bytes()
    profile = _model_memory_profile(model)
    if not budget or not profile:
        return None
    weights, kv_per_token = profile
    spare = int(budget * _MEM_SAFETY_FRACTION) - weights
    if spare <= 0:
        # Weights alone overshoot the budget; nothing we pick here is safe, so ask for
        # the smallest window and let Ollama do its own (partial-offload) thing.
        return MIN_CTX
    ctx = MIN_CTX
    while ctx * 2 <= DEFAULT_MAX_CTX and (ctx * 2) * kv_per_token <= spare:
        ctx *= 2
    return ctx


def _memory_warning(model: str) -> str | None:
    """A user-facing note when `model` is too big for this machine, else None.

    Worth surfacing: the symptom of an oversized model is the whole system paging
    itself to death, which reads as "my computer froze", not "yTalk picked a bad model".
    """
    capped = _memory_capped_ctx(model)
    if capped is None or capped > MIN_CTX:
        return None
    budget = _gpu_memory_budget_bytes()
    profile = _model_memory_profile(model)
    if not budget or not profile:
        return None
    gb = 1024 ** 3
    return (
        f"{model} needs ~{profile[0] / gb:.1f}GB and this machine can devote "
        f"~{budget / gb:.1f}GB to it, so context is limited to {capped} tokens and long "
        f"transcripts will be trimmed. A smaller model will answer better here."
    )


def _ctx_for(
    prompt: str,
    output_headroom: int = 2048,
    max_ctx: int = DEFAULT_MAX_CTX,
    model: str | None = None,
) -> int:
    """Pick a num_ctx that fits the whole prompt plus room to generate.

    Ollama's default context window (4096) silently truncates longer prompts; with the
    default num_predict that also leaves ~no budget to generate, so a long transcript
    yields a single token. Size the window to the prompt instead, in power-of-two buckets
    so the same transcript reuses one loaded model instance.

    When `model` is given, the window is additionally capped to what actually fits in
    memory beside that model's weights -- a big model on a small machine gets a smaller
    window rather than one that wires up more memory than the system has.
    """
    needed = len(prompt) // 3 + output_headroom  # cheap, slightly conservative token est.
    if model:
        capped = _memory_capped_ctx(model)
        if capped:
            max_ctx = min(max_ctx, capped)
    ctx = MIN_CTX
    while ctx < needed and ctx < max_ctx:
        ctx *= 2
    return min(ctx, max_ctx)


def _fit_prompt(prompt: str, ctx: int, output_headroom: int = 2048, marker: str = "\n\n[... transcript trimmed to fit the context window ...]\n\n") -> str:
    """Trim the middle of `prompt` so it fits `ctx`, keeping the head and tail.

    Without this, a prompt longer than the (now memory-capped) window is silently
    truncated by Ollama -- it drops the tail, which is where the instructions and the
    'SUMMARY:' cue live, so the model answers the wrong question. Cutting the middle
    ourselves keeps both the task framing and the end of the transcript intact.
    """
    budget_chars = max(0, (ctx - output_headroom)) * 3
    if budget_chars <= 0 or len(prompt) <= budget_chars:
        return prompt
    keep = budget_chars - len(marker)
    if keep <= 0:
        return prompt[:budget_chars]
    head = keep * 2 // 3
    return prompt[:head] + marker + prompt[len(prompt) - (keep - head):]


def ollama_generate(prompt: str, model: str, progress_callback=None, status_callback=None) -> str:
    """Stream a completion from Ollama's /api/generate endpoint.

    progress_callback receives the accumulated text after each token. status_callback
    receives "thinking" once (if the model emits thinking tokens before any output)
    and "generating" when the first real response token arrives.
    """
    import json as _json

    ctx = _ctx_for(prompt, model=model)
    prompt = _fit_prompt(prompt, ctx)
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_ctx": ctx, "num_predict": -1},
        },
        timeout=(10, None),
        stream=True,
    )
    resp.raise_for_status()
    accumulated = ""
    thinking = True
    thinking_notified = False
    for line in resp.iter_lines():
        if not line:
            continue
        obj = _json.loads(line)
        token = obj.get("response", "")
        if thinking and not token and obj.get("thinking"):
            if status_callback and not thinking_notified:
                thinking_notified = True
                status_callback("thinking")
            continue
        if token:
            if thinking:
                thinking = False
                if status_callback:
                    status_callback("generating")
            accumulated += token
            if progress_callback:
                progress_callback(accumulated)
    return accumulated.strip()


def summarize(text: str, model: str = "gemma3:4b", progress_callback=None, status_callback=None) -> str:
    """Summarize text using Ollama's API. progress_callback receives accumulated text."""
    print(f"Summarizing with Ollama model '{model}'...")
    prompt = (
        "You are a helpful assistant. Summarize the following transcript concisely. "
        "Include the key points and main ideas.\n"
        "Format rules:\n"
        "- Use markdown headers (##) for sections\n"
        "- Use dashes (-) for bullet points, not asterisks\n"
        "- Keep bullet points short (one or two sentences max)\n"
        "- Use sub-bullets with indentation where needed\n\n"
        f"TRANSCRIPT:\n{text}\n\n"
        "SUMMARY:"
    )
    return ollama_generate(prompt, model, progress_callback, status_callback)


def _clean_chapter_list(text: str) -> str:
    """Strip stray code fences and blank lines a model may wrap around the list."""
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("```")]
    return "\n".join(lines)


def generate_timestamps(text: str, model: str = "gemma3:4b", progress_callback=None, status_callback=None) -> str:
    """Generate YouTube-style chapter timestamps from a timestamped transcript.

    `text` should be a transcript whose paragraphs are prefixed with [M:SS] /
    [H:MM:SS] markers (see format_transcript_segments + format_timestamp). Returns a
    bare, newline-separated chapter list ready to paste into a YouTube comment.
    """
    print(f"Generating timestamps with Ollama model '{model}'...")
    prompt = (
        "You generate YouTube chapter timestamps from a video transcript.\n"
        "Each transcript paragraph is prefixed with its start time as [M:SS] or [H:MM:SS].\n"
        "Produce a concise, chronological chapter list to paste into a YouTube comment.\n"
        "Rules:\n"
        "- The FIRST line must start at 0:00.\n"
        "- Use only timestamps that appear in the transcript; never invent times.\n"
        "- Pick natural topic boundaries; aim for about 5-12 chapters depending on length.\n"
        "- One chapter per line, EXACTLY: 'M:SS Title' (or 'H:MM:SS Title' past an hour).\n"
        "- Titles are concise (2-6 words), Title Case, no trailing punctuation.\n"
        "- Output ONLY the list. No preamble, no commentary, no code fences.\n\n"
        f"TRANSCRIPT:\n{text}\n\n"
        "CHAPTERS:"
    )
    result = ollama_generate(prompt, model, progress_callback, status_callback)
    return _clean_chapter_list(result)


def chat_query(transcript: str, question: str, history: list[dict], model: str, progress_callback=None) -> str:
    """Answer a question about the transcript using Ollama's chat API.
    progress_callback receives (accumulated_text, phase).
    """
    import json as _json

    system_prompt = (
        "You are a helpful assistant. The user has a transcript of a YouTube video. "
        "Answer their questions based on the transcript.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )

    # Size against everything actually sent -- prior turns count toward the window too,
    # so a long chat on a long transcript is the worst case, not the first question.
    history_chars = sum(len(m.get("content", "")) for m in history)
    ctx = _ctx_for(system_prompt + question + "x" * history_chars, model=model)
    # Trim the transcript rather than the conversation: the running chat is what the
    # user can see, so silently dropping it is more surprising than trimming the source.
    system_prompt = _fit_prompt(
        system_prompt, ctx, output_headroom=2048 + (len(question) + history_chars) // 3
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {"num_ctx": ctx, "num_predict": -1},
        },
        timeout=(10, None),
        stream=True,
    )
    resp.raise_for_status()
    accumulated = ""
    for line in resp.iter_lines():
        if not line:
            continue
        obj = _json.loads(line)
        token = obj.get("message", {}).get("content", "")
        if token:
            accumulated += token
            if progress_callback:
                progress_callback(accumulated, "generating")
    return accumulated.strip()


def fetch_ollama_models() -> list[str] | None:
    """Fetch available models from the local Ollama server.

    Returns None on connection failure, [] if no models, or a list of model names.
    """
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return None


WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]


class ChatTextArea(TextArea):
    """Multi-line input: Enter submits, Shift+Enter inserts newline."""

    BINDINGS = [Binding("enter", "submit", "Send", show=False, priority=True)]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def action_submit(self) -> None:
        value = self.text.strip()
        if value:
            self.text = ""
            self.post_message(self.Submitted(value))


class YTalkApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #app-header {
        dock: top;
        height: 4;
        background: $boost;
        color: $accent;
        content-align: center middle;
        text-style: bold;
    }
    TabbedContent {
        height: 1fr;
    }
    TabPane {
        padding: 1 2;
    }
    TabbedContent > Tabs {
        background: $boost;
    }
    TabbedContent Tab {
        color: $text;
        text-style: bold;
        padding: 0 3;
    }
    TabbedContent Tab.-active {
        color: $background;
        background: $accent;
        text-style: bold;
    }
    TabbedContent Tab:hover {
        color: $accent;
        background: $surface;
    }
    TabbedContent Tab.-active:hover {
        color: $background;
        background: $accent;
    }
    TabbedContent Tab:disabled {
        color: $text-muted 30%;
        text-style: dim;
        background: $boost;
    }
    TabbedContent Tab:disabled:hover {
        color: $text-muted 30%;
        background: $boost;
        text-style: dim;
    }
    TabbedContent Underline > .underline--bar {
        color: $accent;
        background: $boost;
    }

    /* Setup */
    #form {
        height: auto;
    }
    .form-row {
        height: 3;
        margin-bottom: 1;
    }
    .form-row Label {
        width: 18;
        content-align-vertical: middle;
        padding-top: 1;
    }
    .form-row Input, .form-row Select {
        width: 1fr;
    }
    #btn-row {
        height: 3;
        margin-bottom: 1;
        align-horizontal: center;
    }
    #run-btn {
        min-width: 20;
    }
    #status-bar {
        height: 1;
        padding: 0 2;
        background: $surface;
    }

    /* Transcript */
    #video-meta {
        height: auto;
        padding: 0 0 1 0;
        color: $text-muted;
    }
    #transcript-actions {
        height: 3;
        margin-bottom: 1;
    }
    #transcript-search {
        width: 1fr;
    }
    #transcript-actions Button {
        width: 10;
        margin-left: 1;
    }
    #transcript-log {
        height: 1fr;
        border: solid $primary;
    }

    /* Chat */
    #chat-scroll {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    .msg {
        margin-bottom: 1;
        padding: 0 1;
        height: auto;
    }
    .msg.user {
        color: $accent;
    }
    .msg.ai.streaming {
        border-left: thick $warning;
        padding-left: 1;
    }
    #chat-input-row {
        height: 6;
        margin-top: 1;
    }
    #chat-input {
        width: 1fr;
        height: 6;
    }
    #send-btn {
        width: 10;
        height: 6;
        margin-left: 1;
    }

    /* Summary */
    #summary-actions {
        height: 3;
        margin-bottom: 1;
    }
    #summary-actions Button {
        margin-right: 1;
    }
    #summary-scroll {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    #summary-md {
        height: auto;
    }

    /* Timestamps */
    #timestamps-actions {
        height: 3;
        margin-bottom: 1;
    }
    #timestamps-actions Button {
        margin-right: 1;
    }
    #timestamps-scroll {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    #timestamps-md {
        height: auto;
    }
    """

    TITLE = "yTalk"

    _transcript: str = ""
    _summary: str = ""
    _timestamps: str = ""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Rename the auto-registered palette binding from "palette" to "⚙ Settings".
        # Footer renders it in the right-docked slot; leave show=False to avoid a
        # duplicate entry in the main bindings strip.
        for _key, binding in self._bindings:
            if binding.action in {"command_palette", "app.command_palette"}:
                object.__setattr__(binding, "description", "⚙ Settings")
                object.__setattr__(binding, "tooltip", "Open settings")
                break

    def compose(self) -> ComposeResult:
        yield Static("yTalk", id="app-header", markup=True)
        with TabbedContent(id="tabs", initial="tab-setup"):
            with TabPane("⚙ Setup", id="tab-setup"):
                with Vertical(id="form"):
                    with Horizontal(classes="form-row"):
                        yield Label("YouTube URL:")
                        yield Input(placeholder="https://www.youtube.com/watch?v=...", id="url-input")
                    with Horizontal(classes="form-row"):
                        yield Label("Whisper Model:")
                        yield Select(
                            [(m, m) for m in WHISPER_MODELS],
                            value="base",
                            id="whisper-select",
                        )
                    with Horizontal(classes="form-row"):
                        yield Label("Chat Model:")
                        yield Select([], id="chat-select")
                    with Horizontal(id="btn-row"):
                        yield Button("Run", variant="primary", id="run-btn")
            with TabPane("🔒 Transcript", id="tab-transcript", disabled=True):
                yield Static("No video loaded yet.", id="video-meta", markup=True)
                with Horizontal(id="transcript-actions"):
                    yield Input(placeholder="Search transcript…", id="transcript-search")
                    yield Button("Copy", id="transcript-copy", disabled=True)
                    yield Button("Save", id="transcript-save", disabled=True)
                yield RichLog(id="transcript-log", wrap=True, markup=True, highlight=False)
            with TabPane("🔒 Chat", id="tab-chat", disabled=True):
                yield VerticalScroll(id="chat-scroll")
                with Horizontal(id="chat-input-row"):
                    yield ChatTextArea(id="chat-input")
                    yield Button("Send", id="send-btn", variant="primary", disabled=True)
            with TabPane("🔒 Summary", id="tab-summary", disabled=True):
                with Horizontal(id="summary-actions"):
                    yield Button("Generate", id="summarize-btn", variant="primary", disabled=True)
                    yield Button("Copy", id="summary-copy", disabled=True)
                    yield Button("Save", id="summary-save", disabled=True)
                with VerticalScroll(id="summary-scroll"):
                    yield Markdown("", id="summary-md")
            with TabPane("🔒 Timestamps", id="tab-timestamps", disabled=True):
                with Horizontal(id="timestamps-actions"):
                    yield Button("Generate", id="timestamps-btn", variant="primary", disabled=True)
                    yield Button("Copy", id="timestamps-copy", disabled=True)
                    yield Button("Save", id="timestamps-save", disabled=True)
                with VerticalScroll(id="timestamps-scroll"):
                    yield Markdown("", id="timestamps-md")
        yield Label("Status: Idle", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._chat_history: list[dict] = []
        self._streaming_msg: Markdown | None = None
        self._streaming_buf: str = ""
        self._stream_lock = threading.Lock()
        self._stream_handle = None
        self._summary_buf: str = ""
        self._summary_lock = threading.Lock()
        self._summary_flush_handle = None
        self._transcript_segments: list[dict] = []
        self._transcript_formatted: str = ""
        self._thinking_handle = None
        self._thinking_tick = 0
        self._summary_thinking_handle = None
        self._timestamps_buf: str = ""
        self._timestamps_lock = threading.Lock()
        self._timestamps_flush_handle = None
        self._timestamps_thinking_handle = None
        self._status_spinner_handle = None
        self._status_spinner_base = ""
        self._status_spinner_tick = 0
        tabs = self.query_one(TabbedContent)
        for tid in ("tab-transcript", "tab-chat", "tab-summary", "tab-timestamps"):
            tabs.get_tab(tid).tooltip = "Run the pipeline first"
        self._load_ollama_models()

    @work(thread=True)
    def _load_ollama_models(self) -> None:
        import shutil

        warnings = []

        if not shutil.which("ffmpeg"):
            warnings.append("ffmpeg not found. Install: brew install ffmpeg")

        if find_js_runtime() is None:
            warnings.append(JS_RUNTIME_HINT)

        models = fetch_ollama_models()
        if models is None:
            warnings.append("Ollama not found. Install from ollama.com, then run: ollama pull gemma3:4b")
        elif not models:
            warnings.append("No Ollama models found. Run: ollama pull gemma3:4b")

        if warnings:
            self._set_status("Status: " + " | ".join(warnings))
            if models is None or not models:
                return

        options = [(m, m) for m in models]

        chat_select = self.query_one("#chat-select", Select)
        self.call_from_thread(chat_select.set_options, options)

        if models:
            chat_default = models[0]
            for m in models:
                if m.startswith("gemma3") or m.startswith("phi4-mini"):
                    chat_default = m
            self.call_from_thread(setattr, chat_select, "value", chat_default)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "run-btn":
            self._run_pipeline()
        elif bid == "send-btn":
            self._send_from_input()
        elif bid == "summarize-btn":
            if self._transcript:
                self._run_summarize()
        elif bid == "transcript-copy" and self._transcript:
            self._copy_to_clipboard(self._transcript_formatted or self._transcript, "Transcript")
        elif bid == "transcript-save" and self._transcript:
            self._save_to_file("transcript", self._transcript_formatted or self._transcript)
        elif bid == "summary-copy" and self._summary:
            self._copy_to_clipboard(self._summary, "Summary")
        elif bid == "summary-save" and self._summary:
            self._save_to_file("summary", self._summary)
        elif bid == "timestamps-btn":
            if self._transcript:
                self._run_timestamps()
        elif bid == "timestamps-copy" and self._timestamps:
            self._copy_to_clipboard(self._timestamps, "Timestamps")
        elif bid == "timestamps-save" and self._timestamps:
            self._save_to_file("timestamps", self._timestamps)

    def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        if not self.query_one("#send-btn", Button).disabled:
            self._send_chat(event.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "transcript-search":
            self._render_transcript(event.value)

    def _send_from_input(self) -> None:
        chat_input = self.query_one("#chat-input", ChatTextArea)
        value = chat_input.text.strip()
        if value:
            chat_input.text = ""
            self._send_chat(value)

    @work(thread=True)
    def _run_pipeline(self) -> None:
        url = self.query_one("#url-input", Input).value.strip()
        if not url:
            self._set_status("Error: Please enter a YouTube URL")
            return

        whisper_model = self.query_one("#whisper-select", Select).value
        if whisper_model is Select.BLANK:
            whisper_model = "base"

        btn = self.query_one("#run-btn", Button)
        send_btn = self.query_one("#send-btn", Button)
        summarize_btn = self.query_one("#summarize-btn", Button)
        transcript_copy = self.query_one("#transcript-copy", Button)
        transcript_save = self.query_one("#transcript-save", Button)
        summary_copy = self.query_one("#summary-copy", Button)
        summary_save = self.query_one("#summary-save", Button)
        timestamps_btn = self.query_one("#timestamps-btn", Button)
        timestamps_copy = self.query_one("#timestamps-copy", Button)
        timestamps_save = self.query_one("#timestamps-save", Button)

        self.call_from_thread(setattr, btn, "disabled", True)
        self.call_from_thread(setattr, send_btn, "disabled", True)
        self.call_from_thread(setattr, summarize_btn, "disabled", True)
        self.call_from_thread(setattr, transcript_copy, "disabled", True)
        self.call_from_thread(setattr, transcript_save, "disabled", True)
        self.call_from_thread(setattr, summary_copy, "disabled", True)
        self.call_from_thread(setattr, summary_save, "disabled", True)
        self.call_from_thread(setattr, timestamps_btn, "disabled", True)
        self.call_from_thread(setattr, timestamps_copy, "disabled", True)
        self.call_from_thread(setattr, timestamps_save, "disabled", True)

        # Reset state
        self._transcript = ""
        self._summary = ""
        self._timestamps = ""
        self._chat_history = []

        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)

        def _clear_chat():
            for child in list(chat_scroll.children):
                child.remove()

        self.call_from_thread(_clear_chat)
        self.call_from_thread(self.query_one("#summary-md", Markdown).update, "")
        self.call_from_thread(self.query_one("#timestamps-md", Markdown).update, "")
        self.call_from_thread(self.query_one("#video-meta", Static).update, "Loading…")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                self._set_status("Status: Downloading audio...")
                logger.info("pipeline: download start url=%s", url)
                audio_path, meta = download_audio(url, tmpdir, progress_callback=self._set_status)
                logger.info("pipeline: download done duration=%ss", meta.get("duration"))

                mins, secs = divmod(meta["duration"], 60)
                meta_text = (
                    f"[b]{rich_escape(meta['title'])}[/b]\n"
                    f"{rich_escape(meta['channel'])}  •  {mins}:{secs:02d}"
                )
                self.call_from_thread(self.query_one("#video-meta", Static).update, meta_text)

                self._set_status("Status: Transcribing with Whisper...")
                logger.info("pipeline: transcribe start model=%s", whisper_model)
                transcript, segments = transcribe(audio_path, whisper_model, progress_callback=self._set_status)
                logger.info("pipeline: transcribe done chars=%d segments=%d", len(transcript), len(segments))
                os.remove(audio_path)

                self._transcript = transcript
                self._transcript_segments = segments
                self._transcript_formatted = format_transcript_segments(segments) or transcript
                self.call_from_thread(self._render_transcript, "")

                self.call_from_thread(setattr, summarize_btn, "disabled", False)
                self.call_from_thread(setattr, timestamps_btn, "disabled", False)
                self.call_from_thread(setattr, transcript_copy, "disabled", False)
                self.call_from_thread(setattr, transcript_save, "disabled", False)

            self._set_status("Status: Done! Review the transcript.")

            def _open_workspace():
                tabs = self.query_one(TabbedContent)
                for tid, label in (
                    ("tab-transcript", "Transcript"),
                    ("tab-chat", "Chat"),
                    ("tab-summary", "Summary"),
                    ("tab-timestamps", "Timestamps"),
                ):
                    tabs.enable_tab(tid)
                    tab = tabs.get_tab(tid)
                    tab.label = label
                    tab.tooltip = None
                tabs.active = "tab-transcript"

            self.call_from_thread(_open_workspace)
        except Exception as e:
            self._report_error("download/transcribe", e)
        finally:
            self.call_from_thread(setattr, btn, "disabled", False)
            if self._transcript:
                self.call_from_thread(setattr, send_btn, "disabled", False)

    @work(thread=True)
    def _run_summarize(self) -> None:
        summarize_btn = self.query_one("#summarize-btn", Button)
        copy_btn = self.query_one("#summary-copy", Button)
        save_btn = self.query_one("#summary-save", Button)
        self.call_from_thread(setattr, summarize_btn, "disabled", True)
        self.call_from_thread(setattr, copy_btn, "disabled", True)
        self.call_from_thread(setattr, save_btn, "disabled", True)

        def _switch_to_summary():
            self.query_one(TabbedContent).active = "tab-summary"

        self.call_from_thread(_switch_to_summary)

        chat_model = self.query_one("#chat-select", Select).value
        if chat_model is Select.BLANK:
            chat_model = "gemma3:4b"

        warning = _memory_warning(chat_model)
        if warning:
            logger.warning("memory: %s", warning)
            self._set_status(f"Status: {warning}")

        self._start_status_spinner(f"{chat_model} · generating summary")

        summary_md = self.query_one("#summary-md", Markdown)
        self.call_from_thread(summary_md.update, "*Generating summary…*")

        with self._summary_lock:
            self._summary_buf = ""

        def _start_timers():
            self._summary_flush_handle = self.set_interval(0.05, self._flush_summary)
            self._thinking_tick = 0
            self._summary_thinking_handle = self.set_interval(0.4, self._tick_thinking_summary)

        self.call_from_thread(_start_timers)

        try:
            def _on_summary_token(text):
                with self._summary_lock:
                    self._summary_buf = text

            def _on_summary_status(phase):
                if phase == "thinking":
                    self._set_status_spinner_base(f"{chat_model} · thinking")
                else:
                    self._set_status_spinner_base(f"{chat_model} · generating summary")

            result = summarize(
                self._transcript, chat_model,
                progress_callback=_on_summary_token,
                status_callback=_on_summary_status,
            )
            self._summary = result.strip()
            self.call_from_thread(self._stop_summary_thinking)
            self.call_from_thread(self._flush_summary)
            self.call_from_thread(summary_md.update, self._summary)
            self._stop_status_spinner()
            self._set_status("Status: Summary ready!")
        except Exception as e:
            self._report_error("summarize", e)
        finally:
            self.call_from_thread(self._stop_summary_thinking)
            if self._summary_flush_handle is not None:
                handle = self._summary_flush_handle
                self._summary_flush_handle = None
                self.call_from_thread(handle.stop)
            self.call_from_thread(setattr, summarize_btn, "disabled", False)
            if self._summary:
                self.call_from_thread(setattr, copy_btn, "disabled", False)
                self.call_from_thread(setattr, save_btn, "disabled", False)

    def _flush_summary(self) -> None:
        with self._summary_lock:
            buf = self._summary_buf
        if buf:
            self.query_one("#summary-md", Markdown).update(buf)

    def _tick_thinking_summary(self) -> None:
        with self._summary_lock:
            has_buf = bool(self._summary_buf)
        if has_buf:
            self._stop_summary_thinking()
            return
        self._thinking_tick += 1
        dots = "." * (1 + (self._thinking_tick % 3))
        self.query_one("#summary-md", Markdown).update(f"*Generating summary{dots}*")

    def _stop_summary_thinking(self) -> None:
        if self._summary_thinking_handle is not None:
            self._summary_thinking_handle.stop()
            self._summary_thinking_handle = None

    @work(thread=True)
    def _run_timestamps(self) -> None:
        timestamps_btn = self.query_one("#timestamps-btn", Button)
        copy_btn = self.query_one("#timestamps-copy", Button)
        save_btn = self.query_one("#timestamps-save", Button)
        self.call_from_thread(setattr, timestamps_btn, "disabled", True)
        self.call_from_thread(setattr, copy_btn, "disabled", True)
        self.call_from_thread(setattr, save_btn, "disabled", True)

        def _switch_to_timestamps():
            self.query_one(TabbedContent).active = "tab-timestamps"

        self.call_from_thread(_switch_to_timestamps)

        chat_model = self.query_one("#chat-select", Select).value
        if chat_model is Select.BLANK:
            chat_model = "gemma3:4b"

        self._start_status_spinner(f"{chat_model} · generating timestamps")

        timestamps_md = self.query_one("#timestamps-md", Markdown)
        self.call_from_thread(timestamps_md.update, "*Generating timestamps…*")

        with self._timestamps_lock:
            self._timestamps_buf = ""

        def _start_timers():
            self._timestamps_flush_handle = self.set_interval(0.05, self._flush_timestamps)
            self._thinking_tick = 0
            self._timestamps_thinking_handle = self.set_interval(0.4, self._tick_thinking_timestamps)

        self.call_from_thread(_start_timers)

        try:
            transcript_input = (
                format_transcript_segments(self._transcript_segments, timestamp_fmt=format_timestamp)
                if self._transcript_segments
                else self._transcript
            )

            def _on_timestamps_token(text):
                with self._timestamps_lock:
                    self._timestamps_buf = text

            def _on_timestamps_status(phase):
                if phase == "thinking":
                    self._set_status_spinner_base(f"{chat_model} · thinking")
                else:
                    self._set_status_spinner_base(f"{chat_model} · generating timestamps")

            result = generate_timestamps(
                transcript_input, chat_model,
                progress_callback=_on_timestamps_token,
                status_callback=_on_timestamps_status,
            )
            self._timestamps = result.strip()
            self.call_from_thread(self._stop_timestamps_thinking)
            self.call_from_thread(self._flush_timestamps)
            if self._timestamps:
                self.call_from_thread(timestamps_md.update, f"```text\n{self._timestamps}\n```")
            self._stop_status_spinner()
            self._set_status("Status: Timestamps ready! Copy and paste into a YouTube comment.")
        except Exception as e:
            self._report_error("timestamps", e)
        finally:
            self.call_from_thread(self._stop_timestamps_thinking)
            if self._timestamps_flush_handle is not None:
                handle = self._timestamps_flush_handle
                self._timestamps_flush_handle = None
                self.call_from_thread(handle.stop)
            self.call_from_thread(setattr, timestamps_btn, "disabled", False)
            if self._timestamps:
                self.call_from_thread(setattr, copy_btn, "disabled", False)
                self.call_from_thread(setattr, save_btn, "disabled", False)

    def _flush_timestamps(self) -> None:
        with self._timestamps_lock:
            buf = self._timestamps_buf
        if buf:
            self.query_one("#timestamps-md", Markdown).update(f"```text\n{buf}\n```")

    def _tick_thinking_timestamps(self) -> None:
        with self._timestamps_lock:
            has_buf = bool(self._timestamps_buf)
        if has_buf:
            self._stop_timestamps_thinking()
            return
        self._thinking_tick += 1
        dots = "." * (1 + (self._thinking_tick % 3))
        self.query_one("#timestamps-md", Markdown).update(f"*Generating timestamps{dots}*")

    def _stop_timestamps_thinking(self) -> None:
        if self._timestamps_thinking_handle is not None:
            self._timestamps_thinking_handle.stop()
            self._timestamps_thinking_handle = None

    def _send_chat(self, question: str) -> None:
        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        user_msg = Static(
            f"[bold cyan]You:[/] {rich_escape(question)}",
            markup=True,
            classes="msg user",
        )
        chat_scroll.mount(user_msg)
        ai_msg = Markdown("*Thinking.*", classes="msg ai")
        ai_msg.add_class("streaming")
        chat_scroll.mount(ai_msg)
        chat_scroll.scroll_end(animate=False)
        self._streaming_msg = ai_msg
        self._chat_history.append({"role": "user", "content": question})
        self._run_chat_query(question)

    @work(thread=True)
    def _run_chat_query(self, question: str) -> None:
        send_btn = self.query_one("#send-btn", Button)
        self.call_from_thread(setattr, send_btn, "disabled", True)

        chat_model = self.query_one("#chat-select", Select).value
        if chat_model is Select.BLANK:
            chat_model = "gemma3:4b"

        self._start_status_spinner(f"{chat_model} · answering")

        with self._stream_lock:
            self._streaming_buf = ""

        def _start_timers():
            self._stream_handle = self.set_interval(0.05, self._flush_stream)
            self._thinking_tick = 0
            self._thinking_handle = self.set_interval(0.4, self._tick_thinking)

        self.call_from_thread(_start_timers)

        try:
            def _on_chat_token(text, phase):
                with self._stream_lock:
                    self._streaming_buf = text

            answer = chat_query(
                self._transcript, question, self._chat_history[:-1], chat_model,
                progress_callback=_on_chat_token,
            )
            self._chat_history.append({"role": "assistant", "content": answer})
            self.call_from_thread(self._stop_thinking)
            self.call_from_thread(self._flush_stream)
            if self._streaming_msg is not None:
                msg = self._streaming_msg
                self.call_from_thread(msg.update, answer)
                self.call_from_thread(msg.remove_class, "streaming")
            self._stop_status_spinner()
            self._set_status("Status: Done! Ask another question.")
        except Exception as e:
            self._report_error("chat", e)
        finally:
            self.call_from_thread(self._stop_thinking)
            if self._stream_handle is not None:
                handle = self._stream_handle
                self._stream_handle = None
                self.call_from_thread(handle.stop)
            self.call_from_thread(setattr, send_btn, "disabled", False)

    def _flush_stream(self) -> None:
        if self._streaming_msg is None:
            return
        with self._stream_lock:
            buf = self._streaming_buf
        if not buf:
            return
        self._streaming_msg.update(buf)
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        # Follow if user is at or near the bottom
        if scroll.scroll_y >= scroll.max_scroll_y - 2:
            scroll.scroll_end(animate=False)

    def _tick_thinking(self) -> None:
        if self._streaming_msg is None:
            self._stop_thinking()
            return
        with self._stream_lock:
            has_buf = bool(self._streaming_buf)
        if has_buf:
            self._stop_thinking()
            return
        self._thinking_tick += 1
        dots = "." * (1 + (self._thinking_tick % 3))
        self._streaming_msg.update(f"*Thinking{dots}*")

    def _stop_thinking(self) -> None:
        if self._thinking_handle is not None:
            self._thinking_handle.stop()
            self._thinking_handle = None

    _TIMESTAMP_RE = re.compile(r"^(\[\d{2}:\d{2}\])(\s+)")

    def _render_transcript(self, query: str = "") -> None:
        log = self.query_one("#transcript-log", RichLog)
        log.clear()
        text = self._transcript_formatted or self._transcript
        if not text:
            return
        pattern = re.compile(re.escape(query), re.IGNORECASE) if query else None
        total_matches = 0
        paragraphs = text.split("\n\n")
        for i, paragraph in enumerate(paragraphs):
            rich_text = Text()
            ts_match = self._TIMESTAMP_RE.match(paragraph)
            if ts_match:
                rich_text.append(ts_match.group(1), style="bold cyan")
                rich_text.append(ts_match.group(2))
                body_start = ts_match.end()
            else:
                body_start = 0
            body = paragraph[body_start:]
            if pattern is None:
                rich_text.append(body)
            else:
                last = 0
                for m in pattern.finditer(body):
                    rich_text.append(body[last:m.start()])
                    rich_text.append(body[m.start():m.end()], style="reverse yellow")
                    last = m.end()
                    total_matches += 1
                rich_text.append(body[last:])
            log.write(rich_text)
            if i < len(paragraphs) - 1:
                log.write("")
        if pattern is not None:
            label = "match" if total_matches == 1 else "matches"
            self.query_one("#status-bar", Label).update(
                f"Status: {total_matches} {label} for '{query}'"
            )

    def _copy_to_clipboard(self, text: str, label: str) -> None:
        copied = False
        try:
            self.copy_to_clipboard(text)
            copied = True
        except AttributeError:
            try:
                b64 = base64.b64encode(text.encode()).decode()
                sys.stdout.write(f"\x1b]52;c;{b64}\x07")
                sys.stdout.flush()
                copied = True
            except Exception:
                copied = False
        msg = f"Status: {label} copied to clipboard" if copied else f"Status: Copy failed"
        self.query_one("#status-bar", Label).update(msg)

    @work(thread=True)
    def _save_to_file(self, kind: str, text: str) -> None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.expanduser(f"~/ytalk-{kind}-{ts}.txt")
        try:
            with open(path, "w") as f:
                f.write(text)
            self._set_status(f"Status: Saved to {path}")
        except Exception as e:
            self._set_status(f"Error saving: {e}")

    def _set_status(self, text: str) -> None:
        label = self.query_one("#status-bar", Label)
        self.call_from_thread(label.update, text)

    _SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def _start_status_spinner(self, base: str) -> None:
        """Show an animated working indicator in the global status bar until stopped.

        Call from a worker thread (uses call_from_thread). The base text is shown
        followed by a cycling spinner frame; use _set_status_spinner_base to change
        the text on a phase change, and _stop_status_spinner when done.
        """
        def _start():
            self._status_spinner_base = base
            self._status_spinner_tick = 0
            if self._status_spinner_handle is None:
                self._status_spinner_handle = self.set_interval(0.1, self._tick_status_spinner)
            self._tick_status_spinner()

        self.call_from_thread(_start)

    def _set_status_spinner_base(self, base: str) -> None:
        """Update the spinner's base text (e.g. on a phase change) while it keeps animating."""
        self.call_from_thread(setattr, self, "_status_spinner_base", base)

    def _tick_status_spinner(self) -> None:
        frame = self._SPINNER_FRAMES[self._status_spinner_tick % len(self._SPINNER_FRAMES)]
        self._status_spinner_tick += 1
        self.query_one("#status-bar", Label).update(f"Status: {self._status_spinner_base} {frame}")

    def _stop_status_spinner(self) -> None:
        """Stop the status-bar animation. Blocks until the timer is cancelled, so the
        next status message written after this call won't be overwritten by a tick."""
        def _stop():
            if self._status_spinner_handle is not None:
                self._status_spinner_handle.stop()
                self._status_spinner_handle = None

        self.call_from_thread(_stop)

    def _report_error(self, where: str, exc: BaseException) -> None:
        self._stop_status_spinner()
        logger.exception("%s failed", where)
        self._set_status(
            f"Error in {where}: {type(exc).__name__}: {exc}  (see {LOG_PATH})"
        )


def main():
    parser = argparse.ArgumentParser(description="YouTube → Transcribe → Summarize")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--ollama-model", default="gemma3:4b",
                        help="Ollama model for summarization (default: gemma3:4b)")
    parser.add_argument("--output", "-o", default=None,
                        help="Save output to file")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Download
        audio_path, _meta = download_audio(args.url, tmpdir)

        # Step 2: Transcribe
        transcript, _segments = transcribe(audio_path, args.whisper_model)
        os.remove(audio_path)

        print("\n" + "=" * 60)
        print("TRANSCRIPT")
        print("=" * 60)
        print(transcript[:2000] + ("..." if len(transcript) > 2000 else ""))

    # Step 3: Summarize
    summary = summarize(transcript, args.ollama_model)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write("TRANSCRIPT\n" + "=" * 60 + "\n")
            f.write(transcript + "\n\n")
            f.write("SUMMARY\n" + "=" * 60 + "\n")
            f.write(summary + "\n")
        print(f"\nSaved to {args.output}")


def entry_point():
    """Console script entry point."""
    try:
        if len(sys.argv) > 1:
            main()
        else:
            YTalkApp().run()
    except KeyboardInterrupt:
        os._exit(0)


if __name__ == "__main__":
    entry_point()
