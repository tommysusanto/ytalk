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
import os
import re
import sys
import tempfile
import threading
from datetime import datetime

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
        "progress_hooks": [_progress_hook],
    }
    print(f"Downloading audio from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        ydl.download([url])

    meta = {
        "title": info.get("title") or "Unknown title",
        "channel": info.get("uploader") or info.get("channel") or "Unknown channel",
        "duration": int(info.get("duration") or 0),
        "url": url,
    }

    audio_file = os.path.join(output_dir, "audio.mp3")
    if not os.path.exists(audio_file):
        for f in os.listdir(output_dir):
            if f.startswith("audio."):
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


def format_transcript_segments(
    segments: list[dict],
    pause_threshold: float = 1.5,
    max_paragraph_seconds: float = 30.0,
) -> str:
    """Group Whisper segments into paragraphs on >pause_threshold pauses or every
    ~max_paragraph_seconds. Each paragraph is prefixed with [mm:ss] markup."""
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
        mm, ss = divmod(int(start), 60)
        lines.append(f"[{mm:02d}:{ss:02d}] {text}")
    return "\n\n".join(lines)


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
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": True},
        timeout=300,
        stream=True,
    )
    resp.raise_for_status()
    import json as _json
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
            "options": {"num_predict": -1},
        },
        timeout=300,
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
    #summary-md {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    """

    TITLE = "yTalk"

    _transcript: str = ""
    _summary: str = ""

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
                yield Markdown("", id="summary-md")
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
        tabs = self.query_one(TabbedContent)
        for tid in ("tab-transcript", "tab-chat", "tab-summary"):
            tabs.get_tab(tid).tooltip = "Run the pipeline first"
        self._load_ollama_models()

    @work(thread=True)
    def _load_ollama_models(self) -> None:
        import shutil

        warnings = []

        if not shutil.which("yt-dlp"):
            warnings.append("yt-dlp not found. Install: brew install yt-dlp")

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

        self.call_from_thread(setattr, btn, "disabled", True)
        self.call_from_thread(setattr, send_btn, "disabled", True)
        self.call_from_thread(setattr, summarize_btn, "disabled", True)
        self.call_from_thread(setattr, transcript_copy, "disabled", True)
        self.call_from_thread(setattr, transcript_save, "disabled", True)
        self.call_from_thread(setattr, summary_copy, "disabled", True)
        self.call_from_thread(setattr, summary_save, "disabled", True)

        # Reset state
        self._transcript = ""
        self._summary = ""
        self._chat_history = []

        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)

        def _clear_chat():
            for child in list(chat_scroll.children):
                child.remove()

        self.call_from_thread(_clear_chat)
        self.call_from_thread(self.query_one("#summary-md", Markdown).update, "")
        self.call_from_thread(self.query_one("#video-meta", Static).update, "Loading…")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                self._set_status("Status: Downloading audio...")
                audio_path, meta = download_audio(url, tmpdir, progress_callback=self._set_status)

                mins, secs = divmod(meta["duration"], 60)
                meta_text = (
                    f"[b]{rich_escape(meta['title'])}[/b]\n"
                    f"{rich_escape(meta['channel'])}  •  {mins}:{secs:02d}"
                )
                self.call_from_thread(self.query_one("#video-meta", Static).update, meta_text)

                self._set_status("Status: Transcribing with Whisper...")
                transcript, segments = transcribe(audio_path, whisper_model, progress_callback=self._set_status)
                os.remove(audio_path)

                self._transcript = transcript
                self._transcript_segments = segments
                self._transcript_formatted = format_transcript_segments(segments) or transcript
                self.call_from_thread(self._render_transcript, "")

                self.call_from_thread(setattr, summarize_btn, "disabled", False)
                self.call_from_thread(setattr, transcript_copy, "disabled", False)
                self.call_from_thread(setattr, transcript_save, "disabled", False)

            self._set_status("Status: Done! Review the transcript.")

            def _open_workspace():
                tabs = self.query_one(TabbedContent)
                for tid, label in (
                    ("tab-transcript", "Transcript"),
                    ("tab-chat", "Chat"),
                    ("tab-summary", "Summary"),
                ):
                    tabs.enable_tab(tid)
                    tab = tabs.get_tab(tid)
                    tab.label = label
                    tab.tooltip = None
                tabs.active = "tab-transcript"

            self.call_from_thread(_open_workspace)
        except Exception as e:
            self._set_status(f"Error: {e}")
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
                    self._set_status(f"Status: {chat_model} is thinking…")
                else:
                    self._set_status(f"Status: {chat_model} is generating summary…")

            result = summarize(
                self._transcript, chat_model,
                progress_callback=_on_summary_token,
                status_callback=_on_summary_status,
            )
            self._summary = result.strip()
            self.call_from_thread(self._stop_summary_thinking)
            self.call_from_thread(self._flush_summary)
            self.call_from_thread(summary_md.update, self._summary)
            self._set_status("Status: Summary ready!")
        except Exception as e:
            self._set_status(f"Error: {e}")
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
        self._set_status("Status: Thinking…")

        chat_model = self.query_one("#chat-select", Select).value
        if chat_model is Select.BLANK:
            chat_model = "gemma3:4b"

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
            self._set_status("Status: Done! Ask another question.")
        except Exception as e:
            self._set_status(f"Error: {e}")
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
