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
import os
import sys
import tempfile
import time

import requests
import whisper
import yt_dlp

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    TextArea,
)


def download_audio(url: str, output_dir: str, progress_callback=None) -> str:
    """Download audio from YouTube using yt-dlp Python API."""
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
        ydl.download([url])

    audio_file = os.path.join(output_dir, "audio.mp3")
    if not os.path.exists(audio_file):
        for f in os.listdir(output_dir):
            if f.startswith("audio."):
                audio_file = os.path.join(output_dir, f)
                break
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"No audio file found in {output_dir}")
    print(f"Audio saved to: {audio_file}")
    return audio_file


def transcribe(audio_path: str, whisper_model: str = "base", progress_callback=None) -> str:
    """Transcribe audio using OpenAI Whisper."""
    # Prevent tqdm from creating a multiprocessing RLock, which spawns a
    # resource-tracker subprocess and fails with "bad value(s) in fds_to_keep"
    # when called from a Textual worker thread.
    import threading
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
    print(f"Transcription complete ({len(text)} chars).")
    return text


def summarize(text: str, model: str = "gemma3:4b", progress_callback=None, status_callback=None) -> str:
    """Summarize text using Ollama's API."""
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
    """Answer a question about the transcript using Ollama's chat API."""
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
    token_count = 0
    for line in resp.iter_lines():
        if not line:
            continue
        obj = _json.loads(line)
        token = obj.get("message", {}).get("content", "")
        if token:
            accumulated += token
            token_count += 1
            if progress_callback:
                progress_callback(token_count, "generating")
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


def _markdown_to_rich_markup(text: str) -> str:
    """Convert markdown text to Rich console markup for display in RichLog."""
    import re
    lines = text.splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        # Headers
        if stripped.startswith("### "):
            out.append(f"  [bold cyan]{stripped[4:]}[/bold cyan]")
        elif stripped.startswith("## "):
            out.append(f"\n[bold yellow]{stripped[3:]}[/bold yellow]")
        elif stripped.startswith("# "):
            out.append(f"\n[bold magenta]{stripped[2:]}[/bold magenta]")
        # Bullet points (-, *, •)
        elif re.match(r"^\s*[-*•]\s", line):
            indent = len(line) - len(line.lstrip())
            content = re.sub(r"^[-*•]\s+", "", stripped)
            content = re.sub(r"\*\*(.+?)\*\*", r"[bold]\1[/bold]", content)
            prefix = "  " * (indent // 2) + "  •"
            out.append(f"{prefix} {content}")
        elif stripped == "---" or stripped == "***":
            continue
        else:
            converted = re.sub(r"\*\*(.+?)\*\*", r"[bold]\1[/bold]", stripped)
            out.append(converted)
    return "\n".join(out)


WHISPER_MODELS = ["tiny", "base", "small", "base", "large"]


class YTalkApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #form {
        height: auto;
        padding: 1 2;
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
    #output-area {
        padding: 0 2 1 2;
        max-height: 25%;
    }
    .output-label {
        margin-top: 1;
        text-style: bold;
    }
    TextArea {
        height: auto;
        max-height: 12;
    }
    #chat-area {
        height: 3fr;
        padding: 0 2;
    }
    #chat-log {
        height: 1fr;
        border: solid $primary;
    }
    #chat-input-row {
        height: 3;
        margin-top: 1;
    }
    #chat-input {
        width: 1fr;
    }
    #summarize-btn {
        background: transparent;
        border: none;
        text-style: underline;
        color: $text-muted;
        height: 1;
        min-width: 12;
        padding: 0;
    }
    #summarize-btn:hover {
        color: $accent;
        background: transparent;
    }
    #summarize-btn:focus {
        background: transparent;
    }
    #summarize-btn.hidden {
        display: none;
    }
    #send-btn {
        width: 12;
    }
    """

    TITLE = "yTalk"

    _transcript: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
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
        yield Label("Status: Idle", id="status-bar")
        with VerticalScroll(id="output-area"):
            with Collapsible(title="Transcript", id="transcript-section", collapsed=True):
                yield TextArea(read_only=True, id="transcript-area")
        with Vertical(id="chat-area"):
            yield Button("Summarize", id="summarize-btn", classes="hidden")
            yield RichLog(id="chat-log", wrap=True, markup=True)
            with Horizontal(id="chat-input-row"):
                yield Input(placeholder="Ask about the video...", id="chat-input")
                yield Button("Send", id="send-btn", variant="primary", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self._chat_history: list[dict] = []
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
        if event.button.id == "run-btn":
            self._run_pipeline()
        elif event.button.id == "send-btn":
            self._send_chat()
        elif event.button.id == "summarize-btn":
            if self._transcript:
                self._run_summarize()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input" and not self.query_one("#send-btn", Button).disabled:
            self._send_chat()

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
        self.call_from_thread(setattr, btn, "disabled", True)
        self.call_from_thread(setattr, send_btn, "disabled", True)

        # Reset chat state for new pipeline run
        self._transcript = ""
        self._chat_history = []
        chat_log = self.query_one("#chat-log", RichLog)
        self.call_from_thread(chat_log.clear)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Download
                self._set_status("Status: Downloading audio...")
                audio_path = download_audio(url, tmpdir, progress_callback=self._set_status)

                # Transcribe
                self._set_status("Status: Transcribing with Whisper...")
                transcript = transcribe(audio_path, whisper_model, progress_callback=self._set_status)

                # Remove the downloaded audio file to free disk space
                os.remove(audio_path)

                self._transcript = transcript
                self._set_text("#transcript-area", transcript)
                self.call_from_thread(setattr, send_btn, "disabled", False)
                summarize_link = self.query_one("#summarize-btn", Button)
                self.call_from_thread(summarize_link.remove_class, "hidden")

            self._set_status("Status: Done! You can now chat about the video.")
        except Exception as e:
            self._set_status(f"Error: {e}")
        finally:
            self.call_from_thread(setattr, btn, "disabled", False)
            if self._transcript:
                self.call_from_thread(setattr, send_btn, "disabled", False)

    @work(thread=True)
    def _run_summarize(self) -> None:
        send_btn = self.query_one("#send-btn", Button)
        self.call_from_thread(setattr, send_btn, "disabled", True)
        summarize_link = self.query_one("#summarize-btn", Button)
        self.call_from_thread(summarize_link.add_class, "hidden")

        chat_model = self.query_one("#chat-select", Select).value
        if chat_model is Select.BLANK:
            chat_model = "gemma3:4b"

        chat_log = self.query_one("#chat-log", RichLog)
        self.call_from_thread(chat_log.write, "[dim]Generating summary...[/dim]")

        try:
            def _on_summary_token(text):
                pass
            def _on_summary_status(phase):
                if phase == "thinking":
                    self._set_status(f"Status: {chat_model} is thinking...")
                else:
                    self._set_status(f"Status: {chat_model} is generating summary...")
            result = summarize(
                self._transcript, chat_model,
                progress_callback=_on_summary_token,
                status_callback=_on_summary_status,
            )
            formatted = _markdown_to_rich_markup(result)
            self.call_from_thread(chat_log.write, "")
            self.call_from_thread(chat_log.write, "[bold green]───── Summary ─────[/bold green]")
            self.call_from_thread(chat_log.write, "")
            for line in formatted.splitlines():
                self.call_from_thread(chat_log.write, line)
            self.call_from_thread(chat_log.write, "")
            self.call_from_thread(chat_log.write, "[bold green]───────────────────[/bold green]")
            self._set_status("Status: Done!")
        except Exception as e:
            self._set_status(f"Error: {e}")
        finally:
            self.call_from_thread(summarize_link.remove_class, "hidden")
            self.call_from_thread(setattr, send_btn, "disabled", False)

    def _send_chat(self) -> None:
        chat_input = self.query_one("#chat-input", Input)
        question = chat_input.value.strip()
        if not question:
            return
        chat_input.value = ""
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(f"[bold cyan]You:[/bold cyan] {question}")
        self._chat_history.append({"role": "user", "content": question})
        self._run_chat_query(question)

    @work(thread=True)
    def _run_chat_query(self, question: str) -> None:
        send_btn = self.query_one("#send-btn", Button)
        self.call_from_thread(setattr, send_btn, "disabled", True)
        self._set_status("Status: Thinking...")

        chat_model = self.query_one("#chat-select", Select).value
        if chat_model is Select.BLANK:
            chat_model = "gemma3:4b"

        chat_log = self.query_one("#chat-log", RichLog)
        self.call_from_thread(chat_log.write, "[dim]Generating response...[/dim]")

        try:
            last_chat_update = [0.0]
            def _on_chat_token(n, phase):
                now = time.monotonic()
                if now - last_chat_update[0] >= 0.15:
                    last_chat_update[0] = now
                    self._set_status(f"Status: Generating response... ({n} tokens)")
            answer = chat_query(
                self._transcript, question, self._chat_history[:-1], chat_model,
                progress_callback=_on_chat_token,
            )
            self._chat_history.append({"role": "assistant", "content": answer})
            self.call_from_thread(chat_log.write, f"[bold green]AI:[/bold green] {answer}")
            self._set_status("Status: Done! You can now chat about the video.")
        except Exception as e:
            self._set_status(f"Error: {e}")
        finally:
            self.call_from_thread(setattr, send_btn, "disabled", False)

    def _set_status(self, text: str) -> None:
        label = self.query_one("#status-bar", Label)
        self.call_from_thread(label.update, text)

    def _set_text(self, selector: str, text: str) -> None:
        area = self.query_one(selector, TextArea)
        self.call_from_thread(setattr, area, "text", text)


def main():
    parser = argparse.ArgumentParser(description="YouTube → Transcribe → Summarize")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "base", "large"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--ollama-model", default="gemma3:4b",
                        help="Ollama model for summarization (default: gemma3:4b)")
    parser.add_argument("--output", "-o", default=None,
                        help="Save output to file")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Download
        audio_path = download_audio(args.url, tmpdir)

        # Step 2: Transcribe
        transcript = transcribe(audio_path, args.whisper_model)
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
