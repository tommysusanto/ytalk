"""Headless TUI test: run the full pipeline and check widget states.

Usage:
    python test_tui.py [youtube_url]

Default URL is a short video for quick testing.
"""
import asyncio
import sys
from ytalk.app import YTalkApp
from textual.widgets import Button, Label, TextArea, Input

DEFAULT_URL = "https://www.youtube.com/watch?v=DHc1U045o9E"


async def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    print(f"Testing with URL: {url}\n", flush=True)

    app = YTalkApp()
    async with app.run_test(size=(100, 30)) as pilot:
        # Enter URL
        url_input = app.query_one("#url-input", Input)
        url_input.value = url

        # Click Run
        await pilot.click("#run-btn")

        # Monitor progress
        send_btn = app.query_one("#send-btn", Button)
        prev_status = ""
        for i in range(600):  # up to 5 minutes
            await asyncio.sleep(0.5)

            status = str(app.query_one("#status-bar", Label).render())
            transcript_len = len(app.query_one("#transcript-area", TextArea).text)
            send_enabled = not send_btn.disabled

            if status != prev_status:
                print(f"[{i*0.5:5.0f}s] status={status}", flush=True)
                print(f"       transcript={transcript_len} send_enabled={send_enabled}", flush=True)
                prev_status = status

            if "Done" in status or "Error" in status:
                print(f"\n--- RESULT ---", flush=True)
                print(f"Transcript: {transcript_len} chars", flush=True)
                print(f"Send enabled: {send_enabled}", flush=True)
                break
        else:
            print("TIMEOUT - pipeline did not complete in 5 minutes", flush=True)

        app.exit()


if __name__ == "__main__":
    asyncio.run(main())
