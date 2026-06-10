"""Unit checks for ytalk.app._ctx_for (offline — no Ollama needed).

Usage:
    PYTHONPATH=src python tests/test_ctx_for.py
"""
from ytalk.app import _ctx_for


def main():
    # Short prompts use the cheap floor; never below 4096.
    assert _ctx_for("") == 4096
    assert _ctx_for("hello") == 4096
    assert _ctx_for("x" * 1000) == 4096

    # Grows in power-of-two buckets as the prompt gets longer.
    assert _ctx_for("x" * 30_000) == 16384   # ~this video's transcript
    assert _ctx_for("x" * 120_000) == 32768  # ~1hr video

    # Clamped to max_ctx for very long inputs.
    assert _ctx_for("x" * 5_000_000) == 32768

    # Result is always a valid window: >= floor, <= cap, monotonic non-decreasing.
    prev = 0
    for n in range(0, 200_000, 5_000):
        ctx = _ctx_for("x" * n)
        assert 4096 <= ctx <= 32768
        assert ctx >= prev
        prev = ctx

    print("PASS: _ctx_for sizing is correct")


if __name__ == "__main__":
    main()
