"""Unit checks for ytalk.app context sizing (offline — no Ollama needed).

Usage:
    PYTHONPATH=src python tests/test_ctx_for.py
"""
from ytalk import app
from ytalk.app import _ctx_for, _fit_prompt, _memory_capped_ctx, _memory_warning


def test_prompt_sizing():
    """With no model given, the window is sized on prompt length alone."""
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


def _fake_machine(budget_gb, weight_gb, kv_per_token):
    """Pin the memory probes so sizing is testable without a GPU or Ollama."""
    app._mem_profile_cache.clear()
    app._gpu_memory_budget_bytes = lambda: int(budget_gb * 1024**3)
    app._model_memory_profile = lambda m: (int(weight_gb * 1024**3), kv_per_token)


def test_memory_cap():
    """A model that can't afford a big KV cache gets a smaller window."""
    original = (app._gpu_memory_budget_bytes, app._model_memory_profile)
    try:
        # The real crash case: qwen3-coder:30b (17.3GB of weights, 96KiB/token of KV)
        # on a 24GB Mac, whose GPU budget is ~18GB. A 1hr transcript used to ask for
        # 32768 tokens = 3GB of KV on top of the weights, overshooting the budget into
        # host RAM and locking the machine. It must now be capped hard.
        _fake_machine(budget_gb=18, weight_gb=17.3, kv_per_token=96 * 1024)
        assert _memory_capped_ctx("qwen3-coder:30b") == 4096
        assert _ctx_for("x" * 120_000, model="qwen3-coder:30b") == 4096
        assert _memory_warning("qwen3-coder:30b") is not None

        # A model with room to spare is left alone -- the cap must not punish
        # small models on the same machine.
        _fake_machine(budget_gb=18, weight_gb=3.1, kv_per_token=136 * 1024)
        assert _ctx_for("x" * 120_000, model="gemma3:4b") == 32768
        assert _memory_warning("gemma3:4b") is None

        # Whatever the cap lands on, weights + KV must stay inside the budget.
        for weight_gb, kv_kib in [(17.3, 96), (8.3, 270), (9.0, 48), (3.1, 136)]:
            _fake_machine(budget_gb=18, weight_gb=weight_gb, kv_per_token=kv_kib * 1024)
            ctx = _memory_capped_ctx("m")
            used = weight_gb * 1024**3 + ctx * kv_kib * 1024
            assert ctx == 4096 or used <= 18 * 1024**3, (weight_gb, kv_kib, ctx)

        # An undeterminable budget must fall back to prompt-only sizing, not to the floor.
        app._gpu_memory_budget_bytes = lambda: None
        assert _ctx_for("x" * 120_000, model="anything") == 32768
    finally:
        app._gpu_memory_budget_bytes, app._model_memory_profile = original
        app._mem_profile_cache.clear()


def test_fit_prompt():
    """Trimming keeps the head and tail, so the task framing survives."""
    # A prompt that already fits is returned untouched.
    assert _fit_prompt("short", 4096) == "short"

    head, tail = "INSTRUCTIONS-AT-THE-TOP", "SUMMARY:"
    prompt = head + "x" * 500_000 + tail
    fitted = _fit_prompt(prompt, 4096)

    assert len(fitted) < len(prompt)
    assert fitted.startswith(head)      # instructions preserved
    assert fitted.endswith(tail)        # the answer cue preserved
    assert "trimmed" in fitted          # and the cut is visible to the model
    # Comfortably inside the window at the ~3 chars/token estimate.
    assert len(fitted) // 3 < 4096


def main():
    test_prompt_sizing()
    test_memory_cap()
    test_fit_prompt()
    print("PASS: context sizing, memory cap, and prompt fitting are correct")


if __name__ == "__main__":
    main()
