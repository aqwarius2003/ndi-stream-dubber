from dubber.sink import ndi_sink


def test_recurring_jitter_is_absorbed_after_learning():
    buf = ndi_sink.PlayoutBuffer(1000, 1)
    # 200 ms chunks, every fourth one arrives 140 ms late. The next one catches up.
    arrivals = {i * 200 + (140 if i % 4 == 2 else 0) for i in range(28)}
    after_warmup = []
    for ms in range(0, 5500, 20):
        if ms in arrivals:
            buf.push(b"\x01\x00" * 200, ms / 1000, "gemini")
        frame = buf.pull(ms / 1000)
        if ms >= 1600:
            after_warmup.append(frame)
    assert 140 <= buf.target_ms <= 320
    assert all(frame == b"\x01\x00" * 20 for frame in after_warmup)
    assert buf.dropped_samples == 0


def test_silent_pcm_and_empty_queue_have_separate_counters():
    buf = ndi_sink.PlayoutBuffer(1000, 1)
    buf.pull(0)  # Waiting for the first provider packet.
    buf.push(bytes(200), 1, "gemini")
    for i in range(12):
        buf.pull(1 + i * .02)
    assert buf.provider_zero_samples == 100
    assert buf.local_silence_samples == 140
    buf.push(b"\x01\x00" * 200, 12, "gemini")
    assert buf.target_ms == 60  # A long idle period is not packet jitter.
    buf.push(b"\x02\x00" * 200, 12.01, "soniox")
    assert buf.pull(12.09) == b"\x02\x00" * 20


def test_reserve_is_bounded_then_recovers_after_stable_delivery():
    buf = ndi_sink.PlayoutBuffer(1000, 1)
    block = bytes(200)
    for t in [0, .6, 1.2, 1.8]:
        buf.push(block, t, "gemini")
    assert buf.target_ms == 320
    for i in range(70):
        buf.push(block, 1.9 + i * .1, "gemini")
    assert buf.target_ms == 60
