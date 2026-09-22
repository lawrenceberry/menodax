from benchmarks.benchmark_common import (
    TIMEOUT_ERROR,
    format_cached_timing,
    is_timeout,
    timeout_cache_entry,
    timing_value_or_none,
)


def test_timeout_cache_helpers():
    entry = timeout_cache_entry()
    assert entry == {"status": "timeout", "error": TIMEOUT_ERROR}
    assert is_timeout(entry)
    assert format_cached_timing(entry) == TIMEOUT_ERROR
    assert format_cached_timing(None) == "FAILED"
    assert format_cached_timing(3.25) == "3.2 ms"
    assert timing_value_or_none(entry) is None
    assert timing_value_or_none(None) is None
    assert timing_value_or_none(3.25) == 3.25
