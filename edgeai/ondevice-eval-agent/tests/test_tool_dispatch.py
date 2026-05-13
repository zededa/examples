"""Tests for tools.registry.dispatch_tool_calls (parallel tool execution)."""

import time

import pytest

from tools.registry import dispatch_tool_calls, register_tool


@pytest.fixture
def register_sleep_tool():
    """Register a fake slow tool so we can measure wall-clock parallel vs serial."""

    def _sleep_tool(ms: int = 100, label: str = ""):
        time.sleep(ms / 1000.0)
        return {"success": True, "ms": ms, "label": label}

    register_tool(
        name="_test_sleep",
        func=_sleep_tool,
        description="sleep for ms milliseconds then return",
        input_schema={
            "type": "object",
            "properties": {
                "ms": {"type": "integer"},
                "label": {"type": "string"},
            },
            "required": [],
        },
    )
    yield "_test_sleep"


def test_parallel_is_faster_than_serial(register_sleep_tool):
    """Four 150ms sleeps should take < 2x the single-tool duration when parallel."""
    calls = [
        {"id": f"t{i}", "name": register_sleep_tool, "input": {"ms": 150, "label": f"t{i}"}}
        for i in range(4)
    ]

    t0 = time.perf_counter()
    serial = dispatch_tool_calls(calls, parallel=False)
    serial_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel = dispatch_tool_calls(calls, parallel=True, max_workers=4)
    parallel_elapsed = time.perf_counter() - t0

    assert len(serial) == 4 and len(parallel) == 4
    # Serial should be ~600ms, parallel ~150-200ms. Give the CI generous margin.
    assert parallel_elapsed < serial_elapsed * 0.75, (
        f"parallel {parallel_elapsed:.3f}s should be much faster than "
        f"serial {serial_elapsed:.3f}s"
    )


def test_order_preserved(register_sleep_tool):
    """Results come back in the same order as the input tool_calls list."""
    # Inputs with decreasing sleep times so the first one completes last.
    calls = [
        {"id": "a", "name": register_sleep_tool, "input": {"ms": 200, "label": "a"}},
        {"id": "b", "name": register_sleep_tool, "input": {"ms": 100, "label": "b"}},
        {"id": "c", "name": register_sleep_tool, "input": {"ms": 50, "label": "c"}},
    ]
    results = dispatch_tool_calls(calls, parallel=True, max_workers=4)
    assert [r["id"] for r in results] == ["a", "b", "c"]
    assert [r["result"]["label"] for r in results] == ["a", "b", "c"]


def test_empty_list_short_circuits():
    assert dispatch_tool_calls([]) == []


def test_single_call_no_thread_pool(register_sleep_tool):
    """A single tool_call skips the pool entirely (serial path)."""
    results = dispatch_tool_calls(
        [{"id": "only", "name": register_sleep_tool, "input": {"ms": 10}}],
        parallel=True,
    )
    assert len(results) == 1
    assert results[0]["result"]["success"] is True


def test_unknown_tool_does_not_crash_batch(register_sleep_tool):
    """An unknown tool in the batch returns a structured error; other tools still run."""
    results = dispatch_tool_calls(
        [
            {"id": "ok", "name": register_sleep_tool, "input": {"ms": 10, "label": "ok"}},
            {"id": "bad", "name": "does_not_exist", "input": {}},
        ],
        parallel=True,
    )
    by_id = {r["id"]: r for r in results}
    assert by_id["ok"]["result"]["success"] is True
    assert by_id["bad"]["result"]["success"] is False
