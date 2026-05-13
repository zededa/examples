"""
Tests for webapp/mcp/registry.py and webapp/mcp/base.py.

Covers ToolResult, ok(), error_response(), register_tool(), and execute_tool().
"""

import pytest

from tools.base import ToolResult, ok, error_response
from tools.registry import TOOL_SCHEMAS, TOOL_FUNCTIONS, register_tool, execute_tool


# ============================================================================
# Helpers
# ============================================================================

_TEST_TOOL_NAME = "__test_tool_registry_xyz__"


@pytest.fixture(autouse=True)
def _cleanup_test_tool():
    """Remove the temporary test tool after each test."""
    yield
    TOOL_FUNCTIONS.pop(_TEST_TOOL_NAME, None)
    TOOL_SCHEMAS[:] = [s for s in TOOL_SCHEMAS if s["name"] != _TEST_TOOL_NAME]


# ============================================================================
# ToolResult
# ============================================================================


class TestToolResult:
    def test_success_to_dict_has_success_true(self):
        result = ToolResult(success=True, payload={"models": ["a"]})
        d = result.to_dict()
        assert d["success"] is True

    def test_error_to_dict_has_success_false_and_error(self):
        result = ToolResult(success=False, error="something broke")
        d = result.to_dict()
        assert d["success"] is False
        assert "error" in d
        assert d["error"] == "something broke"


# ============================================================================
# ok() / error_response()
# ============================================================================


class TestOkAndErrorResponse:
    def test_ok_returns_success_true(self):
        d = ok()
        assert d["success"] is True

    def test_ok_with_payload(self):
        d = ok(models=["a", "b"])
        assert d["success"] is True
        assert d["models"] == ["a", "b"]

    def test_error_response_returns_success_false(self):
        d = error_response(ValueError("oops"))
        assert d["success"] is False
        assert "oops" in d["error"]

    def test_error_response_includes_context_kwargs(self):
        d = error_response(RuntimeError("fail"), model_name="resnet", operation="test")
        assert d["context"]["model_name"] == "resnet"
        assert d["context"]["operation"] == "test"


# ============================================================================
# register_tool()
# ============================================================================


class TestRegisterTool:
    def test_register_adds_to_schemas_and_functions(self):
        register_tool(
            name=_TEST_TOOL_NAME,
            func=lambda: ok(msg="hi"),
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
        )
        assert _TEST_TOOL_NAME in TOOL_FUNCTIONS
        assert any(s["name"] == _TEST_TOOL_NAME for s in TOOL_SCHEMAS)

    def test_register_duplicate_updates_in_place(self):
        register_tool(
            name=_TEST_TOOL_NAME,
            func=lambda: ok(msg="v1"),
            description="Version 1",
            input_schema={"type": "object", "properties": {}},
        )
        count_before = len(TOOL_SCHEMAS)

        register_tool(
            name=_TEST_TOOL_NAME,
            func=lambda: ok(msg="v2"),
            description="Version 2",
            input_schema={"type": "object", "properties": {}},
        )
        count_after = len(TOOL_SCHEMAS)

        assert count_after == count_before
        schema = next(s for s in TOOL_SCHEMAS if s["name"] == _TEST_TOOL_NAME)
        assert schema["description"] == "Version 2"


# ============================================================================
# execute_tool()
# ============================================================================


class TestExecuteTool:
    def test_execute_success(self):
        register_tool(
            name=_TEST_TOOL_NAME,
            func=lambda: ok(answer=42),
            description="Returns 42",
            input_schema={"type": "object", "properties": {}},
        )
        result = execute_tool(_TEST_TOOL_NAME, {})
        assert result["success"] is True
        assert result["answer"] == 42

    def test_execute_unknown_tool(self):
        result = execute_tool("__nonexistent_tool__", {})
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    def test_execute_bad_args_returns_error(self):
        def needs_arg(x):
            return ok(val=x)

        register_tool(
            name=_TEST_TOOL_NAME,
            func=needs_arg,
            description="needs x",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        # Missing required argument 'x' -> TypeError
        result = execute_tool(_TEST_TOOL_NAME, {})
        assert result["success"] is False

    def test_execute_func_exception_returns_error(self):
        def exploding():
            raise RuntimeError("boom")

        register_tool(
            name=_TEST_TOOL_NAME,
            func=exploding,
            description="always fails",
            input_schema={"type": "object", "properties": {}},
        )
        result = execute_tool(_TEST_TOOL_NAME, {})
        assert result["success"] is False
        assert "boom" in result["error"]
