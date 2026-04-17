"""S5 工具系统验证测试。"""

from pathlib import Path
from typing import Any

import pytest

from praxis.config.schemas import ToolsConfig
from praxis.exceptions import (
    SandboxViolationError,
    ToolError,
    ToolNotFoundError,
    ToolTimeoutError,
)
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.builtins.registration import register_builtins
from praxis.tools.executor import ToolExecutor, validate_arguments
from praxis.tools.override import override_tool
from praxis.tools.registry import ToolRegistry
from praxis.tools.sandbox import Sandbox


def make_sandbox(tmp_path: Path) -> Sandbox:
    config = ToolsConfig(
        allowed_paths=[str(tmp_path)],
        default_timeout=5.0,
        shell_timeout=10.0,
    )
    return Sandbox(config)


def make_sandbox_open() -> Sandbox:
    """无路径限制的沙箱（测试用）。"""
    return Sandbox(ToolsConfig())


# ── Task 5.1: 工具注册表 ─────────────────────────────────────────────────────


class TestToolRegistry:
    """Task 5.1: 工具注册表验证。"""

    def test_register_and_list(self) -> None:
        registry = ToolRegistry()
        for name in ["tool_a", "tool_b", "tool_c"]:
            defn = ToolDefinition(
                name=name,
                description=f"{name} description",
                parameters={"type": "object", "properties": {}},
            )

            async def handler(args: dict[str, Any]) -> str:
                return "ok"

            registry.register(defn, handler)

        assert len(registry.list_tools()) == 3
        assert registry.has_tool("tool_a")
        assert not registry.has_tool("tool_x")

    def test_get_tool_schemas(self) -> None:
        registry = ToolRegistry()
        for name in ["tool_a", "tool_b", "tool_c"]:
            defn = ToolDefinition(
                name=name,
                description=f"{name} desc",
                parameters={"type": "object", "properties": {"x": {"type": "string"}}},
                metadata=ToolMetadata(category="test"),
            )

            async def handler(args: dict[str, Any]) -> str:
                return "ok"

            registry.register(defn, handler)

        schemas = registry.get_tool_schemas()
        assert len(schemas) == 3
        assert all(s["type"] == "function" for s in schemas)
        assert schemas[0]["function"]["name"] == "tool_a"

    def test_get_tool_schemas_filtered(self) -> None:
        registry = ToolRegistry()
        for name, cat in [("a", "file_ops"), ("b", "search"), ("c", "file_ops")]:
            defn = ToolDefinition(
                name=name,
                description="desc",
                parameters={"type": "object", "properties": {}},
                metadata=ToolMetadata(category=cat),
            )

            async def handler(args: dict[str, Any]) -> str:
                return "ok"

            registry.register(defn, handler)

        schemas = registry.get_tool_schemas(category="file_ops")
        assert len(schemas) == 2

    def test_unregister(self) -> None:
        registry = ToolRegistry()
        defn = ToolDefinition(
            name="tmp",
            description="tmp",
            parameters={"type": "object", "properties": {}},
        )

        async def handler(args: dict[str, Any]) -> str:
            return "ok"

        registry.register(defn, handler)
        assert registry.unregister("tmp") is True
        assert registry.unregister("tmp") is False
        assert not registry.has_tool("tmp")

    def test_get_entry_not_found_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError, match="未注册"):
            registry.get_entry("nonexistent")

    def test_get_metadata(self) -> None:
        registry = ToolRegistry()
        defn = ToolDefinition(
            name="t",
            description="d",
            parameters={"type": "object", "properties": {}},
            metadata=ToolMetadata(category="search", readonly=True),
        )

        async def handler(args: dict[str, Any]) -> str:
            return "ok"

        registry.register(defn, handler)
        meta = registry.get_metadata("t")
        assert meta.category == "search"
        assert meta.readonly is True


# ── Task 5.2: 内置文件操作工具 ───────────────────────────────────────────────


class TestFileOps:
    """Task 5.2: 内置文件操作工具验证。"""

    async def test_read_file(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        from praxis.tools.builtins.file_ops.read_file import create_handler

        handler = create_handler(sandbox)
        result = await handler({"file_path": str(test_file)})
        assert "line1" in result
        assert "line2" in result

    async def test_read_file_with_offset(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("a\nb\nc\nd\ne\n")

        from praxis.tools.builtins.file_ops.read_file import create_handler

        handler = create_handler(sandbox)
        result = await handler({"file_path": str(test_file), "offset": 2, "limit": 2})
        assert "b" in result
        assert "c" in result
        assert "a" not in result

    async def test_write_file(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        test_file = tmp_path / "subdir" / "out.txt"

        from praxis.tools.builtins.file_ops.write_file import create_handler

        handler = create_handler(sandbox)
        result = await handler({"file_path": str(test_file), "content": "hello"})
        assert "已写入" in result
        assert test_file.read_text() == "hello"

    async def test_edit_file(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        test_file = tmp_path / "edit.txt"
        test_file.write_text("foo bar baz")

        from praxis.tools.builtins.file_ops.edit_file import create_handler

        handler = create_handler(sandbox)
        result = await handler({
            "file_path": str(test_file),
            "old_string": "bar",
            "new_string": "qux",
        })
        assert "已替换" in result
        assert test_file.read_text() == "foo qux baz"

    async def test_list_dir(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "sub").mkdir()

        from praxis.tools.builtins.file_ops.list_dir import create_handler

        handler = create_handler(sandbox)
        result = await handler({"dir_path": str(tmp_path)})
        assert "a.txt" in result
        assert "b.py" in result
        assert "sub/" in result


# ── Task 5.3: 内置搜索工具 ──────────────────────────────────────────────────


class TestSearchTools:
    """Task 5.3: 内置搜索工具验证。"""

    async def test_grep_search(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        (tmp_path / "a.py").write_text("def hello():\n    pass\n")
        (tmp_path / "b.py").write_text("x = 1\ny = 2\n")

        from praxis.tools.builtins.search.grep_search import create_handler

        handler = create_handler(sandbox)
        result = await handler({"pattern": "def hello", "search_path": str(tmp_path)})
        assert "def hello" in result
        assert "a.py" in result

    async def test_find_by_name(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        (tmp_path / "test_a.py").write_text("")
        (tmp_path / "test_b.py").write_text("")
        (tmp_path / "other.txt").write_text("")

        from praxis.tools.builtins.search.find_by_name import create_handler

        handler = create_handler(sandbox)
        result = await handler({"pattern": "test_*.py", "search_path": str(tmp_path)})
        assert "test_a.py" in result
        assert "test_b.py" in result
        assert "other.txt" not in result

    async def test_code_search(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        (tmp_path / "mod.py").write_text("class MyClass:\n    pass\n\ndef my_func():\n    pass\n")

        from praxis.tools.builtins.search.code_search import create_handler

        handler = create_handler(sandbox)
        result = await handler({"query": "MyClass", "search_path": str(tmp_path)})
        assert "class MyClass" in result


# ── Task 5.4: Shell/系统/自治管理工具 ────────────────────────────────────────


class TestShellAndSystem:
    """Task 5.4: Shell 和系统工具验证。"""

    async def test_run_command(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)

        from praxis.tools.builtins.shell.run_command import create_handler

        handler = create_handler(sandbox)
        result = await handler({"command": "echo hello"})
        assert "hello" in result
        assert "Exit code: 0" in result

    async def test_get_system_info(self) -> None:
        from praxis.tools.builtins.system.get_system_info import handle

        result = await handle({})
        assert "OS:" in result
        assert "Python:" in result

    async def test_ask_user(self) -> None:
        from praxis.tools.builtins.autonomy.ask_user import handle

        result = await handle({"question": "你确定吗？"})
        assert "ASK_USER" in result
        assert "你确定吗" in result

    async def test_attempt_completion(self) -> None:
        from praxis.tools.builtins.autonomy.attempt_completion import handle

        result = await handle({"result": "任务完成"})
        assert "ATTEMPT_COMPLETION" in result
        assert "任务完成" in result


# ── Task 5.5: 工具执行管线 ──────────────────────────────────────────────────


class TestExecutor:
    """Task 5.5: 工具执行管线验证。"""

    async def test_execute_success(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        registry = ToolRegistry()
        (tmp_path / "hello.txt").write_text("hello world")

        register_builtins(registry, sandbox)
        executor = ToolExecutor(registry, sandbox)

        result = await executor.execute(
            "read_file",
            {"file_path": str(tmp_path / "hello.txt")},
            tool_call_id="call-1",
        )
        assert result.success is True
        assert "hello world" in result.content
        assert result.tool_call_id == "call-1"
        assert result.execution_time_ms is not None

    async def test_execute_invalid_args(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        registry = ToolRegistry()
        register_builtins(registry, sandbox)
        executor = ToolExecutor(registry, sandbox)

        result = await executor.execute(
            "read_file",
            {},
            tool_call_id="call-2",
        )
        assert result.success is False
        assert "参数校验失败" in (result.error or "")

    async def test_execute_not_found(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        registry = ToolRegistry()
        executor = ToolExecutor(registry, sandbox)

        with pytest.raises(ToolNotFoundError):
            await executor.execute("nonexistent", {})

    def test_validate_arguments_pass(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert validate_arguments(schema, {"name": "test"}) is None

    def test_validate_arguments_fail(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        error = validate_arguments(schema, {})
        assert error is not None
        assert "name" in error


# ── Task 5.6: 沙箱与工具覆盖 ────────────────────────────────────────────────


class TestSandbox:
    """Task 5.6: 沙箱执行环境验证。"""

    def test_path_in_whitelist(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        result = sandbox.check_path(tmp_path / "some_file.txt")
        assert result == (tmp_path / "some_file.txt").resolve()

    def test_path_outside_whitelist_raises(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        with pytest.raises(SandboxViolationError, match="不在沙箱白名单内"):
            sandbox.check_path("/etc/passwd")

    def test_empty_whitelist_allows_all(self) -> None:
        sandbox = Sandbox(ToolsConfig(allowed_paths=[]))
        result = sandbox.check_path("/any/path")
        assert result is not None

    def test_network_allowed(self) -> None:
        sandbox = Sandbox(ToolsConfig(network_allowed=True))
        sandbox.check_network()

    def test_network_blocked_raises(self) -> None:
        sandbox = Sandbox(ToolsConfig(network_allowed=False))
        with pytest.raises(SandboxViolationError, match="网络出站访问被沙箱策略禁止"):
            sandbox.check_network()


class TestOverride:
    """Task 5.6: 工具覆盖机制验证。"""

    async def test_override_existing_tool(self) -> None:
        registry = ToolRegistry()
        original = ToolDefinition(
            name="my_tool",
            description="original",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        )

        async def original_handler(args: dict[str, Any]) -> str:
            return "original"

        registry.register(original, original_handler)

        new_def = ToolDefinition(
            name="my_tool",
            description="overridden",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}, "y": {"type": "integer"}},
                "required": ["x"],
            },
        )

        async def new_handler(args: dict[str, Any]) -> str:
            return "overridden"

        override_tool(registry, new_def, new_handler)

        entry = registry.get_entry("my_tool")
        assert entry.definition.description == "overridden"
        assert entry.overridden_from == "my_tool"
        result = await entry.handler({"x": "test"})
        assert result == "overridden"

    def test_override_nonexistent_raises(self) -> None:
        registry = ToolRegistry()
        defn = ToolDefinition(
            name="missing",
            description="d",
            parameters={"type": "object", "properties": {}},
        )

        async def handler(args: dict[str, Any]) -> str:
            return "ok"

        with pytest.raises(ToolError, match="不存在"):
            override_tool(registry, defn, handler)

    def test_override_incompatible_schema_raises(self) -> None:
        registry = ToolRegistry()
        original = ToolDefinition(
            name="my_tool",
            description="orig",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
            },
        )

        async def handler(args: dict[str, Any]) -> str:
            return "ok"

        registry.register(original, handler)

        incomplete = ToolDefinition(
            name="my_tool",
            description="bad override",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        )
        with pytest.raises(ToolError, match="缺少原始必填参数"):
            override_tool(registry, incomplete, handler)


# ── 内置工具注册集成 ─────────────────────────────────────────────────────────


class TestBuiltinRegistration:
    """内置工具批量注册验证。"""

    def test_register_all_builtins(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        sandbox = make_sandbox(tmp_path)
        register_builtins(registry, sandbox)

        expected = {
            "read_file", "write_file", "edit_file", "list_dir",
            "grep_search", "find_by_name", "code_search",
            "run_command",
            "web_fetch", "web_search",
            "get_system_info",
            "update_plan", "update_notes", "ask_user", "attempt_completion",
        }
        registered = set(registry.list_tools())
        assert expected == registered

    def test_schemas_export_openai_format(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        sandbox = make_sandbox(tmp_path)
        register_builtins(registry, sandbox)

        schemas = registry.get_tool_schemas()
        assert len(schemas) == 15
        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]
            assert "description" in s["function"]
            assert "parameters" in s["function"]
