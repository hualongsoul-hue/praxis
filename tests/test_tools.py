"""S5 工具系统验证测试。"""

import asyncio
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from praxis.config.schemas import SubagentConfig, ToolsConfig
from praxis.exceptions import (
    ToolError,
    ToolNotFoundError,
    ToolPolicyViolationError,
    ToolTimeoutError,
)
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.subagent.resource_control import ResourceController
from praxis.tools.builtins.file_ops.read_file import create_handler as read_file_handler
from praxis.tools.builtins.network import web_fetch, web_search
from praxis.tools.builtins.registration import register_builtins
from praxis.tools.builtins.search.grep_search import create_handler as grep_search_handler
from praxis.tools.executor import ToolExecutor, validate_arguments
from praxis.tools.override import override_tool
from praxis.tools.policy import ToolPolicy
from praxis.tools.process import ProcessRunner
from praxis.tools.registry import ToolRegistry


def make_sandbox(tmp_path: Path) -> ToolPolicy:
    config = ToolsConfig(
        allowed_paths=[str(tmp_path)],
        default_timeout=5.0,
        shell_timeout=10.0,
    )
    return ToolPolicy(config.model_copy(update={"shell_enabled": True}))


def make_sandbox_open() -> ToolPolicy:
    """启用非文件能力的测试策略。"""
    return ToolPolicy(ToolsConfig(network_allowed=True))


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

    async def test_atomic_write_preserves_original_on_replace_failure(
        self,
        tmp_path: Path,
    ) -> None:
        sandbox = make_sandbox(tmp_path)
        target = tmp_path / "important.txt"
        target.write_text("original", encoding="utf-8")
        from praxis.tools.builtins.file_ops.write_file import create_handler

        with patch("praxis.tools.filesystem.os.replace", side_effect=OSError("disk")):
            with pytest.raises(OSError, match="disk"):
                await create_handler(sandbox)({
                    "file_path": str(target),
                    "content": "replacement",
                })

        assert target.read_text(encoding="utf-8") == "original"
        assert list(tmp_path.glob("praxis-write-*.tmp")) == []

    async def test_write_rejects_symlink_escape(self, tmp_path: Path) -> None:
        safe = tmp_path / "safe"
        outside = tmp_path / "outside"
        safe.mkdir()
        outside.mkdir()
        outside_target = outside / "target.txt"
        outside_target.write_text("original", encoding="utf-8")
        link = safe / "link.txt"
        try:
            link.symlink_to(outside_target)
        except OSError as exc:
            pytest.skip(f"当前平台不允许创建测试符号链接: {exc}")
        sandbox = ToolPolicy(ToolsConfig(allowed_paths=[str(safe)]))
        from praxis.tools.builtins.file_ops.write_file import create_handler

        with pytest.raises(ToolPolicyViolationError):
            await create_handler(sandbox)({
                "file_path": str(link),
                "content": "escaped",
            })

        assert outside_target.read_text(encoding="utf-8") == "original"

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

    async def test_list_dir_bounds_recursive_descendant_count(self, tmp_path: Path) -> None:
        sandbox = ToolPolicy(ToolsConfig(
            allowed_paths=[str(tmp_path)],
            search_max_files=2,
        ))
        subdirectory = tmp_path / "sub"
        subdirectory.mkdir()
        (subdirectory / "a").write_text("a", encoding="utf-8")
        (subdirectory / "b").write_text("b", encoding="utf-8")
        (subdirectory / "c").write_text("c", encoding="utf-8")

        from praxis.tools.builtins.file_ops.list_dir import create_handler

        result = await create_handler(sandbox)({"dir_path": str(tmp_path)})

        assert "sub/  (1 items)" in result
        assert "目录统计已截断" in result


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
        assert "Time:" in result

    async def test_ask_user(self) -> None:
        from praxis.tools.builtins.autonomy.ask_user import handle

        result = await handle({"question": "你确定吗？"})
        assert "ASK_USER" in result
        assert "你确定吗" in result

    async def test_submit_result(self) -> None:
        from praxis.tools.builtins.autonomy.submit_result import handle

        result = await handle({"result": "任务完成"})
        assert "SUBMIT_RESULT" in result
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

    async def test_readonly_concurrency_uses_configured_limit(self, tmp_path: Path) -> None:
        policy = ToolPolicy(
            ToolsConfig(
                allowed_paths=[str(tmp_path)],
                max_concurrent_readonly=1,
            )
        )
        registry = ToolRegistry()
        first_started = asyncio.Event()
        release = asyncio.Event()
        active = 0
        maximum_active = 0

        async def handler(arguments: dict[str, Any]) -> str:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            first_started.set()
            await release.wait()
            active -= 1
            return "ok"

        registry.register(
            ToolDefinition(
                name="bounded_read",
                description="bounded",
                parameters={"type": "object", "properties": {}},
                metadata=ToolMetadata(readonly=True),
            ),
            handler,
        )
        executor = ToolExecutor(registry, policy)
        first = asyncio.create_task(executor.execute("bounded_read", {}))
        second = asyncio.create_task(executor.execute("bounded_read", {}))
        await first_started.wait()
        await asyncio.sleep(0)
        assert maximum_active == 1
        release.set()
        results = await asyncio.gather(first, second)
        assert all(result.success for result in results)
        assert maximum_active == 1

    async def test_read_limit_is_shared_across_executor_instances(self, tmp_path: Path) -> None:
        policy = ToolPolicy(ToolsConfig(
            allowed_paths=[str(tmp_path)],
            max_concurrent_readonly=1,
        ))
        resources = ResourceController(
            SubagentConfig(),
            max_concurrent_readonly=1,
        )
        registry = ToolRegistry()
        started = asyncio.Event()
        release = asyncio.Event()
        active = 0
        maximum_active = 0

        async def handler(arguments: dict[str, Any]) -> str:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            started.set()
            await release.wait()
            active -= 1
            return "ok"

        registry.register(
            ToolDefinition(
                name="shared_read",
                description="shared read",
                parameters={"type": "object", "properties": {}},
                metadata=ToolMetadata(readonly=True),
            ),
            handler,
        )
        first_executor = ToolExecutor(registry, policy, resources=resources)
        second_executor = ToolExecutor(registry, policy, resources=resources)
        first = asyncio.create_task(first_executor.execute("shared_read", {}))
        second = asyncio.create_task(second_executor.execute("shared_read", {}))
        await started.wait()
        await asyncio.sleep(0)

        assert maximum_active == 1
        release.set()
        await asyncio.gather(first, second)
        assert maximum_active == 1

    async def test_writes_to_same_resource_serialize_across_executors(
        self,
        tmp_path: Path,
    ) -> None:
        policy = ToolPolicy(ToolsConfig(allowed_paths=[str(tmp_path)]))
        resources = ResourceController(SubagentConfig())
        registry = ToolRegistry()
        started = asyncio.Event()
        release = asyncio.Event()
        active = 0
        maximum_active = 0

        async def handler(arguments: dict[str, Any]) -> str:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            started.set()
            await release.wait()
            active -= 1
            return str(arguments["content"])

        registry.register(
            ToolDefinition(
                name="shared_write",
                description="shared write",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                },
                metadata=ToolMetadata(readonly=False),
            ),
            handler,
        )
        first_executor = ToolExecutor(registry, policy, resources=resources)
        second_executor = ToolExecutor(registry, policy, resources=resources)
        target = str(tmp_path / "shared.txt")
        first = asyncio.create_task(first_executor.execute(
            "shared_write", {"file_path": target, "content": "first"}
        ))
        second = asyncio.create_task(second_executor.execute(
            "shared_write", {"file_path": target, "content": "second"}
        ))
        await started.wait()
        await asyncio.sleep(0)

        assert maximum_active == 1
        release.set()
        results = await asyncio.gather(first, second)
        assert all(result.success for result in results)
        assert maximum_active == 1

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


class TestToolPolicy:
    """Task 5.6: 工具授权策略验证。"""

    def test_path_in_whitelist(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        result = sandbox.check_path(tmp_path / "some_file.txt")
        assert result == (tmp_path / "some_file.txt").resolve()

    def test_path_outside_whitelist_raises(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(tmp_path)
        with pytest.raises(ToolPolicyViolationError, match="不在工具授权根目录内"):
            sandbox.check_path("/etc/passwd")

    def test_empty_roots_deny_all(self) -> None:
        policy = ToolPolicy(ToolsConfig(allowed_paths=[]))
        with pytest.raises(ToolPolicyViolationError, match="未配置授权根目录"):
            policy.check_path("/any/path")

    async def test_network_allowed(self) -> None:
        sandbox = ToolPolicy(ToolsConfig(network_allowed=True))
        public_address = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        with patch("praxis.network.socket.getaddrinfo", return_value=public_address):
            target = await sandbox.check_url("https://EXAMPLE.com:443/contract?q=1")

        assert target.original_url == "https://EXAMPLE.com:443/contract?q=1"
        assert target.hostname == "example.com"
        assert target.port == 443
        assert target.host_header == "example.com:443"
        assert target.addresses == ("93.184.216.34",)

        changed_host_address = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
        ]
        with (
            patch(
                "praxis.network.socket.getaddrinfo",
                return_value=changed_host_address,
            ),
            pytest.raises(ToolPolicyViolationError, match="私网"),
        ):
            await sandbox.check_url("https://changed.example/contract?q=1")

    def test_network_blocked_raises(self) -> None:
        sandbox = ToolPolicy(ToolsConfig(network_allowed=False))
        with pytest.raises(ToolPolicyViolationError, match="网络出站访问被工具策略禁止"):
            sandbox.check_network()

    async def test_rejects_invalid_url_port(self) -> None:
        policy = ToolPolicy(ToolsConfig(network_allowed=True))
        with pytest.raises(ToolPolicyViolationError, match="端口"):
            await policy.check_url("https://example.com:invalid/")


class TestProcessRunner:
    async def test_timeout_terminates_process_tree(self, tmp_path: Path) -> None:
        runner = ProcessRunner()
        command = f'"{sys.executable}" -c "import time; time.sleep(30)"'
        with pytest.raises(ToolTimeoutError, match="超时"):
            await runner.run_shell(
                command,
                cwd=tmp_path,
                timeout=0.1,
                environment=dict(os.environ),
            )

    async def test_cancellation_terminates_process_tree(self, tmp_path: Path) -> None:
        runner = ProcessRunner()
        command = f'"{sys.executable}" -c "import time; time.sleep(30)"'
        task = asyncio.create_task(runner.run_shell(
            command,
            cwd=tmp_path,
            timeout=30,
            environment=dict(os.environ),
        ))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)


class FailingWebFetchCloseStream(httpx.AsyncByteStream):
    """Stream whose close failure must be safely contained by Web Fetch."""

    async def __aiter__(self):
        yield b"safe"

    async def aclose(self) -> None:
        raise httpx.ReadError("web_fetch_close_secret_marker")


class TestWebFetchSecurity:
    async def test_validates_every_redirect_and_blocks_private_target(self) -> None:
        policy = ToolPolicy(ToolsConfig(
            network_allowed=True,
            allow_private_networks=False,
        ))
        requested_hosts: list[str] = []

        def redirect(request: httpx.Request) -> httpx.Response:
            requested_hosts.append(request.url.host)
            return httpx.Response(
                302,
                headers={"location": "http://127.0.0.1/x"},
            )

        handler = web_fetch.create_handler(
            policy,
            lambda: httpx.MockTransport(redirect),
        )
        with pytest.raises(ToolPolicyViolationError, match="私网"):
            await handler({"url": "http://93.184.216.34/start"})
        assert requested_hosts == ["93.184.216.34"]

    async def test_pins_dns_target_and_preserves_host_and_sni(self, monkeypatch) -> None:
        policy = ToolPolicy(ToolsConfig(network_allowed=True))
        requests: list[httpx.Request] = []

        def resolve_host(host: str, port: int, **options):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, content=b"safe")

        monkeypatch.setattr(socket, "getaddrinfo", resolve_host)
        result = await web_fetch.create_handler(
            policy,
            lambda: httpx.MockTransport(handle),
        )({"url": "https://example.com/data"})

        assert "safe" in result
        assert requests[0].url.host == "93.184.216.34"
        assert requests[0].headers["host"] == "example.com"
        assert requests[0].extensions["sni_hostname"] == "example.com"

    async def test_enforces_streamed_byte_limit(self) -> None:
        policy = ToolPolicy(ToolsConfig(
            network_allowed=True,
            allow_private_networks=True,
            network_max_response_bytes=5,
        ))
        result = await web_fetch.create_handler(
            policy,
            lambda: httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content=b"0123456789",
                )
            ),
        )({
            "url": "http://127.0.0.1/data",
            "max_bytes": 999,
        })
        assert "01234" in result
        assert "012345" not in result
        assert "已截断" in result

    async def test_close_failure_after_success_is_safe(self, caplog) -> None:
        policy = ToolPolicy(ToolsConfig(
            network_allowed=True,
            allow_private_networks=True,
        ))

        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=FailingWebFetchCloseStream())

        caplog.set_level("DEBUG")
        handler = web_fetch.create_handler(
            policy,
            lambda: httpx.MockTransport(handle),
        )
        with pytest.raises(ToolPolicyViolationError) as captured:
            await handler({"url": "http://127.0.0.1/data"})

        rendered = "".join(
            traceback.format_exception(captured.type, captured.value, captured.tb)
        )
        assert "web_fetch_close_secret_marker" not in rendered
        assert "web_fetch_close_secret_marker" not in caplog.text


class TestBoundedFileAndSearchSecurity:
    async def test_read_file_rejects_oversized_input_without_blocking_loop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "large.txt"
        target.write_text("0123456789", encoding="utf-8")
        policy = ToolPolicy(ToolsConfig(
            allowed_paths=[str(tmp_path)],
            max_file_bytes=4,
        ))
        original_read_bytes = Path.read_bytes

        def slow_read(path: Path) -> bytes:
            time.sleep(0.05)
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", slow_read)
        task = asyncio.create_task(read_file_handler(policy)({"file_path": str(target)}))
        await asyncio.sleep(0.01)
        assert not task.done()
        with pytest.raises(ToolPolicyViolationError, match="大小上限"):
            await task

    async def test_search_rejects_nested_quantifier_regex(self, tmp_path: Path) -> None:
        target = tmp_path / "input.txt"
        target.write_text("a" * 100 + "!", encoding="utf-8")
        policy = ToolPolicy(ToolsConfig(allowed_paths=[str(tmp_path)]))

        with pytest.raises(ToolPolicyViolationError, match="正则"):
            await grep_search_handler(policy)({
                "pattern": "(a+)+$",
                "search_path": str(tmp_path),
            })

    async def test_search_stops_at_configured_file_and_byte_bounds(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("needle\n", encoding="utf-8")
        policy = ToolPolicy(ToolsConfig(
            allowed_paths=[str(tmp_path)],
            search_max_files=1,
            search_max_bytes=64,
            search_max_matches=10,
        ))

        result = await grep_search_handler(policy)({
            "pattern": "needle",
            "search_path": str(tmp_path),
        })

        assert result.count("needle") == 1
        assert "搜索范围已截断" in result

    async def test_web_search_validates_redirect_targets(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        policy = ToolPolicy(ToolsConfig(network_allowed=True))

        def resolve_host(*args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
            host = str(args[0])
            address = "127.0.0.1" if host == "localhost" else "93.184.216.34"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]

        monkeypatch.setattr(socket, "getaddrinfo", resolve_host)
        handler = web_search.create_handler(
            policy,
            lambda: httpx.MockTransport(lambda request: httpx.Response(
                302,
                headers={"location": "http://localhost/private"},
            )),
        )

        with pytest.raises(ToolPolicyViolationError, match="私网"):
            await handler({"query": "praxis"})

    async def test_close_failure_does_not_mask_redirect_policy_error(
        self,
        monkeypatch,
        caplog,
    ) -> None:
        policy = ToolPolicy(ToolsConfig(
            network_allowed=True,
            allow_private_networks=False,
        ))

        def resolve_host(host: str, port: int, **options):
            address = "93.184.216.34" if host == "example.com" else "127.0.0.1"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                302,
                headers={"location": "http://127.0.0.1/private"},
                stream=FailingWebFetchCloseStream(),
            )

        monkeypatch.setattr(socket, "getaddrinfo", resolve_host)
        caplog.set_level("DEBUG")
        handler = web_fetch.create_handler(
            policy,
            lambda: httpx.MockTransport(handle),
        )
        with pytest.raises(ToolPolicyViolationError, match="私网") as captured:
            await handler({"url": "http://example.com/start"})

        rendered = "".join(
            traceback.format_exception(captured.type, captured.value, captured.tb)
        )
        assert "web_fetch_close_secret_marker" not in rendered
        assert "web_fetch_close_secret_marker" not in caplog.text

    async def test_cross_host_redirect_uses_separate_connection_pools(
        self,
        monkeypatch,
    ) -> None:
        policy = ToolPolicy(ToolsConfig(network_allowed=True))
        pool_identities: list[object] = []
        requests: list[tuple[object, str, str]] = []

        def resolve_host(host: str, port: int, **options):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
            ]

        def create_transport() -> httpx.AsyncBaseTransport:
            pool_identity = object()
            pool_identities.append(pool_identity)

            def handle(request: httpx.Request) -> httpx.Response:
                requests.append(
                    (
                        pool_identity,
                        request.headers["host"],
                        request.extensions["sni_hostname"],
                    )
                )
                if request.headers["host"] == "host-a.example":
                    return httpx.Response(
                        302,
                        headers={"location": "https://host-b.example/file"},
                    )
                return httpx.Response(200, content=b"safe")

            return httpx.MockTransport(handle)

        monkeypatch.setattr(socket, "getaddrinfo", resolve_host)
        result = await web_fetch.create_handler(policy, create_transport)(
            {"url": "https://host-a.example/file"}
        )

        assert "safe" in result
        assert len(pool_identities) == 2
        assert requests == [
            (pool_identities[0], "host-a.example", "host-a.example"),
            (pool_identities[1], "host-b.example", "host-b.example"),
        ]


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
            "get_system_info", "sleep",
            "update_plan", "update_notes", "ask_user", "submit_result",
        }
        registered = set(registry.list_tools())
        assert expected == registered

    def test_schemas_export_openai_format(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        sandbox = make_sandbox(tmp_path)
        register_builtins(registry, sandbox)

        schemas = registry.get_tool_schemas()
        assert len(schemas) == 16
        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]
            assert "description" in s["function"]
            assert "parameters" in s["function"]


class TestJITTools:
    """S7 JIT 懒加载工具注册。"""

    def test_not_registered_without_content_loader(self) -> None:
        from praxis.context.jit_retrieval import JITRetriever
        from praxis.tools.builtins.jit_ops import register_jit_tools
        from praxis.tools.registry import ToolRegistry

        reg = ToolRegistry()
        assert register_jit_tools(reg, JITRetriever()) == []
        assert not reg.has_tool("jit_load_content")

    async def test_registered_and_loads_with_content_loader(self) -> None:
        from praxis.context.jit_retrieval import ContentLoader, JITRetriever
        from praxis.tools.builtins.jit_ops import register_jit_tools
        from praxis.tools.registry import ToolRegistry

        async def reader(source: str, identifier: str) -> str:
            return f"内容[{source}:{identifier}]"

        jit = JITRetriever()
        jit.set_content_loader(ContentLoader(reader))
        jit.register_identifier("foo", "function", "f.py")

        reg = ToolRegistry()
        names = register_jit_tools(reg, jit)
        assert names == ["jit_load_content"]
        handler = reg.get_entry("jit_load_content").handler
        out = await handler({"identifier": "foo"})
        assert "内容[f.py:foo]" in out
