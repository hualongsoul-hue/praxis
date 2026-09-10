"""S7 上下文引擎单元测试。"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from praxis.config.schemas import ContextConfig
from praxis.context.assembler import PromptAssembler
from praxis.context.compaction import ContextCompactor
from praxis.context.jit_retrieval import ContentLoader, JITRetriever
from praxis.context.masking import ObservationMasker
from praxis.context.tool_injection import ToolInjector
from praxis.models.context import TurnContext
from praxis.models.messages import ImageContent, ImageUrl, TextContent
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.registry import ToolRegistry

# ── 公共 Mock ──────────────────────────────────────────────────────────────

def mock_get_token_count(messages: list[dict[str, Any]], model: str = "default") -> int:
    """根据内容长度近似计算 token 数。"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4 + 1
    return total


def mock_get_max_tokens(model: str = "default") -> int:
    return 128000


ASSEMBLER_MOD = "praxis.context.assembler"
COMPACTION_MOD = "praxis.context.compaction"
MASKING_MOD = "praxis.context.masking"


# ── Task 11.1: 分层 Prompt 组装 ─────────────────────────────────────────────


class TestPromptAssembler:
    """分层 Prompt 组装测试。"""

    @patch(f"{ASSEMBLER_MOD}.get_max_tokens", side_effect=mock_get_max_tokens)
    @patch(f"{ASSEMBLER_MOD}.get_token_count", side_effect=mock_get_token_count)
    def test_basic_assembly(self, mock_tc: Any, mock_mt: Any) -> None:
        config = ContextConfig()
        asm = PromptAssembler(config)
        turn = TurnContext(user_content="你好", user_text="你好")
        result = asm.assemble_prompt(turn)
        assert len(result.messages) >= 2
        assert result.messages[0]["role"] == "system"
        assert result.messages[-1]["role"] == "user"
        assert result.messages[-1]["content"] == "你好"
        assert "system_prompt" in result.layers_included
        assert "current_message" in result.layers_included

    @patch(f"{ASSEMBLER_MOD}.get_max_tokens", side_effect=mock_get_max_tokens)
    @patch(f"{ASSEMBLER_MOD}.get_token_count", side_effect=mock_get_token_count)
    def test_structured_content_is_serialized_in_original_order(
        self,
        mock_tc: Any,
        mock_mt: Any,
    ) -> None:
        assembler = PromptAssembler(ContextConfig())
        turn = TurnContext(
            user_content=[
                TextContent(text="describe"),
                ImageContent(image_url=ImageUrl(url="data:image/png;base64,cG5n")),
            ],
            user_text="describe\n\n[image: image.png, image/png, 3 bytes]",
        )

        result = assembler.assemble_prompt(turn)

        user_messages = [
            message for message in result.messages if message["role"] == "user"
        ]
        assert user_messages == [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,cG5n",
                        "detail": "auto",
                    },
                },
            ],
        }]

    @patch(f"{ASSEMBLER_MOD}.get_max_tokens", side_effect=mock_get_max_tokens)
    @patch(f"{ASSEMBLER_MOD}.get_token_count", side_effect=mock_get_token_count)
    def test_all_layers(self, mock_tc: Any, mock_mt: Any) -> None:
        config = ContextConfig()
        asm = PromptAssembler(config)
        turn = TurnContext(
            user_content="写代码",
            user_text="写代码",
            developer_instructions="使用 Python",
            user_instructions="简洁风格",
        )
        result = asm.assemble_prompt(
            turn,
            memory_index="记忆索引内容",
            working_memory=[{"role": "user", "content": "之前的消息"}],
            semantic_results="语义检索结果",
            skill_index="pdf-processor: 处理 PDF",
        )
        assert "developer_instructions" in result.layers_included
        assert "user_instructions" in result.layers_included
        assert "memory_index" in result.layers_included
        assert "working_memory" in result.layers_included
        assert "semantic_retrieval" in result.layers_included
        assert "skill_index" in result.layers_included
        assert result.token_count > 0

    @patch(f"{ASSEMBLER_MOD}.get_max_tokens", side_effect=mock_get_max_tokens)
    @patch(f"{ASSEMBLER_MOD}.get_token_count", side_effect=mock_get_token_count)
    def test_update_with_result(self, mock_tc: Any, mock_mt: Any) -> None:
        config = ContextConfig()
        asm = PromptAssembler(config)
        asm.update_with_result([{"role": "tool", "content": "结果", "tool_call_id": "1"}])
        assert len(asm.conversation_history) == 1

    @patch(f"{ASSEMBLER_MOD}.get_max_tokens", side_effect=mock_get_max_tokens)
    @patch(f"{ASSEMBLER_MOD}.get_token_count", side_effect=mock_get_token_count)
    def test_update_with_response(self, mock_tc: Any, mock_mt: Any) -> None:
        config = ContextConfig()
        asm = PromptAssembler(config)
        asm.update_with_response({"role": "assistant", "content": "回复"})
        assert len(asm.conversation_history) == 1

    @patch(f"{ASSEMBLER_MOD}.get_max_tokens", side_effect=mock_get_max_tokens)
    @patch(f"{ASSEMBLER_MOD}.get_token_count", side_effect=mock_get_token_count)
    def test_get_token_usage(self, mock_tc: Any, mock_mt: Any) -> None:
        config = ContextConfig()
        asm = PromptAssembler(config)
        usage = asm.get_token_usage()
        assert usage.max_tokens == 128000
        assert usage.compaction_needed is False

    @patch(f"{ASSEMBLER_MOD}.get_max_tokens", side_effect=mock_get_max_tokens)
    @patch(f"{ASSEMBLER_MOD}.get_token_count", side_effect=mock_get_token_count)
    def test_add_file_ref_keeps_recent(self, mock_tc: Any, mock_mt: Any) -> None:
        config = ContextConfig(recent_file_refs_keep=3)
        asm = PromptAssembler(config)
        for i in range(5):
            asm.add_file_ref(f"file{i}.py")
        assert len(asm.file_refs) == 3
        assert asm.file_refs[0] == "file2.py"

    @patch(f"{ASSEMBLER_MOD}.get_max_tokens", side_effect=mock_get_max_tokens)
    @patch(f"{ASSEMBLER_MOD}.get_token_count", side_effect=mock_get_token_count)
    def test_system_prompt_override(self, mock_tc: Any, mock_mt: Any) -> None:
        config = ContextConfig()
        asm = PromptAssembler(config)
        turn = TurnContext(
            user_content="hi",
            user_text="hi",
            system_prompt_override="自定义系统提示",
        )
        result = asm.assemble_prompt(turn)
        assert "自定义系统提示" in result.messages[0]["content"]

    @patch(f"{ASSEMBLER_MOD}.get_max_tokens", side_effect=mock_get_max_tokens)
    @patch(f"{ASSEMBLER_MOD}.get_token_count", side_effect=mock_get_token_count)
    def test_tool_schemas_included(self, mock_tc: Any, mock_mt: Any) -> None:
        config = ContextConfig()
        asm = PromptAssembler(config)
        asm.set_tool_schemas([{"type": "function", "function": {"name": "test"}}])
        turn = TurnContext(user_content="test", user_text="test")
        result = asm.assemble_prompt(turn)
        assert len(result.tools) == 1


# ── Task 11.2: 动态工具集注入 ───────────────────────────────────────────────


def make_registry_with_tools() -> ToolRegistry:
    """创建包含多类别工具的注册表。"""
    reg = ToolRegistry()
    tools = [
        ("read_file", "file_ops"),
        ("write_file", "file_ops"),
        ("grep_search", "search"),
        ("run_command", "shell"),
        ("get_system_info", "system"),
    ]
    for name, category in tools:
        reg.register(
            ToolDefinition(
                name=name,
                description=f"{name} 工具",
                parameters={"type": "object", "properties": {}},
                metadata=ToolMetadata(category=category),
            ),
            handler=AsyncMock(return_value="ok"),
        )
    return reg


class TestToolInjector:
    """动态工具集注入测试。"""

    def test_general_stage(self) -> None:
        reg = make_registry_with_tools()
        injector = ToolInjector(reg)
        tools = injector.get_tools_for_stage("general")
        assert len(tools) >= 4

    def test_planning_stage_minimal(self) -> None:
        reg = make_registry_with_tools()
        injector = ToolInjector(reg)
        tools = injector.get_tools_for_stage("planning")
        names = {t["function"]["name"] for t in tools}
        assert "get_system_info" in names
        assert "run_command" not in names

    def test_coding_stage(self) -> None:
        reg = make_registry_with_tools()
        injector = ToolInjector(reg)
        tools = injector.get_tools_for_stage("coding")
        names = {t["function"]["name"] for t in tools}
        assert "read_file" in names
        assert "run_command" in names

    def test_load_extension_group(self) -> None:
        reg = make_registry_with_tools()
        injector = ToolInjector(reg)
        # 初始 planning 阶段没有 file_ops
        tools = injector.get_tools_for_stage("planning")
        names_before = {t["function"]["name"] for t in tools}
        assert "read_file" not in names_before
        # 加载扩展组
        injector.load_extension_group("file_ops")
        tools2 = injector.get_tools_for_stage("planning")
        names_after = {t["function"]["name"] for t in tools2}
        assert "read_file" in names_after

    def test_unload_extension_group(self) -> None:
        reg = make_registry_with_tools()
        injector = ToolInjector(reg)
        injector.load_extension_group("file_ops")
        injector.unload_extension_group("file_ops")
        assert "file_ops" not in injector.get_loaded_groups()

    def test_dedup(self) -> None:
        reg = make_registry_with_tools()
        injector = ToolInjector(reg)
        tools = injector.get_tools_for_stage("coding")
        names = [t["function"]["name"] for t in tools]
        assert len(names) == len(set(names))


# ── Task 11.3: 上下文压缩 ───────────────────────────────────────────────────


class TestContextCompactor:
    """上下文压缩测试。"""

    @pytest.mark.parametrize("important", ["call", "first_result", "second_result"])
    async def test_compaction_preserves_complete_tool_exchange(self, important: str) -> None:
        from tests.test_runtime import FakeGateway

        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "architecture" if important == "call" else "",
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "first", "arguments": "{}"}},
                    {"id": "b", "type": "function", "function": {"name": "second", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "bug" if important == "first_result" else "ok"},
            {"role": "tool", "tool_call_id": "b", "content": "error" if important == "second_result" else "ok"},
        ]
        expected_exchange = messages[1:]
        await ContextCompactor(ContextConfig(), FakeGateway()).compact(messages, [])
        assert messages[1:] == expected_exchange

    async def test_zero_file_reference_retention_really_discards_references(self) -> None:
        from tests.test_runtime import FakeGateway

        result = await ContextCompactor(
            ContextConfig(recent_file_refs_keep=0), FakeGateway(),
        ).compact([{"role": "user", "content": "hello"}], ["a.py", "b.py"])
        assert result.retained_file_refs == []

    async def test_empty_summary_does_not_destroy_history(self) -> None:
        from tests.test_runtime import FakeGateway

        class EmptySummaryGateway(FakeGateway):
            async def complete(self, *args: Any, **kwargs: Any):
                response = await super().complete(*args, **kwargs)
                return response.model_copy(update={"content": ""})

        messages = [{"role": "user", "content": "unfinished request"}]
        await ContextCompactor(ContextConfig(), EmptySummaryGateway()).compact(messages, [])
        assert messages == [{"role": "user", "content": "unfinished request"}]

    @patch(f"{COMPACTION_MOD}.get_token_count", side_effect=mock_get_token_count)
    async def test_compact(self, mock_tc: Any) -> None:
        config = ContextConfig()
        gateway = AsyncMock()

        async def mock_summarize(gw: Any, content: str, **kw: Any) -> str:
            return "摘要内容"

        compactor = ContextCompactor(config, gateway)
        messages = [
            {"role": "user", "content": "普通消息" * 50},
            {"role": "assistant", "content": "普通回复" * 50},
            {"role": "user", "content": "关于架构的讨论"},
        ]

        with patch("praxis.context.compaction.summarize", side_effect=mock_summarize):
            result = await compactor.compact(messages, ["a.py", "b.py"])

        assert result.original_tokens > 0
        assert result.compacted_tokens < result.original_tokens
        assert len(result.retained_file_refs) <= config.recent_file_refs_keep

    def test_is_critical(self) -> None:
        assert ContextCompactor.is_critical({"content": "架构决策记录"}) is True
        assert ContextCompactor.is_critical({"content": "fix bug #123"}) is True
        assert ContextCompactor.is_critical({"content": "好的"}) is False

    def test_combine_messages(self) -> None:
        msgs = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "回复"},
        ]
        combined = ContextCompactor.combine_messages(msgs)
        assert "[user] 你好" in combined
        assert "[assistant] 回复" in combined

    @patch(f"{COMPACTION_MOD}.get_token_count", side_effect=mock_get_token_count)
    async def test_compact_preserves_critical(self, mock_tc: Any) -> None:
        config = ContextConfig()
        gateway = AsyncMock()

        async def mock_summarize(gw: Any, content: str, **kw: Any) -> str:
            return "摘要"

        compactor = ContextCompactor(config, gateway)
        messages = [
            {"role": "user", "content": "普通内容"},
            {"role": "assistant", "content": "关于架构的重要决策"},
        ]

        with patch("praxis.context.compaction.summarize", side_effect=mock_summarize):
            await compactor.compact(messages, [])

        contents = [m.get("content", "") for m in messages]
        assert any("架构" in c for c in contents)


# ── Task 11.4: 观察遮蔽 ─────────────────────────────────────────────────────


class TestObservationMasker:
    """观察遮蔽测试。"""

    @patch(f"{MASKING_MOD}.get_token_count", side_effect=mock_get_token_count)
    def test_mask_by_distance(self, mock_tc: Any) -> None:
        config = ContextConfig(masking_turn_distance=2)
        masker = ObservationMasker(config)
        messages = [
            {"role": "assistant", "content": "回复1"},
            {"role": "tool", "tool_call_id": "1", "content": "旧工具输出"},
            {"role": "assistant", "content": "回复2"},
            {"role": "assistant", "content": "回复3"},
            {"role": "tool", "tool_call_id": "2", "content": "新工具输出"},
        ]
        count = masker.apply_masking(messages, current_turn=4)
        assert count >= 1
        assert messages[1]["content"] == "[工具输出已遮蔽——调用记录保留]"
        assert messages[1]["tool_call_id"] == "1"

    @patch(f"{MASKING_MOD}.get_token_count", side_effect=lambda msgs, model="default": 3000)
    def test_mask_by_size(self, mock_tc: Any) -> None:
        config = ContextConfig(masking_turn_distance=100, masking_token_threshold=2000)
        masker = ObservationMasker(config)
        messages = [
            {"role": "tool", "tool_call_id": "1", "content": "x" * 10000},
        ]
        count = masker.apply_masking(messages, current_turn=0)
        assert count == 1
        assert "遮蔽" in messages[0]["content"]

    @patch(f"{MASKING_MOD}.get_token_count", side_effect=mock_get_token_count)
    def test_no_mask_recent(self, mock_tc: Any) -> None:
        config = ContextConfig(masking_turn_distance=10)
        masker = ObservationMasker(config)
        messages = [
            {"role": "tool", "tool_call_id": "1", "content": "新输出"},
        ]
        count = masker.apply_masking(messages, current_turn=0)
        assert count == 0
        assert messages[0]["content"] == "新输出"

    @patch(f"{MASKING_MOD}.get_token_count", side_effect=mock_get_token_count)
    def test_preserves_tool_call_id(self, mock_tc: Any) -> None:
        config = ContextConfig(masking_turn_distance=0)
        masker = ObservationMasker(config)
        messages = [
            {"role": "assistant", "content": "a"},
            {"role": "tool", "tool_call_id": "abc-123", "content": "data"},
        ]
        masker.apply_masking(messages, current_turn=5)
        assert messages[1]["tool_call_id"] == "abc-123"


# ── Task 11.5: 即时检索 ─────────────────────────────────────────────────────


class TestJITRetriever:
    """即时检索测试。"""

    def test_register_and_index(self) -> None:
        jit = JITRetriever()
        jit.register_identifier("src/main.py", "file", "/project")
        jit.register_identifier("def foo()", "function", "/project/main.py")
        index = jit.get_identifier_index()
        assert len(index) == 2
        assert index[0]["kind"] == "file"

    def test_unregister(self) -> None:
        jit = JITRetriever()
        jit.register_identifier("x.py", "file", "/project")
        assert jit.unregister_identifier("x.py") is True
        assert jit.unregister_identifier("x.py") is False
        assert len(jit.get_identifier_index()) == 0

    def test_get_by_kind(self) -> None:
        jit = JITRetriever()
        jit.register_identifier("a.py", "file", "/p")
        jit.register_identifier("def f()", "function", "/p")
        jit.register_identifier("b.py", "file", "/p")
        files = jit.get_identifiers_by_kind("file")
        assert len(files) == 2

    def test_add_and_get_examples(self) -> None:
        jit = JITRetriever()
        jit.add_example("coding", "写排序", "这是快排实现")
        jit.add_example("coding", "写搜索", "这是二分搜索")
        jit.add_example("debugging", "修 bug", "检查日志")
        examples = jit.get_examples_for_task("coding", max_examples=2)
        assert len(examples) == 4  # 2 examples * 2 messages each
        assert examples[0]["role"] == "user"
        assert examples[1]["role"] == "assistant"

    def test_get_examples_max_limit(self) -> None:
        jit = JITRetriever()
        for i in range(5):
            jit.add_example("coding", f"问题{i}", f"回答{i}")
        examples = jit.get_examples_for_task("coding", max_examples=2)
        assert len(examples) == 4

    async def test_load_content(self) -> None:
        jit = JITRetriever()
        jit.register_identifier("main.py", "file", "/project")

        async def mock_read(source: str, identifier: str) -> str:
            return f"content of {identifier}"

        loader = ContentLoader(mock_read)
        jit.set_content_loader(loader)
        content = await jit.load_content("main.py")
        assert content == "content of main.py"

    async def test_load_content_no_loader(self) -> None:
        jit = JITRetriever()
        jit.register_identifier("x.py", "file", "/p")
        content = await jit.load_content("x.py")
        assert content is None

    async def test_load_content_not_found(self) -> None:
        jit = JITRetriever()
        content = await jit.load_content("nonexistent")
        assert content is None


class TestJITInjection:
    """S7 JIT few-shot 与标识符索引注入。"""

    def test_assemble_injects_few_shot_and_identifier(self) -> None:
        from praxis.config.schemas import ContextConfig
        from praxis.context.assembler import PromptAssembler
        from praxis.models.context import TurnContext

        asm = PromptAssembler(ContextConfig(), model="gpt-4o-mini")
        prompt = asm.assemble_prompt(
            TurnContext(user_content="实现快排", user_text="实现快排"),
            identifier_index="- [function] quicksort (algos.py)",
            few_shot_messages=[
                {"role": "user", "content": "示例问"},
                {"role": "assistant", "content": "示例答"},
            ],
        )
        # 标识符索引进系统消息
        assert "identifier_index" in prompt.messages[0]["content"]
        assert "quicksort" in prompt.messages[0]["content"]
        # few-shot 作为演示消息出现
        contents = [m.get("content") for m in prompt.messages]
        assert "示例问" in contents and "示例答" in contents
        assert "few_shot" in prompt.layers_included

    async def test_loop_pulls_jit_by_task_stage(self) -> None:
        """循环 prepare_run 应按 task_stage 从 JITRetriever 取示例与索引。"""
        from unittest.mock import AsyncMock, MagicMock

        from praxis.config.schemas import ContextConfig
        from praxis.context.assembler import PromptAssembler
        from praxis.context.jit_retrieval import JITRetriever
        from praxis.models.context import RunContext, TurnContext
        from praxis.orchestrator.loop import OrchestrationLoop

        jit = JITRetriever()
        jit.add_example("coding", "如何排序", "用 sorted()", tags=["py"])
        jit.register_identifier("quicksort", "function", "algos.py")

        loop = OrchestrationLoop(
            config=MagicMock(),
            gateway=MagicMock(),
            assembler=PromptAssembler(ContextConfig(), model="gpt-4o-mini"),
            tool_coordinator=MagicMock(),
            guardrails=AsyncMock(),
            termination=MagicMock(),
            strategy=MagicMock(),
            parser=MagicMock(),
            emitter=MagicMock(),
            jit_retriever=jit,
        )
        loop.guardrails.check_input = AsyncMock(
            return_value=MagicMock(tripwire=False, verdict=None)
        )
        loop.strategy.get_step_instruction = MagicMock(return_value="")
        ctx = RunContext(turn_context=TurnContext(
            user_content="排序",
            user_text="排序",
            task_stage="coding",
        ))

        early = await loop.prepare_run(ctx)
        assert early is None
        # JIT 段应按 task_stage 填充 few-shot 与标识符索引
        assert ctx.few_shot_messages and ctx.few_shot_messages[0]["content"] == "如何排序"
        assert "quicksort" in ctx.identifier_index
