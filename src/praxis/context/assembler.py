"""分层 Prompt 组装。

按 8 层优先级栈组装完整 Prompt：
1. 系统提示  2. 工具定义  3. 开发者指令  4. 用户指令
5. 记忆索引  6. 工作记忆  7. 语义检索  8. 当前消息
XML/Markdown 标签分隔，关键内容放置首尾。
"""

from typing import Any

from praxis.config.subsystems import ContextConfig
from praxis.gateway.metering import get_max_tokens, get_token_count
from praxis.models.context import AssembledPrompt, TokenUsage, TurnContext
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("context.assembler")

DEFAULT_SYSTEM_PROMPT = (
    "你是 Praxis Agent，一个高效的 AI 编程助手。\n"
    "严格遵循用户指令，产出高质量的代码和分析。"
)


class PromptAssembler:
    """分层 Prompt 组装器。

    消费 S5（工具 Schema）、S6（记忆内容）和 S14（技能索引），
    产出可直接送入 LLM 的完整 Prompt。
    """

    def __init__(
        self,
        config: ContextConfig,
        model: str = "default",
    ) -> None:
        self.config = config
        self.model = model
        self.conversation_history: list[dict[str, Any]] = []
        self.tool_schemas: list[dict[str, Any]] = []
        self.file_refs: list[str] = []
        self.compaction_count: int = 0

    def assemble_prompt(
        self,
        turn: TurnContext,
        memory_index: str = "",
        working_memory: list[dict[str, Any]] | None = None,
        semantic_results: str = "",
        skill_index: str = "",
    ) -> AssembledPrompt:
        """组装完整 Prompt。

        按 8 层优先级栈组装，关键内容放置首尾（对抗 Lost in the Middle 效应）。

        Args:
            turn: 当前轮次上下文。
            memory_index: S6 记忆索引文本。
            working_memory: S6 工作记忆消息历史。
            semantic_results: S6 语义检索结果文本。
            skill_index: S14 技能索引文本。

        Returns:
            组装后的完整 Prompt。
        """
        messages: list[dict[str, Any]] = []
        layers: list[str] = []

        # Layer 1: 系统提示（最高优先级，首位放置）
        system_prompt = turn.system_prompt_override or DEFAULT_SYSTEM_PROMPT
        system_content = system_prompt

        # Layer 3 & 4: 开发者 + 用户指令注入系统提示
        if turn.developer_instructions:
            system_content += f"\n\n<developer_instructions>\n{turn.developer_instructions}\n</developer_instructions>"
            layers.append("developer_instructions")

        if turn.user_instructions:
            system_content += f"\n\n<user_instructions>\n{turn.user_instructions}\n</user_instructions>"
            layers.append("user_instructions")

        # Layer 5: 技能索引注入系统提示
        if skill_index:
            system_content += f"\n\n<skill_index>\n{skill_index}\n</skill_index>"
            layers.append("skill_index")

        # Layer 5: 记忆索引注入系统提示
        if memory_index:
            system_content += f"\n\n<memory_index>\n{memory_index}\n</memory_index>"
            layers.append("memory_index")

        messages.append({"role": "system", "content": system_content})
        layers.append("system_prompt")

        # Layer 6: 工作记忆（对话历史）
        if working_memory:
            messages.extend(working_memory)
            layers.append("working_memory")

        # Layer 7: 语义检索结果
        if semantic_results:
            messages.append({
                "role": "system",
                "content": f"<semantic_context>\n{semantic_results}\n</semantic_context>",
            })
            layers.append("semantic_retrieval")

        # 历史对话
        if self.conversation_history:
            messages.extend(self.conversation_history)
            layers.append("conversation_history")

        # Layer 8: 当前用户消息（末位放置，确保 LLM 关注）
        messages.append({"role": "user", "content": turn.user_message})
        layers.append("current_message")

        # Layer 2: 工具定义
        layers.append("tool_definitions")

        max_tokens = get_max_tokens(self.model)
        token_count = get_token_count(messages, self.model)

        emit_metric(
            "context_assembled_tokens",
            float(token_count),
            {"model": self.model, "layers": str(len(layers))},
            "histogram",
        )

        log.info(
            "Prompt 组装完成",
            token_count=token_count,
            max_tokens=max_tokens,
            layers_count=len(layers),
        )

        return AssembledPrompt(
            messages=messages,
            tools=list(self.tool_schemas),
            token_count=token_count,
            max_tokens=max_tokens,
            layers_included=layers,
        )

    def update_with_result(self, tool_results: list[dict[str, Any]]) -> None:
        """更新上下文：追加工具执行结果。

        Args:
            tool_results: 工具执行结果消息列表。
        """
        self.conversation_history.extend(tool_results)

    def update_with_response(self, assistant_msg: dict[str, Any]) -> None:
        """更新上下文：追加助手响应。

        Args:
            assistant_msg: 助手响应消息。
        """
        self.conversation_history.append(assistant_msg)
        self.total_turns += 1

    def get_token_usage(self) -> TokenUsage:
        """获取当前上下文 Token 用量。"""
        max_tokens = get_max_tokens(self.model)
        current = get_token_count(self.conversation_history, self.model) if self.conversation_history else 0
        ratio = current / max_tokens if max_tokens > 0 else 0.0
        return TokenUsage(
            current_tokens=current,
            max_tokens=max_tokens,
            usage_ratio=ratio,
            compaction_needed=ratio >= self.config.compaction_threshold,
        )

    def set_tool_schemas(self, schemas: list[dict[str, Any]]) -> None:
        """设置当前可用工具 Schema。"""
        self.tool_schemas = schemas

    def add_file_ref(self, file_ref: str) -> None:
        """添加文件引用。"""
        if file_ref not in self.file_refs:
            self.file_refs.append(file_ref)
            keep = self.config.recent_file_refs_keep
            if len(self.file_refs) > keep:
                self.file_refs = self.file_refs[-keep:]

    @property
    def total_turns(self) -> int:
        """当前总轮次数。"""
        return sum(1 for m in self.conversation_history if m.get("role") == "assistant")

    @total_turns.setter
    def total_turns(self, value: int) -> None:
        """setter 仅用于内部兼容，实际轮次从对话历史中计算。"""
