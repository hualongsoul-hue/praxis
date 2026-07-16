"""上下文压缩（Compaction）。

Token 超过阈值（默认 80%）自动触发，
调用 S4 summarize 生成摘要，
保留规则（架构决策/Bug/实现细节保留，冗余丢弃），
保留最近 5 个关键文件引用，工具结果清除策略。
"""

from typing import Any

from praxis.config.schemas import ContextConfig
from praxis.gateway.metering import get_token_count
from praxis.gateway.router import GatewayRouter
from praxis.gateway.tasks import summarize
from praxis.models.context import CompactionResult
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("context.compaction")

RETENTION_KEYWORDS = frozenset({
    "架构", "设计", "决策", "bug", "修复", "错误", "实现",
    "architecture", "design", "decision", "fix", "error", "implement",
})


class ContextCompactor:
    """上下文压缩器。

    负责在 Token 超限时压缩对话历史，保留关键信息。
    """

    def __init__(
        self,
        config: ContextConfig,
        gateway: GatewayRouter,
        model: str = "default",
    ) -> None:
        self.config = config
        self.gateway = gateway
        self.model = model

    async def compact(
        self,
        messages: list[dict[str, Any]],
        file_refs: list[str],
    ) -> CompactionResult:
        """压缩消息历史。

        策略：
        1. 清除旧工具结果原始输出（保留调用记录）
        2. 将非关键消息合并为摘要
        3. 保留含关键词的消息
        4. 保留最近的文件引用

        Args:
            messages: 当前消息列表。
            file_refs: 文件引用列表。

        Returns:
            压缩结果。
        """
        original_tokens = get_token_count(messages, self.model)

        # 分离关键消息和可压缩消息
        critical: list[dict[str, Any]] = []
        compressible: list[dict[str, Any]] = []

        for msg in messages:
            if self.is_critical(msg):
                critical.append(msg)
            else:
                compressible.append(msg)

        # 生成摘要
        summary = ""
        if compressible:
            combined = self.combine_messages(compressible)
            summary = await summarize(
                self.gateway,
                combined,
                instruction=(
                    "请对以下对话内容进行精炼摘要，保留所有关键技术决策、"
                    "未解决问题和实现细节。去除冗余工具输出和重复内容。"
                ),
                model=self.model,
            )

        # 重建消息列表
        compacted: list[dict[str, Any]] = []
        if summary:
            compacted.append({
                "role": "system",
                "content": f"<compaction_summary>\n{summary}\n</compaction_summary>",
            })
        compacted.extend(critical)

        # 保留最近的文件引用
        keep = self.config.recent_file_refs_keep
        retained_refs = file_refs[-keep:] if len(file_refs) > keep else list(file_refs)

        compacted_tokens = get_token_count(compacted, self.model)

        emit_metric(
            "context_compaction",
            float(original_tokens - compacted_tokens),
            {"original": str(original_tokens), "compacted": str(compacted_tokens)},
            "histogram",
        )
        log.info(
            "上下文压缩完成",
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            critical_count=len(critical),
            summary_length=len(summary),
        )

        # 替换原始消息列表
        messages.clear()
        messages.extend(compacted)

        return CompactionResult(
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            summary=summary,
            retained_file_refs=retained_refs,
        )

    def strip_tool_outputs(
        self,
        messages: list[dict[str, Any]],
        keep_recent: int = 3,
    ) -> int:
        """清除旧工具输出内容（保留调用记录）。

        轻量级压缩：只清除 tool role 消息的 content，
        保留最近 keep_recent 条。

        Args:
            messages: 消息列表（原地修改）。
            keep_recent: 保留最近的工具结果数。

        Returns:
            清除的消息数。
        """
        tool_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "tool"
        ]
        if len(tool_indices) <= keep_recent:
            return 0

        to_strip = tool_indices[:-keep_recent]
        for idx in to_strip:
            messages[idx] = {
                "role": "tool",
                "tool_call_id": messages[idx].get("tool_call_id", ""),
                "content": "[输出已省略]",
            }
        return len(to_strip)

    @staticmethod
    def is_critical(msg: dict[str, Any]) -> bool:
        """判断消息是否包含关键信息。

        Args:
            msg: 消息字典。

        Returns:
            是否关键。
        """
        content = msg.get("content", "")
        if msg.get("role") == "user" and isinstance(content, list):
            return True
        if not isinstance(content, str):
            return False
        content_lower = content.lower()
        return any(kw in content_lower for kw in RETENTION_KEYWORDS)

    @staticmethod
    def combine_messages(messages: list[dict[str, Any]]) -> str:
        """将消息列表合并为单一文本。"""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                parts.append(f"[{role}] {content}")
        return "\n\n".join(parts)
