"""模型辅助记忆提取（Extraction）。

每种认知类型使用独立提取提示，通过 S4 调用 LLM 分析对话内容，
区分有意义洞察和例行对话，单次对话可提取多条多类型记忆。
"""

import json
from typing import Any

from praxis.gateway.chat import chat
from praxis.gateway.router import GatewayRouter
from praxis.models.memory import (
    MemoryEntry,
    MemoryScope,
    MemoryType,
)
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("memory.extraction")

SEMANTIC_EXTRACTION_PROMPT = (
    "你是一个记忆提取专家。分析以下对话，提取出值得长期存储的**事实、知识和用户偏好**。\n"
    "忽略例行对话和无意义的闲聊。\n"
    "以 JSON 数组格式输出，每个条目包含：\n"
    '- "content": 记忆内容（简洁清晰的事实陈述）\n'
    '- "tags": 标签数组\n'
    '- "confidence": 0.0~1.0 置信度\n'
    "如果没有值得提取的信息，返回空数组 []。\n"
    "仅输出 JSON 数组，不添加其他文字。"
)

EPISODIC_EXTRACTION_PROMPT = (
    "你是一个记忆提取专家。分析以下对话，提取出**关键交互摘要和决策过程**。\n"
    "关注：用户做出了什么重要决策？解决了什么问题？经历了什么关键步骤？\n"
    "以 JSON 数组格式输出，每个条目包含：\n"
    '- "content": 交互摘要\n'
    '- "context": 情境上下文\n'
    '- "reasoning": 推理过程\n'
    '- "action": 采取的行动\n'
    '- "outcome": 达成的结果\n'
    '- "tags": 标签数组\n'
    "如果没有值得提取的信息，返回空数组 []。\n"
    "仅输出 JSON 数组，不添加其他文字。"
)

PROCEDURAL_EXTRACTION_PROMPT = (
    "你是一个记忆提取专家。分析以下对话，提取出**工作流程、使用模式和行为准则**。\n"
    "关注：用户惯用的工作流程、代码风格偏好、反复使用的操作模式。\n"
    "以 JSON 数组格式输出，每个条目包含：\n"
    '- "content": 流程或模式描述\n'
    '- "steps": 步骤数组（如适用）\n'
    '- "scenarios": 适用场景数组\n'
    '- "tags": 标签数组\n'
    "如果没有值得提取的信息，返回空数组 []。\n"
    "仅输出 JSON 数组，不添加其他文字。"
)

EXTRACTION_PROMPTS: dict[MemoryType, str] = {
    MemoryType.SEMANTIC: SEMANTIC_EXTRACTION_PROMPT,
    MemoryType.EPISODIC: EPISODIC_EXTRACTION_PROMPT,
    MemoryType.PROCEDURAL: PROCEDURAL_EXTRACTION_PROMPT,
}


class MemoryExtractor:
    """模型辅助记忆提取器。

    通过 S4（GatewayRouter）调用 LLM，从对话内容中提取多类型记忆。
    """

    def __init__(self, gateway: GatewayRouter, model: str | None = None) -> None:
        self.gateway = gateway
        self.model = model

    async def extract(
        self,
        conversation: list[dict[str, str]],
        scope: MemoryScope,
        memory_types: list[MemoryType] | None = None,
    ) -> list[MemoryEntry]:
        """从对话中提取记忆。

        对每种指定的认知类型分别调用 LLM 提取。

        Args:
            conversation: 对话消息列表（role + content 格式）。
            scope: 提取的记忆归属作用域。
            memory_types: 需要提取的认知类型列表，默认全部三种。

        Returns:
            提取出的记忆条目列表。
        """
        if memory_types is None:
            memory_types = [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.PROCEDURAL]

        conversation_text = self.format_conversation(conversation)
        all_entries: list[MemoryEntry] = []

        for mem_type in memory_types:
            if mem_type not in EXTRACTION_PROMPTS:
                continue
            entries = await self.extract_type(conversation_text, scope, mem_type)
            all_entries.extend(entries)

        emit_metric(
            "memory_extraction_total",
            float(len(all_entries)),
            {"scope": scope.to_string()},
            "counter",
        )
        return all_entries

    async def extract_type(
        self,
        conversation_text: str,
        scope: MemoryScope,
        memory_type: MemoryType,
    ) -> list[MemoryEntry]:
        """提取单一认知类型的记忆。

        Args:
            conversation_text: 格式化后的对话文本。
            scope: 记忆作用域。
            memory_type: 目标认知类型。

        Returns:
            提取出的记忆条目列表。
        """
        system_prompt = EXTRACTION_PROMPTS[memory_type]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conversation_text},
        ]

        response = await chat(self.gateway, messages, model=self.model)
        raw_text = response.content or "[]"

        items = self.parse_extraction_response(raw_text)
        entries: list[MemoryEntry] = []
        for item in items:
            entry = self.build_entry(item, scope, memory_type)
            if entry is not None:
                entries.append(entry)

        log.info(
            "记忆提取完成",
            memory_type=memory_type.value,
            count=len(entries),
        )
        return entries

    @staticmethod
    def format_conversation(conversation: list[dict[str, str]]) -> str:
        """格式化对话为提取用文本。"""
        lines: list[str] = []
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    @staticmethod
    def parse_extraction_response(raw_text: str) -> list[dict[str, Any]]:
        """解析 LLM 提取响应为结构化列表。

        容错处理：如果 JSON 解析失败，返回空列表。
        """
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, ValueError):
            log.warning("记忆提取响应解析失败", raw_text=raw_text[:200])
            return []

    @staticmethod
    def build_entry(
        item: dict[str, Any],
        scope: MemoryScope,
        memory_type: MemoryType,
    ) -> MemoryEntry | None:
        """从提取结果构建记忆条目。

        Args:
            item: 单条提取结果字典。
            scope: 记忆作用域。
            memory_type: 认知类型。

        Returns:
            记忆条目，若 content 为空则返回 None。
        """
        content = item.get("content", "").strip()
        if not content:
            return None

        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        confidence = item.get("confidence", 0.8)
        if not isinstance(confidence, (int, float)):
            confidence = 0.8
        confidence = max(0.0, min(1.0, float(confidence)))

        return MemoryEntry(
            memory_type=memory_type,
            scope=scope,
            content=content,
            summary=content[:150],
            tags=tags,
            confidence=confidence,
            metadata={
                "source": "extraction",
                "extraction_type": memory_type.value,
            },
        )
