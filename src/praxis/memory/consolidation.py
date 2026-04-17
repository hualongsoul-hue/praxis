"""记忆整合（Consolidation）。

新提取的记忆先经语义搜索匹配已有记忆，LLM 评估做出 ADD/UPDATE/NOOP 决策。
冲突解决：优先最新，旧标 INACTIVE。语义去重。失败保守 ADD。
"""

import json
from typing import Any

from praxis.gateway.chat import chat
from praxis.gateway.router import GatewayRouter
from praxis.memory.retention import RetentionManager
from praxis.memory.vector_store import VectorStore
from praxis.models.memory import (
    ConsolidationAction,
    MemoryEntry,
    MemoryScope,
)
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("memory.consolidation")

CONSOLIDATION_PROMPT = (
    "你是一个记忆整合专家。判断新记忆与已有记忆的关系。\n"
    "以 JSON 格式输出，包含以下字段：\n"
    '- "action": "add"（新信息）、"update"（更新已有）或 "noop"（冗余/重复）\n'
    '- "reasoning": 判断理由（简短说明）\n'
    '- "merged_content": 如果 action 为 "update"，提供合并后的内容；否则留空字符串\n'
    "仅输出 JSON，不添加其他文字。"
)


class ConsolidationResult:
    """整合决策结果。"""

    def __init__(
        self,
        action: ConsolidationAction,
        reasoning: str,
        merged_content: str = "",
        matched_id: str | None = None,
    ) -> None:
        self.action = action
        self.reasoning = reasoning
        self.merged_content = merged_content
        self.matched_id = matched_id


class MemoryConsolidator:
    """记忆整合器。

    协调 VectorStore（语义搜索）、RetentionManager（版本管理）
    和 GatewayRouter（LLM 评估）。
    """

    def __init__(
        self,
        gateway: GatewayRouter,
        vector_store: VectorStore,
        retention: RetentionManager,
        model: str | None = None,
        similarity_threshold: float = 0.75,
    ) -> None:
        self.gateway = gateway
        self.vector_store = vector_store
        self.retention = retention
        self.model = model
        self.similarity_threshold = similarity_threshold

    async def consolidate(self, new_entry: MemoryEntry) -> ConsolidationResult:
        """对单条新记忆执行整合决策。

        流程：
        1. 语义搜索匹配已有记忆
        2. 若无相似记忆 → ADD
        3. 若有相似记忆 → LLM 评估 → ADD/UPDATE/NOOP

        Args:
            new_entry: 新提取的记忆条目。

        Returns:
            整合决策结果。
        """
        similar = await self.vector_store.search(
            query=new_entry.content,
            scopes=[new_entry.scope],
            memory_type=new_entry.memory_type,
            top_k=3,
            min_score=self.similarity_threshold,
        )

        if not similar:
            await self.vector_store.add(new_entry)
            emit_metric("memory_consolidation", 1.0, {"action": "add"}, "counter")
            log.info("整合决策: ADD（无相似记忆）", memory_id=new_entry.memory_id)
            return ConsolidationResult(
                action=ConsolidationAction.ADD,
                reasoning="无相似已有记忆，新增存储",
            )

        best_match, best_score = similar[0]
        decision = await self.evaluate(new_entry, best_match, best_score)

        if decision.action == ConsolidationAction.ADD:
            await self.vector_store.add(new_entry)
        elif decision.action == ConsolidationAction.UPDATE:
            updated = MemoryEntry(
                memory_type=new_entry.memory_type,
                scope=new_entry.scope,
                content=decision.merged_content or new_entry.content,
                summary=(decision.merged_content or new_entry.content)[:150],
                tags=list(set(best_match.tags + new_entry.tags)),
                confidence=max(best_match.confidence, new_entry.confidence),
                metadata={
                    **best_match.metadata,
                    **new_entry.metadata,
                    "source": "consolidation_update",
                },
            )
            await self.retention.supersede(best_match, updated, decision.reasoning)
            await self.vector_store.remove(best_match.memory_id)
            await self.vector_store.add(updated)
            decision.matched_id = best_match.memory_id

        emit_metric(
            "memory_consolidation",
            1.0,
            {"action": decision.action.value},
            "counter",
        )
        log.info(
            "整合决策",
            action=decision.action.value,
            memory_id=new_entry.memory_id,
            matched_id=decision.matched_id,
        )
        return decision

    async def evaluate(
        self,
        new_entry: MemoryEntry,
        existing: MemoryEntry,
        similarity: float,
    ) -> ConsolidationResult:
        """通过 LLM 评估新旧记忆关系。

        Args:
            new_entry: 新记忆。
            existing: 已有最相似记忆。
            similarity: 语义相似度。

        Returns:
            整合决策。失败时保守返回 ADD。
        """
        user_content = (
            f"语义相似度: {similarity:.2f}\n\n"
            f"已有记忆:\n{existing.content}\n\n"
            f"新记忆:\n{new_entry.content}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": CONSOLIDATION_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await chat(self.gateway, messages, model=self.model)
            raw_text = response.content or ""
            return self.parse_decision(raw_text)
        except Exception as exc:
            log.warning(
                "整合评估失败，保守 ADD",
                error=str(exc),
                memory_id=new_entry.memory_id,
            )
            return ConsolidationResult(
                action=ConsolidationAction.ADD,
                reasoning=f"LLM 评估失败，保守新增: {exc}",
            )

    @staticmethod
    def parse_decision(raw_text: str) -> ConsolidationResult:
        """解析 LLM 整合决策。

        解析失败时返回保守 ADD。
        """
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            data = json.loads(text)
            action_str = data.get("action", "add").lower()
            action = ConsolidationAction(action_str)
            return ConsolidationResult(
                action=action,
                reasoning=data.get("reasoning", ""),
                merged_content=data.get("merged_content", ""),
            )
        except (json.JSONDecodeError, ValueError):
            log.warning("整合决策解析失败，保守 ADD", raw_text=raw_text[:200])
            return ConsolidationResult(
                action=ConsolidationAction.ADD,
                reasoning=f"无法解析 LLM 响应: {raw_text[:100]}",
            )

    async def consolidate_batch(
        self,
        entries: list[MemoryEntry],
    ) -> list[ConsolidationResult]:
        """批量整合多条记忆。

        Args:
            entries: 待整合的记忆列表。

        Returns:
            每条记忆的整合结果。
        """
        results: list[ConsolidationResult] = []
        for entry in entries:
            result = await self.consolidate(entry)
            results.append(result)
        return results
