"""梦境整理（Dream Consolidation）。

借鉴人类 REM 睡眠概念，在后台定期整理记忆：
时间锚定、矛盾消解、陈旧清理、索引精简。
"""

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from praxis.gateway.chat import chat
from praxis.gateway.router import GatewayRouter
from praxis.memory.lifecycle import LifecycleManager
from praxis.memory.scope import ScopedMemoryStore
from praxis.models.memory import (
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
)
from praxis.persistence.store import PersistenceStore
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("memory.dream")

DREAM_NAMESPACE = "dream_meta"

DREAM_SYSTEM_PROMPT = (
    "你是一个记忆整理专家。审查以下记忆条目列表，执行以下任务：\n"
    "1. **时间锚定**：将模糊时间引用替换为具体日期\n"
    "2. **矛盾消解**：检测互相矛盾的记忆，标记需要处理的条目\n"
    "3. **陈旧清理**：标记引用已不存在的文件、已完成的任务等过时记忆\n"
    "4. **索引精简**：建议合并冗余条目\n\n"
    "以 JSON 格式输出，包含：\n"
    '- "anchored": [{\"memory_id\": \"...\", \"updated_content\": \"...\"}] 时间锚定更新\n'
    '- "conflicts": [{\"ids\": [\"id1\", \"id2\"], \"resolution\": \"...\"}] 矛盾对\n'
    '- "stale": [\"id1\", \"id2\"] 陈旧条目 ID 列表\n'
    '- "merge_suggestions": [{\"ids\": [\"id1\", \"id2\"], \"merged_content\": \"...\"}]\n'
    '- "summary": 整理摘要文字\n'
    "仅输出 JSON，不添加其他文字。"
)


class DreamReport(BaseModel):
    """梦境整理报告。"""

    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    total_reviewed: int = 0
    anchored_count: int = 0
    conflicts_resolved: int = 0
    stale_marked: int = 0
    merged_count: int = 0
    summary: str = ""


class DreamConsolidator:
    """梦境整理器。

    通过 S4 驱动 LLM 执行记忆整理，
    整理报告通过 S2 记录。
    """

    def __init__(
        self,
        gateway: GatewayRouter,
        scoped_store: ScopedMemoryStore,
        lifecycle: LifecycleManager,
        meta_store: PersistenceStore,
        model: str | None = None,
        min_hours_since_last: float = 24.0,
        min_sessions: int = 5,
    ) -> None:
        self.gateway = gateway
        self.scoped_store = scoped_store
        self.lifecycle = lifecycle
        self.meta_store = meta_store
        self.model = model
        self.min_hours_since_last = min_hours_since_last
        self.min_sessions = min_sessions

    async def should_run(self, session_count: int) -> bool:
        """检查是否满足触发条件。

        Args:
            session_count: 自上次整理以来的新会话数。

        Returns:
            是否应执行整理。
        """
        meta = await self.meta_store.load(DREAM_NAMESPACE, "last_run")
        if meta is None:
            return session_count >= self.min_sessions

        last_run_str = meta.get("completed_at", "")
        if not last_run_str:
            return session_count >= self.min_sessions

        last_run = datetime.fromisoformat(last_run_str)
        now = datetime.now(timezone.utc)
        hours_since = (now - last_run).total_seconds() / 3600.0

        return hours_since >= self.min_hours_since_last and session_count >= self.min_sessions

    async def run_dream(
        self,
        scopes: list[MemoryScope],
    ) -> DreamReport:
        """执行梦境整理。

        Args:
            scopes: 需要整理的作用域列表。

        Returns:
            整理报告。
        """
        report = DreamReport()
        all_entries: list[MemoryEntry] = []

        for scope in scopes:
            entries = await self.scoped_store.list_scope(scope)
            all_entries.extend(entries)

        report.total_reviewed = len(all_entries)

        if not all_entries:
            report.completed_at = datetime.now(timezone.utc)
            report.summary = "无记忆条目需要整理"
            return report

        entries_text = self.format_entries(all_entries)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": DREAM_SYSTEM_PROMPT},
            {"role": "user", "content": entries_text},
        ]

        response = await chat(self.gateway, messages, model=self.model)
        raw_text = response.content or "{}"
        actions = self.parse_dream_response(raw_text)

        entry_map = {e.memory_id: e for e in all_entries}

        report.anchored_count = await self.apply_anchoring(
            actions.get("anchored", []), entry_map,
        )
        report.stale_marked = await self.apply_stale_marking(
            actions.get("stale", []), entry_map,
        )
        report.conflicts_resolved = len(actions.get("conflicts", []))
        report.merged_count = await self.apply_merges(
            actions.get("merge_suggestions", []), entry_map,
        )
        report.summary = actions.get("summary", "")
        report.completed_at = datetime.now(timezone.utc)

        await self.save_run_meta(report)

        emit_metric("dream_anchored", float(report.anchored_count), {}, "counter")
        emit_metric("dream_stale", float(report.stale_marked), {}, "counter")
        emit_metric("dream_merged", float(report.merged_count), {}, "counter")

        log.info(
            "梦境整理完成",
            reviewed=report.total_reviewed,
            anchored=report.anchored_count,
            stale=report.stale_marked,
            merged=report.merged_count,
        )
        return report

    async def apply_anchoring(
        self,
        anchored: list[dict[str, str]],
        entry_map: dict[str, MemoryEntry],
    ) -> int:
        """应用时间锚定更新。"""
        count = 0
        for item in anchored:
            mid = item.get("memory_id", "")
            new_content = item.get("updated_content", "")
            entry = entry_map.get(mid)
            if entry and new_content:
                entry.content = new_content
                entry.summary = new_content[:150]
                entry.updated_at = datetime.now(timezone.utc)
                await self.scoped_store.update(entry)
                count += 1
        return count

    async def apply_stale_marking(
        self,
        stale_ids: list[str],
        entry_map: dict[str, MemoryEntry],
    ) -> int:
        """标记陈旧记忆为 INACTIVE。"""
        count = 0
        for mid in stale_ids:
            entry = entry_map.get(mid)
            if entry:
                await self.lifecycle.mark_inactive(entry, "梦境整理: 陈旧记忆")
                count += 1
        return count

    async def apply_merges(
        self,
        merge_suggestions: list[dict[str, Any]],
        entry_map: dict[str, MemoryEntry],
    ) -> int:
        """应用合并建议。"""
        count = 0
        for suggestion in merge_suggestions:
            ids = suggestion.get("ids", [])
            merged_content = suggestion.get("merged_content", "")
            if len(ids) < 2 or not merged_content:
                continue

            primary = entry_map.get(ids[0])
            if primary is None:
                continue

            merged = MemoryEntry(
                memory_type=primary.memory_type,
                scope=primary.scope,
                content=merged_content,
                summary=merged_content[:150],
                tags=primary.tags,
                metadata={**primary.metadata, "source": "dream_merge"},
            )

            for mid in ids:
                old = entry_map.get(mid)
                if old:
                    await self.lifecycle.mark_inactive(old, "梦境整理: 合并")

            await self.scoped_store.save(merged)
            count += 1
        return count

    async def save_run_meta(self, report: DreamReport) -> None:
        """保存整理元数据。"""
        await self.meta_store.save(
            DREAM_NAMESPACE,
            "last_run",
            report.model_dump(mode="json"),
        )

    @staticmethod
    def format_entries(entries: list[MemoryEntry]) -> str:
        """格式化记忆条目列表为 LLM 可读文本。"""
        lines: list[str] = []
        for entry in entries:
            lines.append(
                f"[{entry.memory_id}] ({entry.memory_type.value}) "
                f"创建于 {entry.created_at.isoformat()}: {entry.content}"
            )
        return "\n".join(lines)

    @staticmethod
    def parse_dream_response(raw_text: str) -> dict[str, Any]:
        """解析梦境整理 LLM 响应。"""
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            return {}
        except (json.JSONDecodeError, ValueError):
            log.warning("梦境整理响应解析失败", raw_text=raw_text[:200])
            return {}
