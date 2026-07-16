"""结果聚合。

聚合多个子代理输出，检测冲突并标记。
"""

from praxis.models.subagent import ConflictMarker, SubagentResult
from praxis.telemetry.logger import get_logger

log = get_logger("subagent.aggregation")


class ResultAggregator:
    """子代理结果聚合器。

    综合分析多个子代理的输出，检测冲突，生成聚合摘要。
    """

    def aggregate(self, results: list[SubagentResult]) -> dict[str, object]:
        """聚合多个子代理结果。

        Args:
            results: 子代理结果列表。

        Returns:
            聚合结果，包含 combined_summary、key_findings、conflicts、metadata。
        """
        summaries: list[str] = []
        all_findings: list[str] = []
        total_tokens = 0
        total_turns = 0

        for r in results:
            if r.summary:
                summaries.append(f"[{r.subagent_id}] {r.summary}")
            all_findings.extend(r.key_findings)
            total_tokens += r.total_tokens
            total_turns += r.total_turns

        conflicts = self.detect_conflicts(results)
        if conflicts:
            log.warning("检测到结果冲突", conflict_count=len(conflicts))

        return {
            "combined_summary": "\n\n".join(summaries),
            "key_findings": all_findings,
            "conflicts": [c.model_dump() for c in conflicts],
            "total_tokens": total_tokens,
            "total_turns": total_turns,
            "subagent_count": len(results),
        }

    @staticmethod
    def detect_conflicts(results: list[SubagentResult]) -> list[ConflictMarker]:
        """检测多个子代理结果间的冲突。

        通过元数据中的 answer 字段检测不同结论。

        Args:
            results: 子代理结果列表。

        Returns:
            冲突标记列表。
        """
        conflicts: list[ConflictMarker] = []

        # 按元数据中的 answer_key 分组检测
        answer_map: dict[str, list[tuple[str, str]]] = {}
        for r in results:
            for key, value in r.metadata.items():
                if key.startswith("answer_"):
                    answer_map.setdefault(key, []).append((r.subagent_id, str(value)))

        for key, entries in answer_map.items():
            unique_values = {entry[1] for entry in entries}
            if len(unique_values) > 1:
                conflicts.append(ConflictMarker(
                    field=key,
                    values=list(unique_values),
                    subagent_ids=[entry[0] for entry in entries],
                    description=f"子代理对 {key} 给出不同结论",
                ))

        return conflicts
