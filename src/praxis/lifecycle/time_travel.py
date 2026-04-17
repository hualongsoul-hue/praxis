"""时间旅行调试。

rollback 回退到任意历史检查点，
list_checkpoints 查看历史，
回退后 S6/S7/S11 状态全部恢复，可从回退点重新运行。
"""

from praxis.guardrails.engine import GuardrailEngine
from praxis.lifecycle.checkpoint import CheckpointManager
from praxis.lifecycle.resume import SessionResumer
from praxis.lifecycle.session import Session, SessionFactory
from praxis.models.lifecycle import CheckpointInfo
from praxis.telemetry.logger import get_logger
from praxis.tools.registry import ToolRegistry

log = get_logger("lifecycle.time_travel")


class TimeTravelManager:
    """时间旅行管理器。

    支持回退到任意历史检查点并重新运行。
    """

    def __init__(
        self,
        checkpoint_mgr: CheckpointManager,
        resumer: SessionResumer,
    ) -> None:
        self.checkpoint_mgr = checkpoint_mgr
        self.resumer = resumer

    async def list_checkpoints(self, session_id: str) -> list[CheckpointInfo]:
        """列出会话的所有检查点。

        Args:
            session_id: 会话 ID。

        Returns:
            检查点摘要列表（按时间排序）。
        """
        return await self.checkpoint_mgr.list_checkpoints(session_id)

    async def rollback(
        self,
        session_id: str,
        checkpoint_id: str,
        guardrails: GuardrailEngine,
        registry: ToolRegistry | None = None,
        model: str = "default",
    ) -> Session | None:
        """回退到指定检查点。

        从指定检查点恢复完整状态，返回可重新运行的 Session。

        Args:
            session_id: 会话 ID。
            checkpoint_id: 目标检查点 ID。
            guardrails: 护栏引擎。
            registry: 工具注册表（可选）。
            model: LLM 模型名。

        Returns:
            恢复后的 Session，失败时返回 None。
        """
        session = await self.resumer.resume_session(
            session_id=session_id,
            guardrails=guardrails,
            registry=registry,
            model=model,
            checkpoint_id=checkpoint_id,
        )

        if session is None:
            log.warning(
                "回退失败: 检查点未找到",
                session_id=session_id,
                checkpoint_id=checkpoint_id,
            )
            return None

        # 清除回退点之后的检查点（可选，保留历史分叉能力）
        log.info(
            "时间旅行: 回退完成",
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            turn=session.metadata.total_turns,
        )
        return session

    async def rollback_and_prune(
        self,
        session_id: str,
        checkpoint_id: str,
        guardrails: GuardrailEngine,
        registry: ToolRegistry | None = None,
        model: str = "default",
    ) -> Session | None:
        """回退并清除后续检查点。

        回退到指定点，并删除该点之后的所有检查点。

        Args:
            session_id: 会话 ID。
            checkpoint_id: 目标检查点 ID。
            guardrails: 护栏引擎。
            registry: 工具注册表。
            model: LLM 模型名。

        Returns:
            恢复后的 Session。
        """
        session = await self.rollback(
            session_id, checkpoint_id, guardrails, registry, model
        )
        if session is None:
            return None

        # 获取所有检查点，删除目标之后的
        all_cps = await self.checkpoint_mgr.list_checkpoints(session_id)
        target_found = False
        for cp in all_cps:
            if cp.checkpoint_id == checkpoint_id:
                target_found = True
                continue
            if target_found:
                await self.checkpoint_mgr.delete_checkpoint(
                    session_id, cp.checkpoint_id
                )

        # 更新最新检查点引用
        await self.checkpoint_mgr.store.save(
            "checkpoints",
            f"{session_id}:latest",
            checkpoint_id,
        )

        log.info(
            "时间旅行: 回退并清理完成",
            session_id=session_id,
            checkpoint_id=checkpoint_id,
        )
        return session
