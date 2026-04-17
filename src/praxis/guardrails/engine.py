"""三层护栏架构。

输入护栏 check_input、工具护栏 check_tool_call、输出护栏 check_output。
绊线触发时返回 block + tripwire 标记。
"""

from typing import Any

from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import RuleEngine, RuleTarget
from praxis.models.guardrails import GuardrailVerdict, VerdictType
from praxis.models.telemetry import AuditEvent
from praxis.models.tools import ToolMetadata
from praxis.telemetry.audit import record_audit


class GuardrailEngine:
    """三层护栏引擎。

    统一管理输入/工具/输出护栏，聚合规则引擎和权限管理器。
    """

    def __init__(
        self,
        rule_engine: RuleEngine,
        permission_manager: PermissionManager,
    ) -> None:
        self.rule_engine = rule_engine
        self.permission_manager = permission_manager

    async def check_input(self, user_message: str) -> GuardrailVerdict:
        """输入护栏：检测提示注入、恶意指令。

        Args:
            user_message: 用户输入消息文本。

        Returns:
            裁决结果（pass 或 block）。
        """
        verdict = self.rule_engine.evaluate(RuleTarget.INPUT, user_message)
        await self.audit_verdict("check_input", verdict)
        return verdict

    async def check_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        metadata: ToolMetadata,
    ) -> GuardrailVerdict:
        """工具护栏：基于权限和规则裁决工具调用。

        先检查权限管理器的裁决，如果被 deny 则直接返回。
        否则检查规则引擎对工具调用内容的检测。

        Args:
            tool_name: 工具名称。
            arguments: 工具参数。
            metadata: 工具元数据。

        Returns:
            裁决结果（auto_approve/confirm/deny）。
        """
        perm_verdict = self.permission_manager.check_permission(
            tool_name, metadata, arguments
        )

        if perm_verdict.verdict == VerdictType.DENY:
            await self.audit_verdict("check_tool_call", perm_verdict, tool_name=tool_name)
            return perm_verdict

        content = f"tool={tool_name} args={arguments}"
        rule_verdict = self.rule_engine.evaluate(
            RuleTarget.TOOL_CALL,
            content,
            context={"tool_name": tool_name, "metadata": metadata.model_dump()},
        )

        if rule_verdict.verdict in (VerdictType.BLOCK, VerdictType.DENY):
            await self.audit_verdict("check_tool_call", rule_verdict, tool_name=tool_name)
            return rule_verdict

        await self.audit_verdict("check_tool_call", perm_verdict, tool_name=tool_name)
        return perm_verdict

    async def check_output(self, assistant_response: str) -> GuardrailVerdict:
        """输出护栏：检测敏感信息泄露、内容安全。

        Args:
            assistant_response: 助手响应文本。

        Returns:
            裁决结果（pass 或 block）。
        """
        verdict = self.rule_engine.evaluate(RuleTarget.OUTPUT, assistant_response)
        await self.audit_verdict("check_output", verdict)
        return verdict

    def register_rule(self, rule: Any) -> None:
        """注册自定义护栏规则。"""
        self.rule_engine.register_rule(rule)

    @staticmethod
    async def audit_verdict(
        operation: str,
        verdict: GuardrailVerdict,
        tool_name: str | None = None,
    ) -> None:
        """将裁决记录到 S2 审计持久化通道。"""
        details: dict[str, Any] = {
            "operation": operation,
            "verdict": verdict.verdict.value,
            "reason": verdict.reason,
            "tripwire": verdict.tripwire,
        }
        if tool_name:
            details["tool_name"] = tool_name
        if verdict.rule_name:
            details["rule_name"] = verdict.rule_name

        await record_audit(AuditEvent(
            event_type="guardrail_verdict",
            component="guardrails",
            action=operation,
            details=details,
        ))
