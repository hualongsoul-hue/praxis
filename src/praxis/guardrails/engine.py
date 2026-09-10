"""三层护栏架构。

输入护栏 check_input、工具护栏 check_tool_call、输出护栏 check_output。
绊线触发时返回 block + tripwire 标记。
"""

from typing import Any

from praxis.config.schemas import GuardrailsConfig
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import GuardrailRule, RuleEngine, RuleTarget
from praxis.models.guardrails import GuardrailVerdict, VerdictType
from praxis.models.telemetry import AuditEvent
from praxis.models.tools import ToolMetadata
from praxis.protocols import AuditSink
from praxis.telemetry.audit import NullAuditSink


class GuardrailEngine:
    """三层护栏引擎。

    统一管理输入/工具/输出护栏，聚合规则引擎和权限管理器。
    """

    def __init__(
        self,
        rule_engine: RuleEngine,
        permission_manager: PermissionManager,
        input_enabled: bool = True,
        output_enabled: bool = True,
        audit_sink: AuditSink | None = None,
    ) -> None:
        self.rule_engine = rule_engine
        self.permission_manager = permission_manager
        self.input_enabled = input_enabled
        self.output_enabled = output_enabled
        self.audit_sink = audit_sink or NullAuditSink()

    def for_session(self) -> "GuardrailEngine":
        """Copy immutable rules and policy, never inheriting temporary grants."""
        rule_engine = RuleEngine()
        rule_engine.rules = list(self.rule_engine.rules)
        return GuardrailEngine(
            rule_engine=rule_engine,
            permission_manager=PermissionManager(self.permission_manager.policy),
            input_enabled=self.input_enabled,
            output_enabled=self.output_enabled,
            audit_sink=self.audit_sink,
        )

    async def check_input(self, user_message: str) -> GuardrailVerdict:
        """输入护栏：检测提示注入、恶意指令。

        注意：内置检测为基础启发式（有限正则），可拦截朴素攻击但非完整防护；
        生产环境应叠加更强的检测（如 LLM 判别或专用服务）。

        Args:
            user_message: 用户输入消息文本。

        Returns:
            裁决结果（pass 或 block）。
        """
        if not self.input_enabled:
            return GuardrailVerdict(verdict=VerdictType.PASS, reason="输入护栏已禁用")
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

        # 规则裁决为 BLOCK/DENY，或虽放行但触发绊线，都应返回规则裁决，
        # 避免 ALLOW+tripwire 时绊线标记被 perm_verdict 吞掉。
        if rule_verdict.verdict in (VerdictType.BLOCK, VerdictType.DENY) or rule_verdict.tripwire:
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
        if not self.output_enabled:
            return GuardrailVerdict(verdict=VerdictType.PASS, reason="输出护栏已禁用")
        verdict = self.rule_engine.evaluate(RuleTarget.OUTPUT, assistant_response)
        await self.audit_verdict("check_output", verdict)
        return verdict

    def register_rule(self, rule: GuardrailRule) -> None:
        """注册自定义护栏规则。"""
        self.rule_engine.register_rule(rule)

    async def audit_verdict(
        self,
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

        await self.audit_sink.record(AuditEvent(
            event_type="guardrail_verdict",
            component="guardrails",
            action=operation,
            details=details,
        ))


def build_guardrail_engine(
    config: GuardrailsConfig,
    *,
    audit_sink: AuditSink | None = None,
) -> GuardrailEngine:
    """从 S8 配置装配护栏引擎（消费 default_permission / permissions_file / 启停开关）。

    - 注册内置规则集（提示注入/敏感信息）。
    - 权限：若配置了 permissions_file 则加载声明式权限（YAML），否则以
      default_permission 作为默认裁决。
    - 输入/输出护栏按 input_guardrails_enabled / output_guardrails_enabled 启停。

    Args:
        config: S8 护栏配置。

    Returns:
        装配完毕的 GuardrailEngine。
    """
    from pathlib import Path

    from praxis.config.loader import load_yaml_config

    rule_engine = RuleEngine()
    rule_engine.register_builtin_rules()

    perm_dict: dict[str, Any] = {"default_permission": config.default_permission}
    if config.permissions_file:
        perm_dict.update(load_yaml_config(Path(config.permissions_file)))
    permission_manager = PermissionManager.from_config_dict(perm_dict)

    return GuardrailEngine(
        rule_engine=rule_engine,
        permission_manager=permission_manager,
        input_enabled=config.input_guardrails_enabled,
        output_enabled=config.output_guardrails_enabled,
        audit_sink=audit_sink,
    )
