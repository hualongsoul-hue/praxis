"""护栏规则引擎。

统一规则定义（Pydantic 模型）、内置规则集、自定义规则扩展、
有序评估 + 短路逻辑、裁决审计日志（S2）。
"""

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from praxis.models.guardrails import GuardrailVerdict, VerdictType


class RuleTarget(StrEnum):
    """规则适用目标。"""

    INPUT = "input"
    TOOL_CALL = "tool_call"
    OUTPUT = "output"


class GuardrailRule(BaseModel):
    """护栏规则定义。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    target: RuleTarget
    enabled: bool = True
    priority: int = Field(default=100, description="优先级，数值越小越先评估")
    patterns: tuple[str, ...] = Field(default_factory=tuple, description="正则表达式检测模式")
    verdict: VerdictType = VerdictType.BLOCK
    tripwire: bool = False


BUILTIN_INPUT_RULES: tuple[GuardrailRule, ...] = (
    GuardrailRule(
        name="prompt_injection_basic",
        description="检测基础提示注入模式",
        target=RuleTarget.INPUT,
        priority=10,
        patterns=(
            r"(?i)ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
            r"(?i)you\s+are\s+now\s+(a|an|the)\s+",
            r"(?i)system\s*:\s*",
            r"(?i)jailbreak",
            r"(?i)do\s+anything\s+now",
        ),
        verdict=VerdictType.BLOCK,
        tripwire=True,
    ),
)

BUILTIN_OUTPUT_RULES: tuple[GuardrailRule, ...] = (
    GuardrailRule(
        name="sensitive_data_leak",
        description="检测输出中的敏感信息泄露",
        target=RuleTarget.OUTPUT,
        priority=10,
        patterns=(
            # 键名上下文 + 取值，避免对任意长字符串误报
            r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?\w{16,}",
            r"(?i)(password|passwd)\s*[:=]\s*['\"]?\S{6,}",
            # 具体的高置信度凭证格式（私钥块 / 主流云厂商密钥），而非泛 base64
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
            r"\bAKIA[0-9A-Z]{16}\b",                     # AWS Access Key ID
            r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",           # GitHub token
            r"\bsk-[A-Za-z0-9]{20,}\b",                  # OpenAI 风格密钥
            r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",         # Slack token
        ),
        verdict=VerdictType.BLOCK,
        tripwire=False,
    ),
)


class RuleEngine:
    """护栏规则引擎。

    管理规则注册、有序评估和短路逻辑。
    """

    def __init__(self) -> None:
        self.rules: list[GuardrailRule] = []

    def register_rule(self, rule: GuardrailRule) -> None:
        """注册护栏规则。"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)

    def register_builtin_rules(self) -> None:
        """注册所有内置规则。"""
        for rule in BUILTIN_INPUT_RULES:
            self.register_rule(rule)
        for rule in BUILTIN_OUTPUT_RULES:
            self.register_rule(rule)

    def evaluate(
        self,
        target: RuleTarget,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> GuardrailVerdict:
        """对指定目标内容执行规则评估。

        规则按优先级有序评估，首个 deny/block 触发短路。

        Args:
            target: 规则适用目标（input/tool_call/output）。
            content: 待检测的文本内容。
            context: 额外上下文信息（工具元数据等）。

        Returns:
            最终裁决结果。
        """
        for rule in self.rules:
            if not rule.enabled or rule.target != target:
                continue

            for pattern in rule.patterns:
                if re.search(pattern, content):
                    return GuardrailVerdict(
                        verdict=rule.verdict,
                        reason=f"规则 '{rule.name}' 匹配: {rule.description}",
                        tripwire=rule.tripwire,
                        rule_name=rule.name,
                    )

        if target == RuleTarget.INPUT:
            return GuardrailVerdict(verdict=VerdictType.PASS, reason="输入检测通过")
        if target == RuleTarget.OUTPUT:
            return GuardrailVerdict(verdict=VerdictType.PASS, reason="输出检测通过")
        return GuardrailVerdict(verdict=VerdictType.PASS, reason="检测通过")
