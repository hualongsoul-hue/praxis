"""权限分层系统。

三级权限 auto_approve/confirm/deny，默认限制性策略，
YAML 声明式权限配置（按工具名/类别/路径模式），运行时临时授权。
"""

from typing import Any, Literal

from pydantic import ConfigDict, model_validator

from praxis.models.base import SafeBaseModel
from praxis.models.guardrails import GuardrailVerdict, VerdictType
from praxis.models.tools import ToolMetadata

PermissionLevel = Literal[VerdictType.AUTO_APPROVE, VerdictType.CONFIRM, VerdictType.DENY]


class PermissionRule(SafeBaseModel):
    """权限规则条目。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str | None = None
    category: str | None = None
    permission: PermissionLevel = VerdictType.CONFIRM

    @model_validator(mode="after")
    def validate_selector(self) -> "PermissionRule":
        selectors = [value for value in (self.tool_name, self.category) if value is not None]
        if len(selectors) != 1 or not selectors[0] or selectors[0] != selectors[0].strip():
            raise ValueError("权限规则必须指定一个 tool_name 或 category")
        return self


class PermissionPolicy(SafeBaseModel):
    """权限策略配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_permission: PermissionLevel = VerdictType.CONFIRM
    rules: tuple[PermissionRule, ...] = ()


class PermissionManager:
    """权限管理器。

    支持声明式配置 + 运行时临时授权。
    """

    def __init__(self, policy: PermissionPolicy | None = None) -> None:
        self.policy = policy or PermissionPolicy()
        self.temporary_grants: dict[str, PermissionLevel] = {}

    def check_permission(
        self,
        tool_name: str,
        metadata: ToolMetadata,
        arguments: dict[str, Any] | None = None,
    ) -> GuardrailVerdict:
        """检查工具调用权限。

        评估优先级：临时授权 > 工具名规则 > 类别规则 > 元数据权限级别 > 默认策略。

        Args:
            tool_name: 工具名称。
            metadata: 工具元数据。
            arguments: 工具参数（用于路径模式匹配等）。

        Returns:
            权限裁决结果。
        """
        if tool_name in self.temporary_grants:
            verdict = self.temporary_grants[tool_name]
            return GuardrailVerdict(
                verdict=verdict,
                reason=f"临时授权: {tool_name} -> {verdict.value}",
            )

        for rule in self.policy.rules:
            if rule.tool_name and rule.tool_name == tool_name:
                return GuardrailVerdict(
                    verdict=rule.permission,
                    reason=f"工具名规则匹配: {tool_name}",
                )
        for rule in self.policy.rules:
            if rule.category and rule.category == metadata.category:
                return GuardrailVerdict(
                    verdict=rule.permission,
                    reason=f"类别规则匹配: {metadata.category}",
                )

        # 工具显式声明权限级别时优先；未声明（None）则回落到策略默认权限。
        if metadata.permission_level is not None:
            meta_permission = VerdictType(metadata.permission_level)
            return GuardrailVerdict(
                verdict=meta_permission,
                reason=f"元数据权限级别: {meta_permission.value}",
            )
        return GuardrailVerdict(
            verdict=self.policy.default_permission,
            reason=f"默认策略: {self.policy.default_permission.value}",
        )

    def grant_temporary(self, tool_name: str, permission: PermissionLevel) -> None:
        """运行时临时授权。"""
        rule = PermissionRule(tool_name=tool_name, permission=permission)
        self.temporary_grants[tool_name] = rule.permission

    def revoke_temporary(self, tool_name: str) -> bool:
        """撤销临时授权。返回是否成功撤销。"""
        if tool_name in self.temporary_grants:
            del self.temporary_grants[tool_name]
            return True
        return False

    def clear_temporary(self) -> None:
        """清除所有临时授权。"""
        self.temporary_grants.clear()

    @staticmethod
    def from_config_dict(config: dict[str, Any]) -> "PermissionManager":
        """从配置字典创建权限管理器。

        Args:
            config: 权限配置字典，格式：
                default_permission: "confirm"
                rules:
                  - tool_name: "read_file"
                    permission: "auto_approve"
                  - category: "file_ops"
                    permission: "confirm"
        """
        return PermissionManager(PermissionPolicy.model_validate(config))
