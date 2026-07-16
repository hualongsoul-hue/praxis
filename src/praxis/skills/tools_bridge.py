"""技能与工具系统协同。

技能声明工具依赖（激活时确保工具已注册），
技能附带 Python 脚本通过 S5 执行，list_skill_tools 接口。
"""

from pathlib import Path
from typing import Any

from praxis.models.skills import SkillDefinition
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.skills.paths import resolve_skill_path
from praxis.telemetry.logger import get_logger
from praxis.tools.registry import ToolHandler, ToolRegistry

log = get_logger("skills.tools_bridge")

SKILL_TOOL_CATEGORY = "skill_script"


class SkillToolsBridge:
    """技能与工具系统桥接器。

    负责将技能的脚本注册为工具，检查工具依赖。
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.registered_tools: dict[str, list[str]] = {}

    def check_tool_dependencies(self, skill: SkillDefinition) -> list[str]:
        """检查技能的工具依赖是否满足。

        Args:
            skill: 技能定义。

        Returns:
            缺失的工具名列表（空列表表示全部满足）。
        """
        missing: list[str] = []
        for tool_name in skill.metadata.tools:
            if not self.registry.has_tool(tool_name):
                missing.append(tool_name)
        return missing

    def register_skill_scripts(
        self,
        skill: SkillDefinition,
        executor: ToolHandler | None = None,
    ) -> list[str]:
        """将技能附带的脚本注册为工具。

        Args:
            skill: 技能定义。
            executor: 可选的统一脚本执行器。
                     如果不提供，创建默认的脚本读取器。

        Returns:
            注册的工具名列表。
        """
        registered: list[str] = []
        base = Path(skill.base_path)

        for script_rel in skill.scripts:
            script_path = resolve_skill_path(base, script_rel)
            if not script_path.is_file():
                continue

            tool_name = f"skill_{skill.skill_id}_{script_path.stem}"
            handler = executor or self.create_script_reader(script_path)

            definition = ToolDefinition(
                name=tool_name,
                description=f"技能 {skill.metadata.name} 的脚本: {script_rel}",
                parameters={
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "string",
                            "description": "传递给脚本的参数",
                        },
                    },
                },
                metadata=ToolMetadata(
                    category=SKILL_TOOL_CATEGORY,
                    permission_level="confirm",
                    tags=["skill", skill.skill_id],
                ),
            )

            self.registry.register(definition, handler)
            registered.append(tool_name)

        self.registered_tools[skill.skill_id] = registered
        log.info(
            "技能脚本已注册为工具",
            skill_id=skill.skill_id,
            tools_count=len(registered),
        )
        return registered

    def unregister_skill_scripts(self, skill_id: str) -> list[str]:
        """注销技能相关的所有脚本工具。

        Args:
            skill_id: 技能标识符。

        Returns:
            注销的工具名列表。
        """
        tool_names = self.registered_tools.pop(skill_id, [])
        for tool_name in tool_names:
            self.registry.unregister(tool_name)
        return tool_names

    def list_skill_tools(self, skill_id: str) -> list[str]:
        """列出技能附带的可执行脚本工具名。

        Args:
            skill_id: 技能标识符。

        Returns:
            工具名列表。
        """
        return list(self.registered_tools.get(skill_id, []))

    @staticmethod
    def create_script_reader(script_path: Path) -> ToolHandler:
        """创建脚本读取工具处理函数。

        实际执行由 S11 编排循环通过 S5 run_command 完成，
        此处返回脚本内容供 Agent 审查。

        Args:
            script_path: 脚本文件路径。

        Returns:
            异步工具处理函数。
        """
        async def handler(arguments: dict[str, Any]) -> str:
            return script_path.read_text(encoding="utf-8")
        return handler
