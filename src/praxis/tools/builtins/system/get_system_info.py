"""内置工具：get_system_info。

返回当前系统的基本信息。
"""

import os
import platform
from datetime import datetime
from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata

DEFINITION = ToolDefinition(
    name="get_system_info",
    description="返回当前操作系统、CPU 架构、工作目录、系统时间等信息。",
    parameters={
        "type": "object",
        "properties": {},
    },
    metadata=ToolMetadata(
        category="system",
        permission_level="auto_approve",
        readonly=True,
        tags=["system", "info"],
    ),
)


async def handle(args: dict[str, Any]) -> str:
    """获取系统信息。"""
    lines = [
        f"OS: {platform.system()} {platform.release()}",
        f"Platform: {platform.platform()}",
        f"Architecture: {platform.machine()}",
        f"CWD: {os.getcwd()}",
        f"CPU count: {os.cpu_count()}",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    return "\n".join(lines)
