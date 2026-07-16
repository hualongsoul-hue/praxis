"""内置工具：edit_file。

在文件中执行精确字符串替换。
"""

from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.policy import ToolPolicy

DEFINITION = ToolDefinition(
    name="edit_file",
    description="在文件中查找 old_string 并替换为 new_string。old_string 必须在文件中唯一匹配。",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件绝对路径",
            },
            "old_string": {
                "type": "string",
                "description": "要被替换的原始文本（必须精确匹配）",
            },
            "new_string": {
                "type": "string",
                "description": "替换后的文本",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    },
    metadata=ToolMetadata(
        category="file_ops",
        permission_level="confirm",
        readonly=False,
        tags=["file", "edit"],
    ),
)


def create_handler(sandbox: ToolPolicy):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        path = sandbox.check_path(args["file_path"])
        old_string = args["old_string"]
        new_string = args["new_string"]

        if not path.exists():
            return f"文件不存在: {path}"

        content = path.read_text(encoding="utf-8")
        count = content.count(old_string)

        if count == 0:
            return "未找到匹配的文本"
        if count > 1:
            return f"old_string 在文件中匹配 {count} 处，必须唯一匹配"

        new_content = content.replace(old_string, new_string, 1)
        path.write_text(new_content, encoding="utf-8")
        return f"已替换 1 处匹配（文件: {path}）"

    return handle
