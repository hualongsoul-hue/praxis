"""输出解析。

依赖 LLM 原生 tool_calls 结构化输出，
判断逻辑：有 tool_calls → 执行 → 继续；无 → 最终响应 → 退出。
并行工具调用批量解析，Handoff 请求检测。
"""

from typing import Any

from json_repair import repair_json
from pydantic import BaseModel

from praxis.models.responses import ModelResponse
from praxis.models.tools import FunctionCall, ToolCall
from praxis.telemetry.logger import get_logger

log = get_logger("orchestrator.parser")

HANDOFF_FUNCTION_PREFIX = "handoff_to_"


class ParsedOutput:
    """解析后的 LLM 输出。"""

    __slots__ = ("content", "tool_calls", "is_final", "handoff_target")

    def __init__(
        self,
        content: str,
        tool_calls: list[ToolCall],
        is_final: bool,
        handoff_target: str | None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.is_final = is_final
        self.handoff_target = handoff_target


class OutputParser:
    """LLM 输出解析器。"""

    def parse(self, response: ModelResponse) -> ParsedOutput:
        """解析 LLM 响应。

        Args:
            response: S4 返回的模型响应。

        Returns:
            解析结果。
        """
        content = response.content or ""
        tool_calls = response.tool_calls or []

        is_final = len(tool_calls) == 0
        handoff_target = self.detect_handoff(tool_calls)

        if handoff_target:
            log.info("检测到 Handoff 请求", target=handoff_target)

        log.info(
            "LLM 输出已解析",
            has_content=bool(content),
            tool_call_count=len(tool_calls),
            is_final=is_final,
        )

        return ParsedOutput(
            content=content,
            tool_calls=tool_calls,
            is_final=is_final,
            handoff_target=handoff_target,
        )

    @staticmethod
    def detect_handoff(tool_calls: list[ToolCall]) -> str | None:
        """检测 Handoff 请求。

        Args:
            tool_calls: 工具调用列表。

        Returns:
            目标子代理名称，无则 None。
        """
        for tc in tool_calls:
            if tc.function.name.startswith(HANDOFF_FUNCTION_PREFIX):
                return tc.function.name[len(HANDOFF_FUNCTION_PREFIX):]
        return None

    @staticmethod
    def parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
        """解析工具调用参数 JSON 字符串。

        Args:
            raw_arguments: JSON 格式参数字符串。

        Returns:
            参数字典。
        """
        if not raw_arguments:
            return {}
        result = repair_json(raw_arguments, return_objects=True)
        return result if isinstance(result, dict) else {}

    @staticmethod
    def extract_schema_response(
        content: str,
        schema_model: type[BaseModel],
    ) -> BaseModel | None:
        """提取 Pydantic Schema 约束的结构化响应。

        Args:
            content: LLM 响应内容。
            schema_model: 目标 Pydantic 模型类。

        Returns:
            解析后的模型实例，失败则 None。
        """
        if not content:
            return None
        data = repair_json(content, return_objects=True)
        return schema_model.model_validate(data)
