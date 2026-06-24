"""输出解析。

依赖 LLM 原生 tool_calls 结构化输出，
判断逻辑：有 tool_calls → 执行 → 继续；无 → 最终响应 → 退出。
并行工具调用批量解析，Handoff 请求检测。
"""

from typing import Any

from json_repair import repair_json
from pydantic import BaseModel

from praxis.models.responses import ModelResponse, ModelResponseChunk, Usage
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


class StreamDelta(BaseModel):
    """一次 ``StreamAccumulator.feed`` 产生的增量信息。"""

    content: str | None = None
    reasoning: str | None = None
    refusal: str | None = None


class StreamAccumulator:
    """流式响应块累积器。

    将 chat_stream 产出的 ModelResponseChunk 增量拼装为
    完整的 ModelResponse，支持 content、reasoning_content、refusal
    与 tool_calls 增量。
    """

    def __init__(self) -> None:
        self.response_id: str = ""
        self.model: str = ""
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.refusal_parts: list[str] = []
        self.tool_call_buffers: dict[int, dict[str, str]] = {}
        self.finish_reason: str | None = None
        self.usage: Usage | None = None
        self.system_fingerprint: str | None = None

    def feed(self, chunk: ModelResponseChunk) -> StreamDelta:
        """喂入一个 chunk，返回本次的 content / reasoning / refusal 增量。"""
        if chunk.id:
            self.response_id = chunk.id
        if chunk.model:
            self.model = chunk.model
        if chunk.finish_reason:
            self.finish_reason = chunk.finish_reason
        if chunk.usage is not None:
            self.usage = chunk.usage
        if chunk.system_fingerprint:
            self.system_fingerprint = chunk.system_fingerprint

        delta = StreamDelta()
        if chunk.delta_content:
            self.content_parts.append(chunk.delta_content)
            delta.content = chunk.delta_content
        if chunk.delta_reasoning_content:
            self.reasoning_parts.append(chunk.delta_reasoning_content)
            delta.reasoning = chunk.delta_reasoning_content
        if chunk.delta_refusal:
            self.refusal_parts.append(chunk.delta_refusal)
            delta.refusal = chunk.delta_refusal

        if chunk.delta_tool_calls:
            for tc_delta in chunk.delta_tool_calls:
                idx = tc_delta.index
                if idx not in self.tool_call_buffers:
                    self.tool_call_buffers[idx] = {"id": "", "name": "", "arguments": ""}
                buf = self.tool_call_buffers[idx]
                if tc_delta.id:
                    buf["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        buf["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        buf["arguments"] += tc_delta.function.arguments

        return delta

    def build_response(self) -> ModelResponse:
        """累积完成后构建完整 ModelResponse。"""
        content = "".join(self.content_parts) or None
        reasoning = "".join(self.reasoning_parts) or None
        refusal = "".join(self.refusal_parts) or None
        tool_calls: list[ToolCall] | None = None
        if self.tool_call_buffers:
            tool_calls = [
                ToolCall(
                    id=buf["id"],
                    type="function",
                    function=FunctionCall(name=buf["name"], arguments=buf["arguments"]),
                )
                for _, buf in sorted(self.tool_call_buffers.items())
            ]
        usage = self.usage or Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return ModelResponse(
            id=self.response_id,
            content=content,
            reasoning_content=reasoning,
            refusal=refusal,
            tool_calls=tool_calls,
            usage=usage,
            model=self.model,
            finish_reason=self.finish_reason,
            created=0,
            system_fingerprint=self.system_fingerprint,
        )


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
        try:
            data = repair_json(content, return_objects=True)
            return schema_model.model_validate(data)
        except Exception as exc:
            log.warning("结构化响应解析失败", error=str(exc))
            return None
