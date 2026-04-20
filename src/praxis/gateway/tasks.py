"""summarize / judge 便捷接口。

摘要接口供 S7 上下文压缩使用，评估接口供 S10 推理型验证使用。
底层复用 chat 接口，附加专用 System Prompt。
"""

from typing import Any

from json_repair import repair_json

from praxis.gateway.chat import chat
from praxis.gateway.router import GatewayRouter
from praxis.models.gateway import JudgeResult

SUMMARIZE_SYSTEM_PROMPT = (
    "你是一个精准的文本摘要助手。请根据用户提供的指令对内容进行摘要，"
    "保留所有关键信息，去除冗余细节。仅输出摘要文本，不添加额外说明。"
)

JUDGE_SYSTEM_PROMPT = (
    "你是一个严格的评估专家。请根据给定的评估标准对内容进行评估。\n"
    "必须以 JSON 格式输出，包含以下字段：\n"
    '- "verdict": true 或 false（是否通过）\n'
    '- "confidence": 0.0 到 1.0 之间的置信度\n'
    '- "reasoning": 评估理由的简要说明\n'
    "仅输出 JSON，不添加其他文字。"
)


async def summarize(
    gateway: GatewayRouter,
    content: str,
    instruction: str = "请对以下内容进行精炼摘要",
    model: str | None = None,
    **kwargs: Any,
) -> str:
    """摘要接口，供 S7 上下文压缩使用。

    Args:
        gateway: 网关路由器实例。
        content: 待摘要的原始内容。
        instruction: 摘要指令。
        model: 模型别名，None 时使用默认模型。
        **kwargs: 额外推理参数透传。

    Returns:
        摘要文本。
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{instruction}\n\n---\n\n{content}"},
    ]
    response = await chat(gateway, messages, model=model, **kwargs)
    return response.content or ""


async def judge(
    gateway: GatewayRouter,
    criteria: str,
    content: str,
    model: str | None = None,
    **kwargs: Any,
) -> JudgeResult:
    """评估接口，供 S10 推理型验证使用。

    Args:
        gateway: 网关路由器实例。
        criteria: 评估标准描述。
        content: 待评估内容。
        model: 模型别名，None 时使用默认模型。
        **kwargs: 额外推理参数透传。

    Returns:
        结构化评估结果。
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"评估标准：{criteria}\n\n待评估内容：\n{content}"},
    ]
    response = await chat(gateway, messages, model=model, **kwargs)
    raw_text = response.content or ""

    data = repair_json(raw_text, return_objects=True)
    if isinstance(data, dict):
        return JudgeResult(
            verdict=data.get("verdict", False),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            raw_response=raw_text,
        )
    return JudgeResult(
        verdict=False,
        confidence=0.0,
        reasoning=f"无法解析 LLM 响应为 JSON: {raw_text[:200]}",
        raw_response=raw_text,
    )
