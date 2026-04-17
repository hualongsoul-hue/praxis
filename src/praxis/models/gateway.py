"""模型网关相关数据模型——S4 内部及跨组件共享。"""

from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    """LLM 评估结果，用于 S10 推理型验证。"""

    verdict: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    raw_response: str | None = None
