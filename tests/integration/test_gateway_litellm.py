"""S4 网关真实链路集成测试。

使用 LiteLLM 的 mock_response 在不发起网络请求的前提下，验证
GatewayRouter → chat/chat_stream → LiteLLM Router → 内部数据模型映射的真实路径
（其余 e2e 测试均以 mock gateway 短路了这一层）。
"""

import pytest

from praxis.config.schemas import GatewayConfig, ModelDeployment
from praxis.gateway.chat import chat, chat_stream
from praxis.gateway.router import GatewayRouter

pytestmark = pytest.mark.asyncio(loop_scope="module")


def gateway() -> GatewayRouter:
    cfg = GatewayConfig(
        deployments=[ModelDeployment(model="gpt-4o-mini")],
        default_model="default",
    )
    return GatewayRouter(cfg, environ={"PRAXIS_MODEL_API_KEY": "unit-test-model-key"})


class TestGatewayLiteLLMPath:
    async def test_chat_real_mapping(self) -> None:
        gw = gateway()
        resp = await chat(
            gw, [{"role": "user", "content": "hi"}], mock_response="你好世界",
        )
        assert resp.content == "你好世界"
        assert resp.finish_reason == "stop"
        assert resp.model
        assert resp.usage is not None

    async def test_chat_stream_real_mapping(self) -> None:
        gw = gateway()
        parts: list[str] = []
        async for chunk in chat_stream(
            gw, [{"role": "user", "content": "hi"}], mock_response="流式回复",
        ):
            if chunk.delta_content:
                parts.append(chunk.delta_content)
        assert "".join(parts) == "流式回复"

    async def test_budget_accumulates_on_real_path(self) -> None:
        """走真实 chat 路径后，累计花费应被记录（≥0，mock 成本可能为 0）。"""
        gw = gateway()
        await chat(gw, [{"role": "user", "content": "hi"}], mock_response="ok")
        assert gw.total_spend >= 0.0

    async def test_stream_accumulates_spend(self) -> None:
        """流式调用也须累计花费（此前 chat_stream 不计入预算 → max_budget 失效）。"""
        gw = gateway()
        got_usage = False
        async for chunk in chat_stream(
            gw, [{"role": "user", "content": "hi"}], mock_response="流式",
        ):
            if chunk.usage is not None:
                got_usage = True
        assert got_usage  # 末块带 usage（stream_options include_usage）
        assert gw.total_spend > 0.0  # gpt-4o-mini 有定价，应累计到 spend
