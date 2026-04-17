"""Praxis 控制台 Agent 示例（非流式）。

交互式 REPL，每轮对话等待完整响应后输出。

用法::

    uv run python examples/console.py
"""

import asyncio
import sys

import json
from datetime import datetime, timezone

from praxis.agent import create_agent_session
from praxis.config.schemas import GatewayConfig, PersistenceConfig
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import RuleEngine
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.persistence.store import create_store
from praxis.tools.registry import ToolRegistry

GATEWAY_CONFIG = GatewayConfig(
    model_list=[
        {
            "model_name": "default",
            "litellm_params": {
                "model": "openai/nvidia/Kimi-K2.5-NVFP4",
                "api_key": "sk-n69uaJWNmHaN2jGWrNmpDkVJ7PuX7rSs4M8LLrJq7icUobpV",
                "api_base": "http://172.24.23.237:3000/v1",
                "max_tokens": 128000,
            },
        },
    ],
    default_model="default",
)


async def main() -> None:
    store = await create_store(PersistenceConfig())

    rule_engine = RuleEngine()
    rule_engine.register_builtin_rules()
    guardrails = GuardrailEngine(rule_engine, PermissionManager())

    gateway = GatewayRouter(GATEWAY_CONFIG)

    async def handle_get_time(_: dict) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    async def handle_calculate(params: dict) -> str:
        expr = params.get("expression", "")
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return json.dumps({"expression": expr, "result": result})
        except Exception as e:
            return json.dumps({"expression": expr, "error": str(e)})

    async def handle_get_weather(params: dict) -> str:
        city = params.get("city", "unknown")
        return json.dumps({"city": city, "temperature": "22°C", "condition": "晴", "humidity": "45%"})

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_current_time",
            description="获取当前 UTC 时间",
            parameters={"type": "object", "properties": {}},
            metadata=ToolMetadata(category="utility", permission_level="auto_approve", readonly=True),
        ),
        handler=handle_get_time,
    )
    registry.register(
        ToolDefinition(
            name="calculate",
            description="计算数学表达式",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '2+3*4'"},
                },
                "required": ["expression"],
            },
            metadata=ToolMetadata(category="utility", permission_level="auto_approve", readonly=True),
        ),
        handler=handle_calculate,
    )
    registry.register(
        ToolDefinition(
            name="get_weather",
            description="查询城市天气",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                },
                "required": ["city"],
            },
            metadata=ToolMetadata(category="utility", permission_level="auto_approve", readonly=True),
        ),
        handler=handle_get_weather,
    )

    session = create_agent_session(store=store, guardrails=guardrails, registry=registry)
    session.loop.gateway = gateway

    print("Praxis Agent 已就绪（非流式）")
    print("输入 /quit 退出，/clear 清空上下文")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            break
        if user_input == "/clear":
            session.assembler.conversation_history.clear()
            session.loop.state.current_turn = 0
            print("上下文已清空。")
            continue

        response = await session.run_turn(user_input)
        print(f"\n{response.content}")
        print(f"  [turns={response.total_turns} tools={response.tool_calls_made} reason={response.termination_reason.value}]")

    await store.close()
    print("\n再见！")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
