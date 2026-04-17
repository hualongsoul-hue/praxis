"""Praxis 流式控制台 Agent 示例。

交互式 REPL，实时显示编排事件（轮次、工具调用、终止），
最终输出 Agent 响应内容。

用法::

    uv run python examples/streaming_console.py
"""

import asyncio
import json
import sys

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


def print_event(event) -> None:
    """将编排事件实时渲染到控制台。"""
    t = event.event_type
    d = event.data

    if t == "turn_start":
        print(f"  [Turn {event.turn}]", flush=True)
    elif t == "llm_request":
        tokens = d.get("token_count", "?")
        print(f"  >> LLM 请求 ({tokens} tokens)", flush=True)
    elif t == "llm_response":
        tc = d.get("tool_call_count", 0)
        if tc:
            print(f"  << LLM 响应 (工具调用 x{tc})", flush=True)
        else:
            print("  << LLM 响应", flush=True)
    elif t == "tool_call_start":
        name = d.get("tool_name", "unknown")
        print(f"  🔧 {name} ...", end="", flush=True)
    elif t == "tool_call_end":
        ok = d.get("success", False)
        print(f" {'✓' if ok else '✗'}", flush=True)
    elif t == "termination":
        reason = d.get("reason", "unknown")
        print(f"  [终止: {reason}]", flush=True)


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

    print("Praxis Agent 已就绪")
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

        print()
        async for event in session.run_turn_stream(user_input):
            print_event(event)

        # 从对话历史中取最后一条助手消息作为最终输出
        for msg in reversed(session.assembler.conversation_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                print(f"\n{msg['content']}")
                break

    await store.close()
    print("\n再见！")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
