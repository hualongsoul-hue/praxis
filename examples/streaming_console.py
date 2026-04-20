"""Praxis 流式控制台 Agent 示例（内建工具）。

交互式 REPL，实时显示编排事件（轮次、工具调用、终止），默认加载 Praxis
所有内建工具。适合观察 Agent 在多轮工具调用中的行为。

用法::

    uv run python examples/streaming_console.py
"""

import asyncio
import json
from pathlib import Path

from praxis.agent import create_agent_session
from praxis.config.schemas import GatewayConfig, PersistenceConfig, ToolsConfig
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import (
    PermissionManager,
    PermissionPolicy,
    PermissionRule,
)
from praxis.guardrails.rules import RuleEngine
from praxis.models.guardrails import VerdictType
from praxis.persistence.store import create_store
from praxis.tools.builtins.autonomy import ask_user
from praxis.tools.override import override_tool


async def interactive_ask_user(args: dict) -> str:
    """覆盖默认 ask_user 为真正的终端交互；用 to_thread 避免阻塞事件循环。"""
    question = args.get("question", "")
    print("\n╭─ Agent 向你提问 ────────────────────")
    print(f"│ {question}")
    print("╰────────────────────────────────────")
    answer = await asyncio.to_thread(input, "  你的回答 > ")
    answer = answer.strip()
    return answer or "（用户未提供回答）"

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
    elif t == "content_delta":
        text = d.get("text", "")
        print(text, end="", flush=True)
    elif t == "llm_response":
        tc = d.get("tool_call_count", 0)
        has_content = d.get("has_content", False)
        if has_content:
            print(flush=True)
        if tc:
            print(f"  << LLM 响应 (工具调用 x{tc})", flush=True)
        else:
            print("  << LLM 响应", flush=True)
    elif t == "tool_call_start":
        name = d.get("tool_name", "unknown")
        args = d.get("arguments") or {}
        args_str = json.dumps(args, ensure_ascii=False)
        if len(args_str) > 200:
            args_str = args_str[:200] + "...(truncated)"
        print(f"  [tool] {name} args={args_str}", flush=True)
    elif t == "tool_call_end":
        name = d.get("tool_name", "unknown")
        if d.get("needs_user_confirm"):
            print(f"  [tool] {name} -> NEEDS_CONFIRM ({d.get('reason', '')})", flush=True)
        elif d.get("skipped"):
            print(f"  [tool] {name} -> SKIPPED ({d.get('reason', '')})", flush=True)
        else:
            ok = d.get("success", False)
            mark = "OK" if ok else "FAIL"
            detail = d.get("content") if ok else d.get("error")
            detail_str = "" if detail is None else str(detail)
            if len(detail_str) > 200:
                detail_str = detail_str[:200] + "...(truncated)"
            ms = d.get("execution_time_ms")
            ms_str = f" [{ms:.0f}ms]" if isinstance(ms, (int, float)) else ""
            suffix = f" {detail_str}" if detail_str else ""
            print(f"  [tool] {name} -> {mark}{ms_str}{suffix}", flush=True)
    elif t == "termination":
        reason = d.get("reason", "unknown")
        print(f"  [终止: {reason}]", flush=True)


async def main() -> None:
    store = await create_store(PersistenceConfig())

    rule_engine = RuleEngine()
    rule_engine.register_builtin_rules()
    auto_approve_rules = [
        PermissionRule(category=c, permission=VerdictType.AUTO_APPROVE)
        for c in ("file_ops", "search", "shell", "system", "autonomy", "general")
    ]
    permissions = PermissionManager(PermissionPolicy(rules=auto_approve_rules))
    guardrails = GuardrailEngine(rule_engine, permissions)

    gateway = GatewayRouter(GATEWAY_CONFIG)

    workspace = Path.cwd().resolve()
    tools_config = ToolsConfig(
        allowed_paths=[str(workspace)],
        shell_timeout=60.0,
        network_allowed=True,
    )

    session = await create_agent_session(
        store=store,
        guardrails=guardrails,
        gateway=gateway,
        tools_config=tools_config,
    )

    # 覆盖 ask_user 为真实终端交互
    override_tool(session.registry, ask_user.DEFINITION, interactive_ask_user)

    print("Praxis Agent 已就绪（流式，内建工具模式）")
    print(f"  沙箱路径: {workspace}")
    print(f"  已注册工具: {', '.join(sorted(session.registry.list_tools()))}")
    print("输入 /quit 退出，/clear 清空上下文，/tools 列出工具")
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
        if user_input == "/tools":
            for name in sorted(session.registry.list_tools()):
                defn = session.registry.get_definition(name)
                print(f"  - {name}: {defn.description}")
            continue

        print()
        async for event in session.run_turn_stream(user_input):
            print_event(event)

    await session.terminate()
    await store.close()
    print("\n再见！")


if __name__ == "__main__":
    asyncio.run(main())
