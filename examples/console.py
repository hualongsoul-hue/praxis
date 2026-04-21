"""Praxis 控制台 Agent 示例（非流式，内建工具）。

交互式 REPL，每轮对话等待完整响应后输出。默认加载 Praxis 所有内建工具
（文件、搜索、Shell、网络、系统、自治管理），可直接让 Agent 读写文件、执行命令、搜索代码等。

用法::

    uv run python examples/console.py
"""

import asyncio
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
    """覆盖默认 ask_user 为真正的终端交互。"""
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
                "model": "openai/glm-5.1",
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
    # 示例默认自动放行所有工具类别，生产环境应细化为 CONFIRM + 交互式批准
    auto_approve_rules = [
        PermissionRule(category=c, permission=VerdictType.AUTO_APPROVE)
        for c in ("file_ops", "search", "shell", "system", "autonomy", "general")
    ]
    permissions = PermissionManager(PermissionPolicy(rules=auto_approve_rules))
    guardrails = GuardrailEngine(rule_engine, permissions)

    gateway = GatewayRouter(GATEWAY_CONFIG)

    # 沙箱限制在工作区内，防止误删系统文件；Shell 命令超时 60s
    workspace = Path.cwd().resolve()
    tools_config = ToolsConfig(
        allowed_paths=[str(workspace)],
        shell_timeout=60.0,
        network_allowed=True,
    )

    # 不传 registry，factory 会自动创建并注册全部内建工具
    session = await create_agent_session(
        store=store,
        guardrails=guardrails,
        gateway=gateway,
        tools_config=tools_config,
    )

    # 覆盖 ask_user 为真实终端交互
    override_tool(session.registry, ask_user.DEFINITION, interactive_ask_user)

    print("Praxis Agent 已就绪（非流式，内建工具模式）")
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

        response = await session.run_turn(user_input)
        print(f"\n{response.content}")
        print(
            f"  [turns={response.total_turns} tools={response.tool_calls_made} "
            f"reason={response.termination_reason.value}]"
        )

    await session.terminate()
    await store.close()
    print("\n再见！")


if __name__ == "__main__":
    asyncio.run(main())
