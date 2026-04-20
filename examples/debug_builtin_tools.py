"""内建工具调试脚本。

非交互式场景，通过一组预设 prompt 让 Agent 自主选择并调用 Praxis 内建工具，
分别以非流式和流式两种方式运行，打印每轮工具调用、参数、事件序列，验证：

- 工具被正确注入到 LLM schema
- LLM 根据任务自主选择工具
- 编排循环事件按顺序发射
- 流式与非流式结果一致

用法::

    uv run python examples/debug_builtin_tools.py
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from praxis.agent import create_agent_session
from praxis.config.schemas import (
    ContextConfig,
    GatewayConfig,
    MemoryConfig,
    OrchestratorConfig,
    PersistenceConfig,
    SessionConfig,
    ToolsConfig,
)
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

# 调试脚本非交互，ask_user 使用脚本化固定应答
SCRIPTED_ASK_USER_REPLY = "是的，请继续完成任务，无需再询问。"


async def scripted_ask_user(args: dict) -> str:
    """覆盖默认 ask_user 为脚本化应答，并打印交互过程。"""
    question = args.get("question", "")
    print(f"    [ask_user] 问题: {question}")
    print(f"    [ask_user] 应答: {SCRIPTED_ASK_USER_REPLY}")
    return SCRIPTED_ASK_USER_REPLY

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

# (标签, prompt, 期望至少调用到的工具集合)
TEST_CASES: list[tuple[str, str, set[str]]] = [
    (
        "list_dir",
        "请列出当前工作目录下的所有子目录名称。",
        {"list_dir"},
    ),
    (
        "read_file",
        "读取 pyproject.toml 的前 30 行，告诉我项目名称和 Python 版本要求。",
        {"read_file"},
    ),
    (
        "grep_search",
        "在当前工作区搜索字符串 'CognitiveMemory' 出现在哪些文件中。",
        {"grep_search"},
    ),
    (
        "write_file",
        "请用 write_file 工具把内容 'hello from praxis' 写到当前工作目录下的 './praxis_demo.txt'，"
        "仅此一步，不要执行任何验证读取。",
        {"write_file"},
    ),
    (
        "run_command",
        "请用 run_command 工具执行命令 'echo hello_praxis'，告诉我输出内容。",
        {"run_command"},
    ),
    (
        "get_system_info",
        "告诉我当前系统的操作系统名称和 Python 版本（不要猜测，用工具查）。",
        {"get_system_info"},
    ),
    (
        "ask_user",
        "请用 ask_user 工具向我提出一个关于调试偏好的问题，然后根据我的回答继续汇报已完成。",
        {"ask_user"},
    ),
    (
        "submit_result",
        "请用 submit_result 工具汇报『调试任务已完成』。",
        {"submit_result"},
    ),
]


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def format_tool_event(event) -> str:
    t = event.event_type
    d = event.data
    if t == "tool_call_start":
        name = d.get("tool_name", "?")
        args = d.get("arguments") or {}
        args_str = json.dumps(args, ensure_ascii=False)
        if len(args_str) > 160:
            args_str = args_str[:160] + "...(truncated)"
        return f"    [call] {name} args={args_str}"
    if t == "tool_call_end":
        name = d.get("tool_name", "?")
        if d.get("needs_user_confirm"):
            return f"    [done] {name} -> NEEDS_CONFIRM ({d.get('reason', '')})"
        if d.get("skipped"):
            return f"    [done] {name} -> SKIPPED ({d.get('reason', '')})"
        ok = d.get("success", False)
        mark = "OK" if ok else "FAIL"
        detail = d.get("content") if ok else d.get("error")
        detail_str = "" if detail is None else str(detail)
        if len(detail_str) > 160:
            detail_str = detail_str[:160] + "...(truncated)"
        ms = d.get("execution_time_ms")
        ms_str = f" [{ms:.0f}ms]" if isinstance(ms, (int, float)) else ""
        suffix = f" {detail_str}" if detail_str else ""
        return f"    [done] {name} -> {mark}{ms_str}{suffix}"
    if t == "termination":
        return f"    [term] {d.get('reason', '?')}"
    return ""


async def build_session(store, gateway, guardrails, workspace: Path):
    tools_config = ToolsConfig(
        allowed_paths=[str(workspace), str(Path(tempfile.gettempdir()).resolve())],
        shell_timeout=30.0,
        network_allowed=True,
    )
    # 禁用后台记忆子系统，避免额外的 LLM embedding 噪声
    memory_config = MemoryConfig(
        background_enabled=False,
        dream_enabled=False,
        load_project_praxis_md=False,
    )
    session = await create_agent_session(
        store=store,
        guardrails=guardrails,
        gateway=gateway,
        tools_config=tools_config,
        orchestrator_config=OrchestratorConfig(max_turns=6),
        context_config=ContextConfig(),
        session_config=SessionConfig(auto_checkpoint=False),
        memory_config=memory_config,
    )
    override_tool(session.registry, ask_user.DEFINITION, scripted_ask_user)
    return session


async def run_non_streaming(store, gateway, guardrails, workspace: Path) -> list[tuple[str, bool, set[str]]]:
    banner("非流式模式 (run_turn)")
    results: list[tuple[str, bool, set[str]]] = []

    for tag, prompt, expected in TEST_CASES:
        print(f"\n--- [{tag}] ---")
        print(f"> {prompt}")
        session = await build_session(store, gateway, guardrails, workspace)
        try:
            response = await session.run_turn(prompt)
            called = {e.data.get("tool_name") for e in response.events
                      if e.event_type == "tool_call_start"}
            called = {n for n in called if n}
            missing = expected - called
            ok = not missing
            status = "PASS" if ok else f"MISSING {sorted(missing)}"
            print(f"  turns={response.total_turns} tools_called={sorted(called)} => {status}")
            reply = (response.content or "").strip()
            if reply:
                preview = reply if len(reply) <= 200 else reply[:200] + "..."
                print(f"  reply: {preview}")
            results.append((tag, ok, called))
        finally:
            await session.terminate()

    return results


async def run_streaming(store, gateway, guardrails, workspace: Path) -> list[tuple[str, bool, set[str]]]:
    banner("流式模式 (run_turn_stream)")
    results: list[tuple[str, bool, set[str]]] = []

    for tag, prompt, expected in TEST_CASES:
        print(f"\n--- [{tag}] ---")
        print(f"> {prompt}")
        session = await build_session(store, gateway, guardrails, workspace)
        called: set[str] = set()
        try:
            async for event in session.run_turn_stream(prompt):
                line = format_tool_event(event)
                if line:
                    print(line)
                if event.event_type == "tool_call_start":
                    name = event.data.get("tool_name")
                    if name:
                        called.add(name)
            missing = expected - called
            ok = not missing
            status = "PASS" if ok else f"MISSING {sorted(missing)}"
            print(f"  tools_called={sorted(called)} => {status}")
            for msg in reversed(session.assembler.conversation_history):
                if msg.get("role") == "assistant" and msg.get("content"):
                    reply = msg["content"].strip()
                    preview = reply if len(reply) <= 200 else reply[:200] + "..."
                    print(f"  reply: {preview}")
                    break
            results.append((tag, ok, called))
        finally:
            await session.terminate()

    return results


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

    # 提前展示一次已注册内建工具清单
    warmup = await build_session(store, gateway, guardrails, workspace)
    try:
        tools = sorted(warmup.registry.list_tools())
        banner(f"已注册内建工具 ({len(tools)})")
        for name in tools:
            defn = warmup.registry.get_definition(name)
            print(f"  - {name}: {defn.description[:60]}")
    finally:
        await warmup.terminate()

    ns_results = await run_non_streaming(store, gateway, guardrails, workspace)
    st_results = await run_streaming(store, gateway, guardrails, workspace)

    banner("汇总")
    print("mode             | case               | status")
    print("-" * 60)
    for tag, ok, called in ns_results:
        print(f"non-streaming    | {tag:<18} | {'PASS' if ok else 'FAIL'} called={sorted(called)}")
    for tag, ok, called in st_results:
        print(f"streaming        | {tag:<18} | {'PASS' if ok else 'FAIL'} called={sorted(called)}")

    await store.close()

    ns_fail = [t for t, ok, _ in ns_results if not ok]
    st_fail = [t for t, ok, _ in st_results if not ok]
    if ns_fail or st_fail:
        print(f"\nFAILED cases: non-streaming={ns_fail} streaming={st_fail}")
        sys.exit(1)
    print("\n全部内建工具调用验证通过")


if __name__ == "__main__":
    asyncio.run(main())
