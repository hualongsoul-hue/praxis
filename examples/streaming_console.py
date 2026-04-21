"""Praxis 流式控制台 Agent 示例（内建工具）。

交互式 REPL，实时显示编排事件（轮次、工具调用、终止），默认加载 Praxis
所有内建工具。适合观察 Agent 在多轮工具调用中的行为。

用法::

    uv run python examples/streaming_console.py
"""

import asyncio
import ctypes
import json
import sys
from pathlib import Path

# 日志重定向到文件，使控制台只显示 Agent 事件；必须早于其他 praxis 子模块 import
from praxis.config.schemas import (  # noqa: E402
    GatewayConfig,
    PersistenceConfig,
    TelemetryConfig,
    ToolsConfig,
)
from praxis.telemetry.logger import configure_logging  # noqa: E402

LOG_FILE = Path("logs/streaming_console.log")
configure_logging(TelemetryConfig(log_level="INFO", log_file=str(LOG_FILE)))

from praxis.agent import create_agent_session  # noqa: E402
from praxis.gateway.router import GatewayRouter  # noqa: E402
from praxis.guardrails.engine import GuardrailEngine  # noqa: E402
from praxis.guardrails.permissions import (  # noqa: E402
    PermissionManager,
    PermissionPolicy,
    PermissionRule,
)
from praxis.guardrails.rules import RuleEngine  # noqa: E402
from praxis.models.guardrails import VerdictType  # noqa: E402
from praxis.persistence.store import create_store  # noqa: E402
from praxis.skills import BUILTIN_SKILLS_PATH, SkillManager  # noqa: E402
from praxis.tools.builtins.autonomy import ask_user  # noqa: E402
from praxis.tools.override import override_tool  # noqa: E402


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
                "model": "openai/glm-5.1",
                "api_key": "sk-n69uaJWNmHaN2jGWrNmpDkVJ7PuX7rSs4M8LLrJq7icUobpV",
                "api_base": "http://172.24.23.237:3000/v1",
                "max_tokens": 128000,
            },
        },
    ],
    default_model="default",
)


# ANSI 颜色与样式（Windows 10+ 控制台默认启用 VT 序列）
ANSI_DIM = "\033[2m"
ANSI_ITALIC = "\033[3m"
ANSI_GRAY = "\033[90m"
ANSI_CYAN = "\033[36m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"


class EventRenderer:
    """将编排事件实时渲染到控制台。

    - 思考过程（reasoning_delta）：灰色斜体，块内逐行添加 ``│ `` 前缀。
    - 正式回答（content_delta）：默认颜色正常显示，带清晰分隔头。
    """

    THINKING_OPEN = f"{ANSI_GRAY}╭─ 💭 思考 ─────────────────────────────{ANSI_RESET}\n{ANSI_GRAY}│ {ANSI_DIM}{ANSI_ITALIC}"
    THINKING_CLOSE = f"{ANSI_RESET}\n{ANSI_GRAY}╰────────────────────────────────────────{ANSI_RESET}"
    ANSWER_HEADER = f"{ANSI_CYAN}▌ 回答{ANSI_RESET}\n"

    def __init__(self) -> None:
        self.reasoning_active = False
        self.answer_header_printed = False

    def open_thinking(self) -> None:
        print(self.THINKING_OPEN, end="", flush=True)
        self.reasoning_active = True

    def close_thinking(self) -> None:
        if self.reasoning_active:
            print(self.THINKING_CLOSE, flush=True)
            self.reasoning_active = False

    def render(self, event) -> None:
        t = event.event_type
        d = event.data

        if t == "turn_start":
            print(f"{ANSI_BOLD}  [Turn {event.turn}]{ANSI_RESET}", flush=True)
            self.answer_header_printed = False
        elif t == "llm_request":
            tokens = d.get("token_count", "?")
            print(f"  {ANSI_DIM}>> LLM 请求 ({tokens} tokens){ANSI_RESET}", flush=True)
        elif t == "reasoning_delta":
            text = d.get("text", "")
            if not self.reasoning_active:
                self.open_thinking()
            # 思考内容内部换行也加上 │ 前缀
            print(text.replace("\n", f"{ANSI_RESET}\n{ANSI_GRAY}│ {ANSI_DIM}{ANSI_ITALIC}"), end="", flush=True)
        elif t == "content_delta":
            text = d.get("text", "")
            self.close_thinking()
            if not self.answer_header_printed:
                print(self.ANSWER_HEADER, end="", flush=True)
                self.answer_header_printed = True
            print(text, end="", flush=True)
        elif t == "llm_response":
            self.close_thinking()
            tc = d.get("tool_call_count", 0)
            has_content = d.get("has_content", False)
            if has_content:
                print(flush=True)
            tags = []
            pt = d.get("prompt_tokens", 0) or 0
            ct = d.get("completion_tokens", 0) or 0
            rt = d.get("reasoning_tokens", 0) or 0
            cpt = d.get("cached_prompt_tokens", 0) or 0
            if pt or ct:
                tok = f"p={pt}"
                if cpt:
                    tok += f"(cached={cpt})"
                tok += f" c={ct}"
                if rt:
                    tok += f"(think={rt})"
                tags.append(tok)
            if d.get("has_reasoning"):
                tags.append("含思考")
            if d.get("has_refusal"):
                tags.append("安全拒答")
            if tc:
                tags.append(f"工具调用 x{tc}")
            suffix = f" ({', '.join(tags)})" if tags else ""
            print(f"  {ANSI_DIM}<< LLM 响应{suffix}{ANSI_RESET}", flush=True)
        elif t == "tool_call_start":
            name = d.get("tool_name", "unknown")
            tc_id = d.get("tool_call_id", "")
            args = d.get("arguments") or {}
            args_str = json.dumps(args, ensure_ascii=False)
            if len(args_str) > 200:
                args_str = args_str[:200] + "...(truncated)"
            print(f"  [tool] {name} id={tc_id} args={args_str}", flush=True)
        elif t == "tool_call_end":
            name = d.get("tool_name", "unknown")
            tc_id = d.get("tool_call_id", "")
            if d.get("needs_user_confirm"):
                print(f"  [tool] {name} id={tc_id} -> NEEDS_CONFIRM ({d.get('reason', '')})", flush=True)
            elif d.get("skipped"):
                print(f"  [tool] {name} id={tc_id} -> SKIPPED ({d.get('reason', '')})", flush=True)
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
                print(f"  [tool] {name} id={tc_id} -> {mark}{ms_str}{suffix}", flush=True)
        elif t == "termination":
            reason = d.get("reason", "unknown")
            print(f"  [终止: {reason}]", flush=True)


def enable_windows_ansi() -> None:
    """Windows 控制台启用 ANSI VT 序列支持。"""
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    STD_OUTPUT_HANDLE = -11
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    mode = ctypes.c_ulong()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)


async def main() -> None:
    enable_windows_ansi()
    print(f"{ANSI_DIM}日志写入: {LOG_FILE.resolve()}{ANSI_RESET}")
    print(f"{ANSI_DIM}（如需实时查看：Get-Content -Wait {LOG_FILE}）{ANSI_RESET}\n")
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

    skill_manager = SkillManager(registry=session.registry, store=store)
    await skill_manager.initialize([BUILTIN_SKILLS_PATH])
    skill_manager.register_disclosure_tools()
    session.loop.skill_manager = skill_manager
    session.skill_manager = skill_manager

    # 覆盖 ask_user 为真实终端交互
    override_tool(session.registry, ask_user.DEFINITION, interactive_ask_user)

    skill_names = [s.name for s in skill_manager.get_skill_index()]
    print("Praxis Agent 已就绪（流式，内建工具模式）")
    print(f"  沙箱路径: {workspace}")
    print(f"  已注册工具: {', '.join(sorted(session.registry.list_tools()))}")
    print(f"  已加载技能: {', '.join(skill_names) if skill_names else '无'}")
    print("输入 /quit 退出，/clear 清空上下文，/tools 列出工具，/skills 列出技能")
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
        if user_input == "/skills":
            for entry in skill_manager.get_skill_index():
                print(f"  - {entry.name}: {entry.description}")
            continue

        print()
        renderer = EventRenderer()
        async for event in session.run_turn_stream(user_input):
            renderer.render(event)

    await session.terminate()
    await store.close()
    print("\n再见！")


if __name__ == "__main__":
    asyncio.run(main())
