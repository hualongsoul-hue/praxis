"""调试脚本：验证技能系统（S14）和 MCP 桥接（S5-MCP）的完整调用链。

测试内容:
1. 技能发现 → 注册 → 披露工具注册 → 索引注入
2. 自动激活（关键词匹配）
3. 披露工具可被 LLM 调用（load_skill / list_skill_files）
4. MCP 工具桥接注册（模拟，无真实 MCP Server）

用法::

    uv run python examples/debug_skills_mcp.py
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from praxis.agent import create_agent_session
from praxis.config.schemas import GatewayConfig, PersistenceConfig
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import RuleEngine
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.persistence.store import PersistenceStore, create_store
from praxis.skills.manager import SkillManager
from praxis.tools.mcp import MCPToolsBridge
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

SKILLS_DIR = str(Path(__file__).parent / "skills")


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def test_skill_system(store: PersistenceStore, registry: ToolRegistry) -> SkillManager:
    """测试技能发现、注册、披露、激活。"""
    section("S14 技能系统")

    manager = SkillManager(registry=registry, store=store)

    # 1. 发现
    skills = await manager.initialize([SKILLS_DIR])
    print(f"  发现技能数: {len(skills)}")
    for s in skills:
        print(f"    - {s.skill_id}: {s.metadata.description}")
        print(f"      文件: {s.files}")
        print(f"      脚本: {s.scripts}")

    # 2. 索引
    index = manager.get_skill_index()
    print(f"\n  技能索引条目: {len(index)}")
    for entry in index:
        print(f"    - {entry.skill_id}: {entry.description[:60]}")

    # 3. 披露工具注册
    disclosure_tools = manager.register_disclosure_tools()
    print(f"\n  已注册披露工具: {disclosure_tools}")

    # 4. 相关性评估
    relevance = manager.evaluate_relevance("review my Python code")
    print(f"\n  相关性评估 (task='review my Python code'):")
    for entry, score in relevance:
        print(f"    - {entry.skill_id}: {score:.2f}")

    # 5. 自动激活
    activated = manager.auto_activate_for_task("review my Python code")
    print(f"\n  自动激活: {activated}")

    # 6. 加载技能
    loaded = manager.load_skill("code-review")
    if loaded:
        print(f"\n  技能内容（前 100 字符）: {loaded.content[:100]}...")

    # 7. 列出附属文件
    files = manager.disclosure.list_skill_files("code-review")
    print(f"  附属文件: {files}")

    # 8. 加载附属文件
    if files:
        content = manager.disclosure.load_skill_file("code-review", files[0])
        if content:
            print(f"  附属文件内容（前 80 字符）: {content[:80]}...")

    # 9. 注册表检查
    registered_tools = [name for name in registry.list_tools()]
    print(f"\n  注册表中全部工具: {registered_tools}")

    return manager


async def test_mcp_bridge(registry: ToolRegistry) -> None:
    """测试 MCP 工具桥接（模拟，无需真实 MCP Server）。"""
    section("S5-MCP 工具桥接")

    bridge = MCPToolsBridge(registry=registry)

    # 模拟注册：直接调用注册方法验证工具格式
    mock_definition = ToolDefinition(
        name="mcp_mock_server_echo",
        description="[MCP:mock_server] 回显输入内容",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "要回显的消息"},
            },
            "required": ["message"],
        },
        metadata=ToolMetadata(category="mcp", readonly=True, tags=["mcp:mock_server"]),
    )

    async def mock_echo_handler(arguments: dict) -> str:
        msg = arguments.get("message", "")
        return json.dumps({"echo": msg, "timestamp": datetime.now(timezone.utc).isoformat()})

    registry.register(mock_definition, mock_echo_handler)
    bridge.server_tools["mock_server"] = ["mcp_mock_server_echo"]

    print(f"  模拟 MCP 工具已注册: mcp_mock_server_echo")

    # 验证工具在注册表中
    has_tool = registry.has_tool("mcp_mock_server_echo")
    print(f"  注册表确认: {has_tool}")

    # 验证 Schema 导出
    schemas = registry.get_tool_schemas(category="mcp")
    print(f"  MCP 类别工具 Schema 数: {len(schemas)}")
    for s in schemas:
        print(f"    - {s['function']['name']}: {s['function']['description'][:50]}")

    # 直接调用
    entry = registry.get_entry("mcp_mock_server_echo")
    result = await entry.handler({"message": "hello from praxis"})
    print(f"  直接调用结果: {result}")


async def test_e2e_with_skills(
    store: PersistenceStore,
    registry: ToolRegistry,
    skill_manager: SkillManager,
) -> None:
    """端到端测试：技能 + MCP 工具通过 Agent 调用。"""
    section("端到端调用测试")

    rule_engine = RuleEngine()
    rule_engine.register_builtin_rules()
    guardrails = GuardrailEngine(rule_engine, PermissionManager())
    gateway = GatewayRouter(GATEWAY_CONFIG)

    session = create_agent_session(
        store=store,
        guardrails=guardrails,
        gateway=gateway,
        registry=registry,
        skill_manager=skill_manager,
    )

    # 打印当前可用工具
    all_tools = registry.list_tools()
    print(f"  会话工具总数: {len(all_tools)}")
    for t in all_tools:
        print(f"    - {t}")

    # 测试 1: 触发披露工具（load_skill）
    print("\n  --- 测试 1: 要求 LLM 加载技能 ---")
    response = await session.run_turn(
        "Load the code-review skill content using the load_skill tool with skill_id='code-review'."
    )
    print(f"  回复: {response.content[:200]}")
    print(f"  turns={response.total_turns} tools={response.tool_calls_made}")

    # 测试 2: 触发 MCP 模拟工具
    print("\n  --- 测试 2: 调用 MCP 模拟工具 ---")
    session.assembler.conversation_history.clear()
    session.loop.state.current_turn = 0
    response = await session.run_turn(
        "Use the mcp_mock_server_echo tool to echo the message 'skill+mcp integration test'."
    )
    print(f"  回复: {response.content[:200]}")
    print(f"  turns={response.total_turns} tools={response.tool_calls_made}")

    # 测试 3: 流式调用
    print("\n  --- 测试 3: 流式调用 list_skill_files ---")
    session.assembler.conversation_history.clear()
    session.loop.state.current_turn = 0
    events = []
    async for event in session.run_turn_stream(
        "Use the list_skill_files tool with skill_id='code-review' to see its files."
    ):
        events.append(event)
        if event.event_type in ("tool_call_start", "tool_call_end", "termination"):
            print(f"    [{event.event_type}] {event.data}")
    print(f"  事件总数: {len(events)}")

    for msg in reversed(session.assembler.conversation_history):
        if msg.get("role") == "assistant" and msg.get("content"):
            print(f"  最终回复: {msg['content'][:200]}")
            break


async def main() -> None:
    store = await create_store(PersistenceConfig())

    registry = ToolRegistry()

    # 注册基础工具
    async def handle_get_time(_: dict) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    registry.register(
        ToolDefinition(
            name="get_current_time",
            description="获取当前 UTC 时间",
            parameters={"type": "object", "properties": {}},
            metadata=ToolMetadata(category="utility", permission_level="auto_approve", readonly=True),
        ),
        handler=handle_get_time,
    )

    # S14: 技能系统
    skill_manager = await test_skill_system(store, registry)

    # S5-MCP: 工具桥接
    await test_mcp_bridge(registry)

    # 端到端
    await test_e2e_with_skills(store, registry, skill_manager)

    await store.close()
    print("\n\n✓ 全部调试完成")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
