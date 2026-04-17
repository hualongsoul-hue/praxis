"""TAO/ReAct 核心循环引擎。

实现完整 TAO 循环：
消息记录 → Prompt 组装 → LLM 推理 → 工具执行 → 上下文更新 → 终止检查。
支持同步/异步/流式三种模式。
"""

import time
from collections.abc import AsyncIterator
from typing import Any

from praxis.config.schemas import OrchestratorConfig
from praxis.context.assembler import PromptAssembler
from praxis.context.tool_injection import ToolInjector
from praxis.gateway.chat import chat
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine
from praxis.memory.pipeline import MemoryPipeline
from praxis.models.context import TurnContext
from praxis.models.guardrails import VerdictType
from praxis.models.memory import WorkingMemoryMessage
from praxis.models.orchestrator import (
    AgentEvent,
    AgentResponse,
    LoopPhase,
    LoopState,
    TerminationReason,
)
from praxis.models.tools import ToolResult
from praxis.orchestrator.events import EventEmitter, StreamCollector
from praxis.orchestrator.parser import OutputParser
from praxis.orchestrator.strategy import LoopStrategy
from praxis.orchestrator.termination import TerminationManager
from praxis.orchestrator.tool_coordination import ToolCoordinator
from praxis.skills.manager import SkillManager
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric
from praxis.verification.registry import VerifierRegistry
from praxis.models.verification import QualityPhase

log = get_logger("orchestrator.loop")


class OrchestrationLoop:
    """TAO/ReAct 核心循环引擎。

    协调 S4（LLM）、S5（工具）、S6（记忆）、S7（上下文）、
    S8（护栏）、S9（恢复）、S10（验证）、S14（技能）
    完成完整的 Agent 轮次。
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        gateway: GatewayRouter,
        assembler: PromptAssembler,
        tool_coordinator: ToolCoordinator,
        guardrails: GuardrailEngine,
        termination: TerminationManager,
        strategy: LoopStrategy,
        parser: OutputParser,
        emitter: EventEmitter,
        tool_injector: ToolInjector | None = None,
        memory: MemoryPipeline | None = None,
        verifier_registry: VerifierRegistry | None = None,
        skill_manager: SkillManager | None = None,
    ) -> None:
        self.config = config
        self.gateway = gateway
        self.assembler = assembler
        self.coordinator = tool_coordinator
        self.guardrails = guardrails
        self.termination = termination
        self.strategy = strategy
        self.parser = parser
        self.emitter = emitter
        self.tool_injector = tool_injector
        self.memory = memory
        self.verifier_registry = verifier_registry
        self.skill_manager = skill_manager
        self.state = LoopState()

    async def run(
        self,
        user_message: str,
        system_prompt_override: str | None = None,
        developer_instructions: str = "",
        user_instructions: str = "",
    ) -> AgentResponse:
        """同步运行完整 Agent 轮次。

        Args:
            user_message: 用户消息。
            system_prompt_override: 系统提示覆盖。
            developer_instructions: 开发者指令。
            user_instructions: 用户指令。

        Returns:
            Agent 最终响应。
        """
        self.state = LoopState(phase=LoopPhase.ASSEMBLING)
        self.emitter.clear()

        turn_context = TurnContext(
            user_message=user_message,
            system_prompt_override=system_prompt_override,
            developer_instructions=developer_instructions,
            user_instructions=user_instructions,
        )

        # S6: 记录用户消息到工作记忆
        if self.memory is not None:
            self.memory.append_message(
                WorkingMemoryMessage(role="user", content=user_message)
            )

        # 输入护栏检查
        input_verdict = await self.guardrails.check_input(user_message)
        if input_verdict.tripwire or input_verdict.verdict == VerdictType.BLOCK:
            return self.make_response(
                content=f"输入被拒绝: {input_verdict.reason}",
                reason=TerminationReason.TRIPWIRE,
            )

        # S6: 获取记忆索引和语义检索
        memory_index_text = ""
        semantic_text = ""
        if self.memory is not None:
            index_entries = await self.memory.get_memory_index()
            if index_entries:
                memory_index_text = "\n".join(
                    f"- [{e.memory_type}] {e.summary}" for e in index_entries
                )
            results = await self.memory.search_memory(user_message, top_k=5)
            if results:
                semantic_text = "\n".join(
                    f"[{r.score:.2f}] {r.entry.content[:200]}" for r in results
                )

        # S14: 获取技能索引 + 自动激活
        skill_index_text = ""
        if self.skill_manager is not None:
            skill_entries = self.skill_manager.get_skill_index()
            if skill_entries:
                skill_index_text = "\n".join(
                    f"- {e.name}: {e.description}" for e in skill_entries
                )
                # 自动激活与当前任务相关的技能（注册技能脚本到工具注册表）
                self.skill_manager.auto_activate_for_task(user_message)

        # S5+S7: 刷新工具 Schema（含内置+MCP+技能脚本+披露工具）
        if self.tool_injector is not None:
            schemas = self.tool_injector.get_tools_for_stage()
            self.assembler.set_tool_schemas(schemas)

        # Plan-and-Execute 模式：注入步骤指令
        plan_context = self.strategy.get_plan_context()
        step_instruction = self.strategy.get_step_instruction()
        if step_instruction:
            turn_context.user_message += step_instruction

        # 主循环
        while True:
            self.state.current_turn += 1
            self.state.phase = LoopPhase.ASSEMBLING
            self.emitter.emit("turn_start", turn=self.state.current_turn)

            start_time = time.perf_counter()

            # Step 1: Prompt 组装（注入 S6 记忆 + S14 技能）
            prompt = self.assembler.assemble_prompt(
                turn_context,
                memory_index=memory_index_text,
                semantic_results=semantic_text or plan_context,
                skill_index=skill_index_text,
            )

            # Step 2: LLM 推理
            self.state.phase = LoopPhase.LLM_CALLING
            self.emitter.emit(
                "llm_request",
                turn=self.state.current_turn,
                data={"token_count": prompt.token_count},
            )

            response = await chat(
                self.gateway,
                prompt.messages,
                tools=prompt.tools if prompt.tools else None,
            )

            self.emitter.emit(
                "llm_response",
                turn=self.state.current_turn,
                data={
                    "has_content": bool(response.content),
                    "tool_call_count": len(response.tool_calls or []),
                },
            )

            # Step 3: 解析输出
            self.state.phase = LoopPhase.PARSING
            parsed = self.parser.parse(response)

            # 安全拒绝检测
            safety_refusal = response.finish_reason == "content_filter"

            # Step 4: 终止检查（自然终止/安全拒绝）
            reason = self.termination.evaluate(
                self.state,
                safety_refusal=safety_refusal,
                is_final_response=parsed.is_final,
                token_usage=self.assembler.get_token_usage(),
            )
            if reason is not None:
                # 输出护栏（仅自然终止时检查）
                if reason == TerminationReason.NATURAL and parsed.content:
                    out_verdict = await self.guardrails.check_output(parsed.content)
                    if out_verdict.tripwire or out_verdict.verdict == VerdictType.BLOCK:
                        return self.make_response(
                            content=f"输出被拒绝: {out_verdict.reason}",
                            reason=TerminationReason.TRIPWIRE,
                        )

                # S6: 记录助手响应到工作记忆
                if self.memory is not None and parsed.content:
                    self.memory.append_message(
                        WorkingMemoryMessage(role="assistant", content=parsed.content)
                    )
                self.assembler.update_with_response({
                    "role": "assistant",
                    "content": parsed.content,
                })
                return self.make_response(content=parsed.content, reason=reason)

            # Step 5: 工具执行
            self.state.phase = LoopPhase.TOOL_EXECUTING
            outcomes = await self.coordinator.execute_tool_calls(
                parsed.tool_calls, self.state.current_turn
            )

            # 检查绊线
            tripwire = any(
                o.skipped and "绊线" in o.skip_reason for o in outcomes
            )

            # 记录助手响应（含工具调用）
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": parsed.content or None,
            }
            if parsed.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in parsed.tool_calls
                ]
            self.assembler.update_with_response(assistant_msg)

            # 注入工具结果
            tool_results: list[dict[str, Any]] = []
            for outcome in outcomes:
                if outcome.result is not None:
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": outcome.tool_call.id,
                        "content": outcome.result.content if outcome.result.success
                        else f"错误: {outcome.result.error}",
                    })
                elif outcome.skipped:
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": outcome.tool_call.id,
                        "content": f"跳过: {outcome.skip_reason}",
                    })

            self.assembler.update_with_result(tool_results)
            self.state.total_tool_calls += len(outcomes)

            # S6: 记录助手响应（含工具调用）到工作记忆
            if self.memory is not None:
                summary_parts = []
                if parsed.content:
                    summary_parts.append(parsed.content)
                for o in outcomes:
                    if o.result is not None:
                        summary_parts.append(
                            f"[{o.tool_call.function.name}]: "
                            f"{o.result.content[:100] if o.result.success else o.result.error}"
                        )
                self.memory.append_message(
                    WorkingMemoryMessage(role="assistant", content="\n".join(summary_parts))
                )

            # S10: 可选变更后验证
            if self.verifier_registry is not None:
                successful_tools = [
                    o for o in outcomes
                    if o.result is not None and o.result.success
                ]
                if successful_tools:
                    verification_results = await self.verifier_registry.run_computational(
                        target={"tool_outcomes": [
                            {"name": o.tool_call.function.name, "result": o.result.content}
                            for o in successful_tools
                        ]},
                        phase=QualityPhase.POST_INTEGRATION,
                    )
                    for vr in verification_results:
                        self.emitter.emit(
                            "verification_result",
                            turn=self.state.current_turn,
                            data={"verifier": vr.verifier_name, "status": vr.status.value},
                        )

            # 更新 turn_context 为空消息（继续循环）
            turn_context = TurnContext(user_message="")

            # 终止检查（绊线/最大轮次/Token）
            reason = self.termination.evaluate(
                self.state,
                tripwire=tripwire,
                token_usage=self.assembler.get_token_usage(),
            )
            if reason is not None:
                return self.make_response(
                    content=parsed.content or "循环终止",
                    reason=reason,
                )

            # Plan-and-Execute: 推进步骤
            if not self.strategy.is_plan_complete():
                self.strategy.advance_step()
                plan_context = self.strategy.get_plan_context()

            self.emitter.emit("turn_end", turn=self.state.current_turn)

            elapsed = (time.perf_counter() - start_time) * 1000
            emit_metric("loop_turn_overhead_ms", elapsed, {}, "histogram")

    async def run_stream(
        self,
        user_message: str,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        """流式运行 Agent 轮次，逐事件输出。

        Args:
            user_message: 用户消息。
            **kwargs: 传递给 run() 的额外参数。

        Yields:
            AgentEvent 事件流。
        """
        collector = StreamCollector()
        self.emitter.add_listener(collector)

        async def run_task() -> AgentResponse:
            result = await self.run(user_message, **kwargs)
            self.emitter.emit(
                "termination",
                turn=self.state.current_turn,
                data={"reason": result.termination_reason.value},
            )
            collector.close()
            return result

        import asyncio
        task = asyncio.create_task(run_task())

        async for event in collector.iter_events():
            yield event

        await task
        self.emitter.remove_listener(collector)

    def abort(self) -> None:
        """中断循环。"""
        self.state.aborted = True
        log.info("循环中断请求已标记")

    def get_state(self) -> LoopState:
        """获取当前循环状态。"""
        return self.state.model_copy()

    def make_response(
        self,
        content: str,
        reason: TerminationReason,
    ) -> AgentResponse:
        """构造最终响应。"""
        self.state.phase = LoopPhase.TERMINATING
        self.state.termination_reason = reason

        self.emitter.emit(
            "termination",
            turn=self.state.current_turn,
            data={"reason": reason.value},
        )

        return AgentResponse(
            content=content,
            tool_calls_made=self.state.total_tool_calls,
            total_turns=self.state.current_turn,
            termination_reason=reason,
            events=self.emitter.get_events(),
        )
