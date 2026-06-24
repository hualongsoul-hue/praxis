"""TAO/ReAct 核心循环引擎。

实现完整 TAO 循环：
消息记录 → Prompt 组装 → LLM 推理 → 工具执行 → 上下文更新 → 终止检查。
支持完整调用（run）和流式调用（run_stream）两种独立路径。
"""

import time
from collections.abc import AsyncIterator
from typing import Any

from praxis.config.schemas import OrchestratorConfig
from praxis.context.assembler import PromptAssembler
from praxis.context.compaction import ContextCompactor
from praxis.context.masking import ObservationMasker
from praxis.context.tool_injection import ToolInjector
from praxis.gateway.chat import chat, chat_stream
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine
from praxis.memory.core import CognitiveMemory
from praxis.models.context import AssembledPrompt, RunContext, TurnContext
from praxis.models.guardrails import VerdictType
from praxis.models.memory import WorkingMemoryMessage
from praxis.models.orchestrator import (
    AgentEvent,
    AgentResponse,
    LoopPhase,
    LoopState,
    TerminationReason,
)

from praxis.models.verification import QualityPhase
from praxis.orchestrator.events import EventEmitter
from praxis.orchestrator.parser import OutputParser, ParsedOutput, StreamAccumulator
from praxis.orchestrator.strategy import LoopStrategy
from praxis.orchestrator.termination import TerminationManager
from praxis.orchestrator.tool_coordination import ToolCoordinator
from praxis.skills.manager import SkillManager
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric
from praxis.verification.registry import VerifierRegistry

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
        memory: CognitiveMemory | None = None,
        verifier_registry: VerifierRegistry | None = None,
        skill_manager: SkillManager | None = None,
        compactor: ContextCompactor | None = None,
        masker: ObservationMasker | None = None,
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
        self.compactor = compactor
        self.masker = masker
        self.state = LoopState()
        # 最近一次在关键节点（turn_start / turn_end / termination）发射的事件，
        # 供流式路径精确 yield，避免依赖 emitter.events[-1] 受子系统插入事件影响。
        self._last_event: AgentEvent | None = None

    # ── 公共准备方法 ─────────────────────────────────────────────────────

    def init_run(
        self,
        user_message: str,
        system_prompt_override: str | None = None,
        developer_instructions: str = "",
        user_instructions: str = "",
        task_stage: str = "general",
    ) -> RunContext:
        """初始化一次 run 的状态和上下文。"""
        self.state = LoopState(phase=LoopPhase.ASSEMBLING)
        self.emitter.clear()
        turn_context = TurnContext(
            user_message=user_message,
            system_prompt_override=system_prompt_override,
            developer_instructions=developer_instructions,
            user_instructions=user_instructions,
            task_stage=task_stage,
        )
        return RunContext(turn_context=turn_context)

    async def prepare_run(
        self,
        user_message: str,
        ctx: RunContext,
    ) -> AgentResponse | None:
        """执行循环前的准备工作：记忆记录、输入护栏、记忆检索、技能加载、工具注入、策略。

        返回 AgentResponse 表示提前终止（如输入被拦截），None 表示继续。
        """
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
        if self.memory is not None:
            index_entries = await self.memory.get_memory_index()
            if index_entries:
                ctx.memory_index = "\n".join(
                    f"- [{e.memory_type}] {e.summary}" for e in index_entries
                )
            results = await self.memory.search_memory(user_message, top_k=5)
            if results:
                ctx.semantic_results = "\n".join(
                    f"[{r.relevance_score:.2f}] {r.entry.content[:200]}" for r in results
                )

        # S14: 获取技能索引 + 自动激活
        if self.skill_manager is not None:
            skill_entries = self.skill_manager.get_skill_index()
            if skill_entries:
                ctx.skill_index = "\n".join(
                    f"- {e.name}: {e.description}" for e in skill_entries
                )
                self.skill_manager.auto_activate_for_task(user_message)

        # S5+S7: 刷新工具 Schema
        if self.tool_injector is not None:
            schemas = self.tool_injector.get_tools_for_stage(ctx.turn_context.task_stage)
            self.assembler.set_tool_schemas(schemas)

        # Plan-and-Execute 模式：注入步骤指令
        step_instruction = self.strategy.get_step_instruction()
        if step_instruction:
            ctx.turn_context.user_message += step_instruction

        return None

    async def prepare_turn(self, ctx: RunContext) -> AssembledPrompt:
        """每轮迭代前的准备：遮蔽、压缩、Prompt 组装、历史追加。"""
        self.state.current_turn += 1
        self.state.phase = LoopPhase.ASSEMBLING
        self._last_event = self.emitter.emit("turn_start", turn=self.state.current_turn)

        # S7 观察遮蔽与压缩仅在历史足够长时触发
        history_len = len(self.assembler.conversation_history)
        if history_len >= 8:
            if self.masker is not None:
                self.masker.apply_masking(
                    self.assembler.conversation_history,
                    self.state.current_turn,
                )
            if self.compactor is not None:
                pre_usage = self.assembler.get_token_usage()
                if pre_usage.compaction_needed:
                    await self.compactor.compact(
                        self.assembler.conversation_history,
                        self.assembler.file_refs,
                    )
                    self.assembler.compaction_count += 1

        # Step 1: Prompt 组装（注入 S6 记忆 + S14 技能）
        prompt = self.assembler.assemble_prompt(
            ctx.turn_context,
            memory_index=ctx.memory_index,
            semantic_results=ctx.semantic_results or self.strategy.get_plan_context(),
            skill_index=ctx.skill_index,
        )

        # 将用户消息存入对话历史
        if ctx.turn_context.user_message:
            self.assembler.conversation_history.append(
                {"role": "user", "content": ctx.turn_context.user_message}
            )

        return prompt

    # ── 公共后处理方法 ────────────────────────────────────────────────────

    def check_final_termination(
        self, parsed: ParsedOutput, finish_reason: str | None,
    ) -> TerminationReason | None:
        """终止条件检查（自然终止 / 安全拒绝）。"""
        safety_refusal = finish_reason == "content_filter"
        return self.termination.evaluate(
            self.state,
            safety_refusal=safety_refusal,
            is_final_response=parsed.is_final,
            token_usage=self.assembler.get_token_usage(),
        )

    async def handle_final_response(
        self,
        parsed: ParsedOutput,
        reason: TerminationReason,
    ) -> AgentResponse:
        """处理最终响应：输出护栏、记忆记录、历史更新。"""
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

    async def process_tool_outcomes(
        self,
        parsed: ParsedOutput,
    ) -> tuple[list[Any], bool]:
        """执行工具调用并处理结果。返回 (outcomes, tripwire)。"""
        self.state.phase = LoopPhase.TOOL_EXECUTING
        outcomes = await self.coordinator.execute_tool_calls(
            parsed.tool_calls, self.state.current_turn
        )

        # 绊线检查
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
            elif outcome.needs_user_confirm:
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": outcome.tool_call.id,
                    "content": f"需要用户确认: {outcome.skip_reason}",
                })

        self.assembler.update_with_result(tool_results)
        self.state.total_tool_calls += len(outcomes)

        # S6: 记录工具调用摘要到工作记忆
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

        return outcomes, tripwire

    def check_handoff_result(
        self, parsed: ParsedOutput, outcomes: list[Any],
    ) -> AgentResponse | None:
        """检查 Handoff 短路。有匹配结果则返回 AgentResponse，否则 None。"""
        if not parsed.handoff_target:
            return None
        for outcome in outcomes:
            if (
                not outcome.skipped
                and outcome.result is not None
                and outcome.result.success
                and outcome.tool_call.function.name.startswith("handoff_to_")
            ):
                # 工具调用计数已在 process_tool_outcomes 累加，此处不再重复累加
                return self.make_response(
                    content=outcome.result.content,
                    reason=TerminationReason.HANDOFF,
                )
        return None

    def finish_turn(
        self,
        ctx: RunContext,
        tripwire: bool,
        start_time: float,
    ) -> AgentResponse | None:
        """轮次末尾：终止检查、策略推进、发射 turn_end。返回 AgentResponse 表示终止。"""
        # 清空用户消息但保留系统/开发者/用户指令配置
        ctx.turn_context = TurnContext(
            user_message="",
            system_prompt_override=ctx.turn_context.system_prompt_override,
            developer_instructions=ctx.turn_context.developer_instructions,
            user_instructions=ctx.turn_context.user_instructions,
            task_stage=ctx.turn_context.task_stage,
        )

        reason = self.termination.evaluate(
            self.state,
            tripwire=tripwire,
            token_usage=self.assembler.get_token_usage(),
        )
        if reason is not None:
            return self.make_response(
                content="循环终止",
                reason=reason,
            )

        # Plan-and-Execute: 推进步骤
        if not self.strategy.is_plan_complete():
            self.strategy.advance_step()

        self._last_event = self.emitter.emit("turn_end", turn=self.state.current_turn)

        elapsed = (time.perf_counter() - start_time) * 1000
        emit_metric("loop_turn_overhead_ms", elapsed, {}, "histogram")

        return None

    # ── 完整调用路径 ───────────────────────────────────────────────────────────────

    async def run(
        self,
        user_message: str,
        system_prompt_override: str | None = None,
        developer_instructions: str = "",
        user_instructions: str = "",
        task_stage: str = "general",
    ) -> AgentResponse:
        """完整调用运行 Agent 轮次。

        Args:
            user_message: 用户消息。
            system_prompt_override: 系统提示覆盖。
            developer_instructions: 开发者指令。
            user_instructions: 用户指令。
            task_stage: 任务阶段（用于工具集过滤），默认 ``general``。

        Returns:
            Agent 最终响应。
        """
        ctx = self.init_run(
            user_message, system_prompt_override,
            developer_instructions, user_instructions, task_stage,
        )

        early = await self.prepare_run(user_message, ctx)
        if early is not None:
            return early

        while True:
            start_time = time.perf_counter()
            prompt = await self.prepare_turn(ctx)

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
                    "has_reasoning": bool(response.reasoning_content),
                    "has_refusal": bool(response.refusal),
                    "tool_call_count": len(response.tool_calls or []),
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "reasoning_tokens": response.usage.reasoning_tokens,
                    "cached_prompt_tokens": response.usage.cached_prompt_tokens,
                },
            )

            # Step 3: 解析输出
            self.state.phase = LoopPhase.PARSING
            parsed = self.parser.parse(response)

            # Step 4: 终止检查
            reason = self.check_final_termination(parsed, response.finish_reason)
            if reason is not None:
                return await self.handle_final_response(parsed, reason)

            # Step 5: 工具执行
            outcomes, tripwire = await self.process_tool_outcomes(parsed)

            # Handoff 短路
            handoff_resp = self.check_handoff_result(parsed, outcomes)
            if handoff_resp is not None:
                return handoff_resp

            # 轮次结束
            end_resp = self.finish_turn(ctx, tripwire, start_time)
            if end_resp is not None:
                return end_resp

    # ── 流式调用路径 ───────────────────────────────────────────────────────────────

    async def run_stream(
        self,
        user_message: str,
        system_prompt_override: str | None = None,
        developer_instructions: str = "",
        user_instructions: str = "",
        task_stage: str = "general",
    ) -> AsyncIterator[AgentEvent]:
        """流式运行 Agent 轮次，逐事件 yield。

        Args:
            user_message: 用户消息。
            system_prompt_override: 系统提示覆盖。
            developer_instructions: 开发者指令。
            user_instructions: 用户指令。
            task_stage: 任务阶段（用于工具集过滤），默认 ``general``。

        Yields:
            AgentEvent 事件流。
        """
        ctx = self.init_run(
            user_message, system_prompt_override,
            developer_instructions, user_instructions, task_stage,
        )

        early = await self.prepare_run(user_message, ctx)
        if early is not None:
            yield self._last_event
            return

        while True:
            start_time = time.perf_counter()
            prompt = await self.prepare_turn(ctx)

            # yield turn_start 事件
            yield self._last_event

            # LLM 推理
            self.state.phase = LoopPhase.LLM_CALLING
            llm_req_event = self.emitter.emit(
                "llm_request",
                turn=self.state.current_turn,
                data={"token_count": prompt.token_count},
            )
            yield llm_req_event

            # 逐 chunk 消费响应
            accumulator = StreamAccumulator()
            async for chunk in chat_stream(
                self.gateway,
                prompt.messages,
                tools=prompt.tools if prompt.tools else None,
            ):
                delta = accumulator.feed(chunk)
                if delta.reasoning:
                    reasoning_event = self.emitter.emit(
                        "reasoning_delta",
                        turn=self.state.current_turn,
                        data={"text": delta.reasoning},
                    )
                    yield reasoning_event
                if delta.content:
                    delta_event = self.emitter.emit(
                        "content_delta",
                        turn=self.state.current_turn,
                        data={"text": delta.content},
                    )
                    yield delta_event

            response = accumulator.build_response()

            llm_resp_event = self.emitter.emit(
                "llm_response",
                turn=self.state.current_turn,
                data={
                    "has_content": bool(response.content),
                    "has_reasoning": bool(response.reasoning_content),
                    "has_refusal": bool(response.refusal),
                    "tool_call_count": len(response.tool_calls or []),
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "reasoning_tokens": response.usage.reasoning_tokens,
                    "cached_prompt_tokens": response.usage.cached_prompt_tokens,
                },
            )
            yield llm_resp_event

            # 解析输出
            self.state.phase = LoopPhase.PARSING
            parsed = self.parser.parse(response)

            # 终止检查
            reason = self.check_final_termination(parsed, response.finish_reason)
            if reason is not None:
                await self.handle_final_response(parsed, reason)
                yield self._last_event
                return

            # 工具执行（tool_call_start/end 事件由 coordinator 通过 emitter 产生）
            pre_event_count = len(self.emitter.events)
            outcomes, tripwire = await self.process_tool_outcomes(parsed)

            # yield 工具执行期间产生的所有事件
            for event in self.emitter.events[pre_event_count:]:
                yield event

            # Handoff 短路
            handoff_resp = self.check_handoff_result(parsed, outcomes)
            if handoff_resp is not None:
                yield self._last_event
                return

            # 轮次结束
            end_resp = self.finish_turn(ctx, tripwire, start_time)
            if end_resp is not None:
                yield self._last_event
                return

            # yield turn_end 事件
            yield self._last_event

    # ── 控制方法 ──────────────────────────────────────────────────────────

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

        self._last_event = self.emitter.emit(
            "termination",
            turn=self.state.current_turn,
            data={"reason": reason.value, "content": content},
        )

        return AgentResponse(
            content=content,
            tool_calls_made=self.state.total_tool_calls,
            total_turns=self.state.current_turn,
            termination_reason=reason,
            events=self.emitter.get_events(),
        )
