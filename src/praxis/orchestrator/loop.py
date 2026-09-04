"""TAO/ReAct 核心循环引擎。

实现完整 TAO 循环：
消息记录 → Prompt 组装 → LLM 推理 → 工具执行 → 上下文更新 → 终止检查。
支持完整调用（run）和流式调用（run_stream）两种独立路径。
"""

import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from inspect import isawaitable
from typing import Any, cast

from json_repair import repair_json

from praxis.config.schemas import OrchestratorConfig
from praxis.context.assembler import PromptAssembler
from praxis.context.compaction import ContextCompactor
from praxis.context.jit_retrieval import JITRetriever
from praxis.context.masking import ObservationMasker
from praxis.context.tool_injection import ToolInjector
from praxis.gateway.calls import complete as chat
from praxis.gateway.calls import stream as chat_stream
from praxis.guardrails.engine import GuardrailEngine
from praxis.memory.core import CognitiveMemory
from praxis.models.context import AssembledPrompt, RunContext, TurnContext
from praxis.models.guardrails import VerdictType
from praxis.models.memory import WorkingMemoryMessage
from praxis.models.messages import ResolvedUserInput
from praxis.models.orchestrator import (
    AgentEvent,
    AgentResponse,
    LoopPhase,
    LoopState,
    StrategyMode,
    TerminationReason,
)
from praxis.models.responses import ModelResponse
from praxis.models.verification import QualityPhase, VerificationStatus, VerificationType
from praxis.orchestrator.engine import OrchestrationEngine
from praxis.orchestrator.events import EventEmitter
from praxis.orchestrator.parser import OutputParser, ParsedOutput, StreamAccumulator
from praxis.orchestrator.strategy import LoopStrategy, PlanStep
from praxis.orchestrator.termination import TerminationManager
from praxis.orchestrator.tool_coordination import ToolCallOutcome, ToolCoordinator
from praxis.protocols import ModelGateway
from praxis.skills.manager import SkillManager
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric
from praxis.verification.gav import GAVController, GAVVerifyRequest
from praxis.verification.registry import VerifierRegistry

log = get_logger("orchestrator.loop")

PLANNING_SYSTEM_PROMPT = (
    "你是任务规划专家。将用户任务分解为有序、可独立执行的步骤。\n"
    "要求：步骤 3~8 个，粒度适中、彼此衔接、覆盖完整任务。\n"
    "仅输出 JSON 数组，每项为 {\"description\": \"步骤描述\", \"tool_hint\": \"可选工具提示\"}，"
    "不要输出任何额外文字。"
)


class OrchestrationLoop:
    """TAO/ReAct 核心循环引擎。

    协调 S4（LLM）、S5（工具）、S6（记忆）、S7（上下文）、
    S8（护栏）、S9（恢复）、S10（验证）、S14（技能）
    完成完整的 Agent 轮次。
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        gateway: ModelGateway,
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
        compaction_min_history: int = 8,
        jit_retriever: JITRetriever | None = None,
        model: str = "default",
    ) -> None:
        self.config = config
        self.gateway = gateway
        # 本会话用于 LLM 推理的模型别名；传给 chat/chat_stream，否则恒用网关 default_model
        self.model = model
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
        self.compaction_min_history = compaction_min_history
        # S10→S11 验证反馈控制器（GAV Verify 阶段编排与上下文注入格式化）
        self.gav = GAVController()
        # S7 即时检索：按任务阶段注入 few-shot 示例与轻量标识符索引
        self.jit_retriever = jit_retriever
        self.state = LoopState()
        # 最近一次在关键节点（turn_start / turn_end / termination）发射的事件，
        # 供流式路径精确 yield，避免依赖 emitter.events[-1] 受子系统插入事件影响。
        self.latest_event: AgentEvent | None = None
        self.engine = OrchestrationEngine(self)

    # ── 公共准备方法 ─────────────────────────────────────────────────────

    def init_run(
        self,
        user_input: ResolvedUserInput,
        system_prompt_override: str | None = None,
        developer_instructions: str = "",
        user_instructions: str = "",
        task_stage: str = "general",
    ) -> RunContext:
        """初始化一次 run 的状态和上下文。"""
        self.state = LoopState(phase=LoopPhase.ASSEMBLING)
        self.emitter.begin_run()
        self.strategy.begin_request()
        turn_context = TurnContext(
            user_content=user_input.content,
            user_text=user_input.text_projection,
            system_prompt_override=system_prompt_override,
            developer_instructions=developer_instructions,
            user_instructions=user_instructions,
            task_stage=task_stage,
        )
        return RunContext(
            turn_context=turn_context,
            safe_input_projection=user_input.text_projection,
        )

    async def prepare_run(
        self,
        ctx: RunContext,
    ) -> AgentResponse | None:
        """执行循环前的准备工作：记忆记录、输入护栏、记忆检索、技能加载、工具注入、策略。

        返回 AgentResponse 表示提前终止（如输入被拦截），None 表示继续。
        """
        # S6: 记录用户消息到工作记忆
        user_text = ctx.turn_context.user_text

        if self.memory is not None:
            self.memory.append_message(
                WorkingMemoryMessage(role="user", content=user_text)
            )

        # 输入护栏检查
        input_verdict = await self.guardrails.check_input(user_text)
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
            results = await self.memory.search_memory(user_text, top_k=5)
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
                self.skill_manager.auto_activate_for_task(user_text)

        # S7 JIT: 按任务阶段取 few-shot 示例 + 标识符索引
        if self.jit_retriever is not None:
            ctx.few_shot_messages = self.jit_retriever.get_examples_for_task(
                ctx.turn_context.task_stage
            )
            id_index = self.jit_retriever.get_identifier_index()
            if id_index:
                ctx.identifier_index = "\n".join(
                    f"- [{e['kind']}] {e['identifier']} ({e['source']})" for e in id_index
                )

        # S5+S7: 刷新工具 Schema
        if self.tool_injector is not None:
            schemas = self.tool_injector.get_tools_for_stage(ctx.turn_context.task_stage)
            self.assembler.set_tool_schemas(schemas)

        # Plan-and-Execute 模式：首次进入时先生成计划（否则 plan 恒空，退化为 ReAct）
        if self.strategy.mode == StrategyMode.PLAN_AND_EXECUTE and not self.strategy.plan:
            await self.generate_plan(user_text)

        # Plan-and-Execute 模式：注入步骤指令
        step_instruction = self.strategy.get_step_instruction()
        if step_instruction:
            ctx.turn_context.user_instructions += step_instruction

        return None

    async def generate_plan(self, user_text: str) -> None:
        """Plan-and-Execute：调用 LLM 将任务分解为有序步骤并 set_plan。

        失败或解析不出步骤时静默退化为 ReAct（plan 留空）。
        """
        self.state.phase = LoopPhase.PLANNING
        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        try:
            response = await chat(self.gateway, messages, model=self.model)
        except Exception as exc:  # 规划失败不应中断主流程
            log.warning("计划生成失败，退化为 ReAct", error=str(exc))
            return

        data = cast(object, repair_json(response.content or "[]", return_objects=True))
        steps: list[PlanStep] = []
        if isinstance(data, list):
            for raw_item in cast(list[object], data):
                if isinstance(raw_item, dict):
                    item = cast(dict[str, Any], raw_item)
                else:
                    continue
                if item.get("description"):
                    steps.append(PlanStep(
                        description=str(item["description"]),
                        tool_hint=str(item.get("tool_hint", "")),
                    ))
        if steps:
            self.strategy.set_plan(steps)
            self.emitter.emit(
                "plan_created",
                turn=self.state.current_turn,
                data={
                    "request_id": self.strategy.request_id,
                    "step_count": len(steps),
                },
            )
        else:
            log.info("未解析出有效计划步骤，退化为 ReAct")

    async def prepare_turn(self, ctx: RunContext) -> AssembledPrompt:
        """每轮迭代前的准备：遮蔽、压缩、Prompt 组装、历史追加。"""
        self.state.current_turn += 1
        self.state.phase = LoopPhase.ASSEMBLING
        self.latest_event = self.emitter.emit("turn_start", turn=self.state.current_turn)

        # S7 观察遮蔽与压缩仅在历史足够长时触发（阈值由 S7 配置驱动）
        history_len = len(self.assembler.conversation_history)
        if history_len >= self.compaction_min_history:
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
                    active_message = ctx.input_history_message
                    if active_message is not None:
                        active_index = next(
                            (
                                index
                                for index, message in enumerate(
                                    self.assembler.conversation_history
                                )
                                if message is active_message
                            ),
                            None,
                        )
                        if active_index is None:
                            raise RuntimeError(
                                "context compaction discarded the active user input"
                            )
                        ctx.input_history_index = active_index

        # Step 1: Prompt 组装（注入 S6 记忆 + S14 技能 + 计划进度）
        # 计划上下文与语义检索并存（此前用 or 互斥：有记忆结果时计划进度会丢失）
        plan_context = self.strategy.get_plan_context()
        semantic_block = "\n\n".join(
            s for s in (ctx.semantic_results, plan_context) if s
        )
        prompt = self.assembler.assemble_prompt(
            ctx.turn_context,
            memory_index=ctx.memory_index,
            semantic_results=semantic_block,
            skill_index=ctx.skill_index,
            identifier_index=ctx.identifier_index,
            few_shot_messages=ctx.few_shot_messages,
        )

        # 将用户消息存入对话历史
        if ctx.input_history_index is None and ctx.turn_context.user_content:
            current_message = prompt.messages[-1]
            history_message = {
                "role": "user",
                "content": current_message["content"],
            }
            self.assembler.conversation_history.append(history_message)
            ctx.input_history_index = len(self.assembler.conversation_history) - 1
            ctx.input_history_message = history_message

        return prompt

    def sanitize_run_input(self, ctx: RunContext) -> None:
        """Replace active provider content with its safe text projection."""
        history = self.assembler.conversation_history
        index = ctx.input_history_index
        history_message = ctx.input_history_message

        live_message = next(
            (
                message
                for message in history
                if history_message is not None and message is history_message
            ),
            None,
        )
        if live_message is None and index is not None and 0 <= index < len(history):
            indexed_message = history[index]
            is_structured_user = (
                indexed_message.get("role") == "user"
                and isinstance(indexed_message.get("content"), list)
            )
            matches_active_input = (
                history_message is None or indexed_message == history_message
            )
            if is_structured_user and matches_active_input:
                live_message = indexed_message

        safe_message = {
            "role": "user",
            "content": ctx.safe_input_projection,
        }
        if live_message is not None:
            live_message.clear()
            live_message.update(safe_message)
            return

        for message in history:
            if (
                message.get("role") == "user"
                and isinstance(message.get("content"), list)
            ):
                message.clear()
                message.update({
                    "role": "user",
                    "content": ctx.safe_input_projection,
                })

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
    ) -> tuple[list[ToolCallOutcome], bool]:
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
            summary_parts: list[str] = []
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
                verified_outcomes: list[dict[str, str]] = []
                for outcome in successful_tools:
                    result = outcome.result
                    if result is not None:
                        verified_outcomes.append({
                            "name": outcome.tool_call.function.name,
                            "result": result.content,
                        })
                verification_results = await self.verifier_registry.run_computational(
                    target={"tool_outcomes": verified_outcomes},
                    phase=QualityPhase.POST_INTEGRATION,
                )
                for vr in verification_results:
                    self.emitter.emit(
                        "verification_result",
                        turn=self.state.current_turn,
                        data={"verifier": vr.verifier_name, "status": vr.status.value},
                    )
                # S10→S11 反馈：GAV 评估，验证失败时把结构化反馈注入上下文，
                # 供下一轮 LLM 自我修正（否则验证结果仅产生事件、无人消费）。
                if verification_results:
                    gav_response = self.gav.evaluate(GAVVerifyRequest(
                        results=verification_results,
                        quadrant=GAVController.select_quadrant(
                            is_feedforward=False,
                            verification_type=VerificationType.COMPUTATIONAL,
                        ),
                    ))
                    # 验证基础设施故障同样不能静默批准模型输出。
                    has_failure = any(
                        r.status in {VerificationStatus.FAIL, VerificationStatus.ERROR}
                        for r in verification_results
                    )
                    if not gav_response.passed and has_failure:
                        feedback = GAVController.format_for_context(gav_response)
                        self.assembler.update_with_result([{
                            "role": "system",
                            "content": f"<verification_feedback>\n{feedback}\n</verification_feedback>",
                        }])
                        self.emitter.emit(
                            "gav_feedback",
                            turn=self.state.current_turn,
                            data={"passed": False, "retry_hint": gav_response.retry_hint},
                        )

        return outcomes, tripwire

    def check_handoff_result(
        self, parsed: ParsedOutput, outcomes: list[ToolCallOutcome],
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
            user_content=None,
            user_text="",
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

        self.latest_event = self.emitter.emit("turn_end", turn=self.state.current_turn)

        elapsed = (time.perf_counter() - start_time) * 1000
        emit_metric("loop_turn_overhead_ms", elapsed, {}, "histogram")

        return None

    def emit_model_request(self, prompt: AssembledPrompt) -> AgentEvent:
        """Record a normalized model-request event."""
        self.state.phase = LoopPhase.LLM_CALLING
        return self.emitter.emit(
            "llm_request",
            turn=self.state.current_turn,
            data={"token_count": prompt.token_count},
        )

    def emit_model_response(self, response: ModelResponse) -> AgentEvent:
        """Record mode-neutral response metadata for accounting and diagnostics."""
        return self.emitter.emit(
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

    async def complete_model(self, prompt: AssembledPrompt) -> ModelResponse:
        """Execute the complete-response model phase."""
        self.emit_model_request(prompt)
        response = await chat(
            self.gateway,
            prompt.messages,
            model=self.model,
            tools=prompt.tools if prompt.tools else None,
        )
        self.emit_model_response(response)
        return response

    async def stream_model(
        self,
        prompt: AssembledPrompt,
    ) -> AsyncGenerator[ModelResponse | None, None]:
        """Execute the streaming model phase and expose event flush boundaries."""
        self.emit_model_request(prompt)
        yield None
        accumulator = StreamAccumulator()
        response_stream = chat_stream(
            self.gateway,
            prompt.messages,
            model=self.model,
            tools=prompt.tools if prompt.tools else None,
        )
        try:
            async for chunk in response_stream:
                delta = accumulator.feed(chunk)
                if delta.reasoning:
                    self.emitter.emit(
                        "reasoning_delta",
                        turn=self.state.current_turn,
                        data={"text": delta.reasoning},
                    )
                    yield None
                if delta.content:
                    self.emitter.emit(
                        "content_delta",
                        turn=self.state.current_turn,
                        data={"text": delta.content},
                    )
                    yield None
        finally:
            close_stream = getattr(response_stream, "aclose", None)
            if callable(close_stream):
                close_result = close_stream()
                if isawaitable(close_result):
                    await close_result

        response = accumulator.build_response()
        self.emit_model_response(response)
        yield response

    def parse_model_response(self, response: ModelResponse) -> ParsedOutput:
        """Parse a normalized model response and enter the parsing phase."""
        self.state.phase = LoopPhase.PARSING
        return self.parser.parse(response)

    # ── 完整调用路径 ───────────────────────────────────────────────────────────────

    async def run(
        self,
        user_input: ResolvedUserInput,
        system_prompt_override: str | None = None,
        developer_instructions: str = "",
        user_instructions: str = "",
        task_stage: str = "general",
    ) -> AgentResponse:
        """完整调用运行 Agent 轮次。

        Args:
            user_input: 已解析的用户输入。
            system_prompt_override: 系统提示覆盖。
            developer_instructions: 开发者指令。
            user_instructions: 用户指令。
            task_stage: 任务阶段（用于工具集过滤），默认 ``general``。

        Returns:
            Agent 最终响应。
        """
        ctx = self.init_run(
            user_input, system_prompt_override,
            developer_instructions, user_instructions, task_stage,
        )

        try:
            return await self.execute_run(ctx)
        finally:
            self.sanitize_run_input(ctx)

    async def execute_run(self, ctx: RunContext) -> AgentResponse:
        """Reduce the shared transition stream to its terminal response."""
        terminal: AgentResponse | None = None
        async for transition in self.engine.transitions(ctx, streaming=False):
            if transition.response is not None:
                terminal = transition.response
        if terminal is None:
            raise RuntimeError("编排状态机结束但未产生响应")
        return terminal

    # ── 流式调用路径 ───────────────────────────────────────────────────────────────

    async def run_stream(
        self,
        user_input: ResolvedUserInput,
        system_prompt_override: str | None = None,
        developer_instructions: str = "",
        user_instructions: str = "",
        task_stage: str = "general",
    ) -> AsyncGenerator[AgentEvent, None]:
        """流式运行 Agent 轮次，逐事件 yield。

        Args:
            user_input: 已解析的用户输入。
            system_prompt_override: 系统提示覆盖。
            developer_instructions: 开发者指令。
            user_instructions: 用户指令。
            task_stage: 任务阶段（用于工具集过滤），默认 ``general``。

        Yields:
            AgentEvent 事件流。
        """
        ctx = self.init_run(
            user_input, system_prompt_override,
            developer_instructions, user_instructions, task_stage,
        )

        stream = self.stream_run(ctx)
        try:
            async with aclosing(stream):
                async for event in stream:
                    yield event
        finally:
            self.sanitize_run_input(ctx)

    async def stream_run(self, ctx: RunContext) -> AsyncGenerator[AgentEvent, None]:
        """Project public events from the same transition engine used by run()."""
        async for transition in self.engine.transitions(ctx, streaming=True):
            if transition.event is not None:
                yield transition.event

    # ── 控制方法 ──────────────────────────────────────────────────────────

    def abort(self) -> None:
        """中断循环。"""
        self.state.aborted = True
        log.info("循环中断请求已标记")

    def last_event(self) -> AgentEvent:
        """返回最近事件；内部状态违反事件投影约束时立即失败。"""
        if self.latest_event is None:
            raise RuntimeError("编排循环尚未产生事件")
        return self.latest_event

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

        self.latest_event = self.emitter.emit(
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
