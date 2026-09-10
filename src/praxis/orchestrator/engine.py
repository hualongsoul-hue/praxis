"""Single transition engine shared by streaming and complete orchestration."""

import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Protocol

from praxis.models.context import AssembledPrompt, RunContext
from praxis.models.orchestrator import AgentEvent, AgentResponse, TerminationReason
from praxis.models.responses import ModelResponse
from praxis.orchestrator.events import EventEmitter
from praxis.orchestrator.parser import ParsedOutput
from praxis.orchestrator.tool_coordination import ToolCallOutcome


@dataclass(frozen=True, slots=True)
class OrchestrationTransition:
    """One observable event or the terminal response of an orchestration run."""

    event: AgentEvent | None = None
    response: AgentResponse | None = None

    def __post_init__(self) -> None:
        if (self.event is None) == (self.response is None):
            raise ValueError("transition 必须且只能包含 event 或 response")


class OrchestrationDriver(Protocol):
    """Operations supplied by :class:`OrchestrationLoop` to the transition engine."""

    emitter: EventEmitter

    async def prepare_run(self, ctx: RunContext) -> AgentResponse | None: ...

    async def prepare_turn(self, ctx: RunContext) -> AssembledPrompt: ...

    async def complete_model(self, prompt: AssembledPrompt) -> ModelResponse: ...

    def stream_model(
        self,
        prompt: AssembledPrompt,
    ) -> AsyncGenerator[ModelResponse | None, None]: ...

    def parse_model_response(self, response: ModelResponse) -> ParsedOutput: ...

    def check_final_termination(
        self,
        parsed: ParsedOutput,
        finish_reason: str | None,
    ) -> TerminationReason | None: ...

    async def handle_final_response(
        self,
        parsed: ParsedOutput,
        reason: TerminationReason,
    ) -> AgentResponse: ...

    async def process_tool_outcomes(
        self,
        parsed: ParsedOutput,
    ) -> tuple[list[ToolCallOutcome], bool]: ...

    def check_handoff_result(
        self,
        parsed: ParsedOutput,
        outcomes: list[ToolCallOutcome],
    ) -> AgentResponse | None: ...

    def finish_turn(
        self,
        ctx: RunContext,
        tripwire: bool,
        start_time: float,
    ) -> AgentResponse | None: ...


class OrchestrationEngine:
    """Own the only turn-control loop and expose mode-neutral transitions."""

    def __init__(self, driver: OrchestrationDriver) -> None:
        self.driver = driver

    async def transitions(
        self,
        context: RunContext,
        *,
        streaming: bool,
    ) -> AsyncGenerator[OrchestrationTransition, None]:
        """Run orchestration and yield every event plus one terminal response."""
        event_cursor = 0

        early_response = await self.driver.prepare_run(context)
        for event in self.driver.emitter.events[event_cursor:]:
            yield OrchestrationTransition(event=event)
        event_cursor = len(self.driver.emitter.events)
        if early_response is not None:
            yield OrchestrationTransition(response=early_response)
            return

        while True:
            start_time = time.perf_counter()
            prompt = await self.driver.prepare_turn(context)
            for event in self.driver.emitter.events[event_cursor:]:
                yield OrchestrationTransition(event=event)
            event_cursor = len(self.driver.emitter.events)

            response: ModelResponse | None = None
            if streaming:
                async with aclosing(self.driver.stream_model(prompt)) as model_stream:
                    async for model_update in model_stream:
                        for event in self.driver.emitter.events[event_cursor:]:
                            yield OrchestrationTransition(event=event)
                        event_cursor = len(self.driver.emitter.events)
                        if model_update is not None:
                            response = model_update
            else:
                response = await self.driver.complete_model(prompt)
                for event in self.driver.emitter.events[event_cursor:]:
                    yield OrchestrationTransition(event=event)
                event_cursor = len(self.driver.emitter.events)

            if response is None:
                raise RuntimeError("模型阶段结束但未产生完整响应")

            parsed = self.driver.parse_model_response(response)
            reason = self.driver.check_final_termination(parsed, response.finish_reason)
            if reason is not None:
                terminal = await self.driver.handle_final_response(parsed, reason)
                for event in self.driver.emitter.events[event_cursor:]:
                    yield OrchestrationTransition(event=event)
                yield OrchestrationTransition(response=terminal)
                return

            outcomes, tripwire = await self.driver.process_tool_outcomes(parsed)
            for event in self.driver.emitter.events[event_cursor:]:
                yield OrchestrationTransition(event=event)
            event_cursor = len(self.driver.emitter.events)

            handoff = self.driver.check_handoff_result(parsed, outcomes)
            if handoff is not None:
                for event in self.driver.emitter.events[event_cursor:]:
                    yield OrchestrationTransition(event=event)
                yield OrchestrationTransition(response=handoff)
                return

            terminal = self.driver.finish_turn(context, tripwire, start_time)
            for event in self.driver.emitter.events[event_cursor:]:
                yield OrchestrationTransition(event=event)
            if terminal is not None:
                yield OrchestrationTransition(response=terminal)
                return
