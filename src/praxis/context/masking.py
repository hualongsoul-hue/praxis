"""观察遮蔽（Observation Masking）。

隐藏旧工具输出内容但保持调用记录可见，
策略可配置：按轮次距离（默认 >10 轮）、按输出大小（默认 >2000 Token）。
"""

from typing import Any

from praxis.config.schemas import ContextConfig
from praxis.gateway.metering import get_token_count
from praxis.telemetry.logger import get_logger

log = get_logger("context.masking")

MASKED_PLACEHOLDER = "[工具输出已遮蔽——调用记录保留]"


class ObservationMasker:
    """观察遮蔽器。

    按轮次距离和输出大小对旧工具输出进行遮蔽，
    保持工具调用记录可见以维持对话连贯性。
    """

    def __init__(self, config: ContextConfig, model: str = "default") -> None:
        self.turn_distance = config.masking_turn_distance
        self.token_threshold = config.masking_token_threshold
        self.model = model

    def apply_masking(
        self,
        messages: list[dict[str, Any]],
        current_turn: int,
    ) -> int:
        """对消息列表应用观察遮蔽。

        Args:
            messages: 消息列表（原地修改）。
            current_turn: 当前轮次序号。

        Returns:
            遮蔽的消息数。
        """
        masked_count = 0
        turn_counter = 0

        for i, msg in enumerate(messages):
            role = msg.get("role", "")

            if role == "assistant":
                turn_counter += 1

            if role != "tool":
                continue

            distance = current_turn - turn_counter
            content = msg.get("content", "")

            should_mask = False

            if distance > self.turn_distance:
                should_mask = True

            if not should_mask and isinstance(content, str):
                token_count = get_token_count(
                    [{"role": "user", "content": content}],
                    self.model,
                )
                if token_count > self.token_threshold:
                    should_mask = True

            if should_mask and content != MASKED_PLACEHOLDER:
                messages[i] = {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": MASKED_PLACEHOLDER,
                }
                masked_count += 1

        if masked_count > 0:
            log.info(
                "观察遮蔽已应用",
                masked_count=masked_count,
                current_turn=current_turn,
            )

        return masked_count
