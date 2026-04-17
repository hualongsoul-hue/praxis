"""视觉验证（Visual Verification）。

通过 Playwright 截取页面截图，提交 S4 多模态能力比对，
返回截图路径 + 视觉判定 + 差异描述。
"""

import base64
import json
import time
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

from praxis.gateway.chat import chat
from praxis.gateway.router import GatewayRouter
from praxis.models.verification import (
    FailureDetail,
    VerificationResult,
    VerificationStatus,
    VerificationType,
)
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("verification.visual")

VISUAL_SYSTEM_PROMPT = (
    "你是一个视觉验证专家。分析提供的页面截图，判断是否满足预期要求。\n"
    "以 JSON 格式输出：\n"
    '- "pass": true 或 false\n'
    '- "confidence": 0.0~1.0 置信度\n'
    '- "description": 描述页面内容\n'
    '- "differences": 与预期不符之处（数组）\n'
    "仅输出 JSON，不添加其他文字。"
)


class VisualVerifier:
    """视觉验证器。

    Playwright 截图 + S4 多模态 LLM 评估。
    """

    def __init__(
        self,
        gateway: GatewayRouter,
        model: str | None = None,
        screenshot_dir: str = "data/screenshots",
    ) -> None:
        self.gateway = gateway
        self.model = model
        self.screenshot_dir = Path(screenshot_dir)

    async def capture_screenshot(self, url: str, output_path: Path) -> Path:
        """通过 Playwright 截取页面截图。

        Args:
            url: 页面 URL。
            output_path: 截图保存路径。

        Returns:
            截图文件路径。
        """
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle")
            await page.screenshot(path=str(output_path), full_page=True)
            await browser.close()

        return output_path

    async def verify(
        self,
        url: str,
        expectations: str,
        screenshot_name: str | None = None,
    ) -> VerificationResult:
        """执行视觉验证。

        Args:
            url: 页面 URL。
            expectations: 预期描述。
            screenshot_name: 可选截图文件名。

        Returns:
            视觉验证结果。
        """
        start = time.perf_counter()
        name = screenshot_name or f"visual_{int(time.time())}.png"
        output_path = self.screenshot_dir / name

        try:
            screenshot_path = await self.capture_screenshot(url, output_path)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            log.error("截图失败", url=url, error=str(exc))
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verification_type=VerificationType.VISUAL,
                verifier_name="visual",
                feedback=f"页面截图失败: {exc}",
                duration_ms=elapsed,
            )

        try:
            result = await self.evaluate_screenshot(screenshot_path, expectations)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            log.error("视觉评估失败", error=str(exc))
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verification_type=VerificationType.VISUAL,
                verifier_name="visual",
                feedback=f"视觉评估失败: {exc}",
                metadata={"screenshot": str(screenshot_path)},
                duration_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        result.duration_ms = elapsed
        result.metadata["screenshot"] = str(screenshot_path)

        emit_metric(
            "verification_visual",
            1.0,
            {"status": result.status.value},
            "counter",
        )
        return result

    async def evaluate_screenshot(
        self,
        screenshot_path: Path,
        expectations: str,
    ) -> VerificationResult:
        """通过 S4 多模态 LLM 评估截图。

        Args:
            screenshot_path: 截图文件路径。
            expectations: 预期描述。

        Returns:
            验证结果。
        """
        image_bytes = screenshot_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": VISUAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"预期要求：{expectations}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                ],
            },
        ]

        response = await chat(self.gateway, messages, model=self.model)
        raw_text = response.content or ""

        try:
            data = json.loads(raw_text)
            passed = data.get("pass", False)
            confidence = data.get("confidence", 0.0)
            description = data.get("description", "")
            differences = data.get("differences", [])

            failures: list[FailureDetail] = []
            for diff in differences:
                failures.append(FailureDetail(
                    message=str(diff),
                    severity="warning",
                    rule="visual_difference",
                ))

            status = VerificationStatus.PASS if passed else VerificationStatus.FAIL
            return VerificationResult(
                status=status,
                verification_type=VerificationType.VISUAL,
                verifier_name="visual",
                score=confidence,
                failures=failures,
                feedback=description,
                metadata={"raw_response": raw_text},
            )
        except (json.JSONDecodeError, ValueError):
            log.warning("视觉评估响应解析失败", raw_text=raw_text[:200])
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verification_type=VerificationType.VISUAL,
                verifier_name="visual",
                feedback=f"无法解析 LLM 响应: {raw_text[:200]}",
                metadata={"raw_response": raw_text},
            )


async def run_visual(
    gateway: GatewayRouter,
    url: str,
    expectations: str,
    model: str | None = None,
) -> VerificationResult:
    """便捷函数：执行视觉验证。

    Args:
        gateway: S4 网关路由器。
        url: 页面 URL。
        expectations: 预期描述。
        model: 模型别名。

    Returns:
        验证结果。
    """
    verifier = VisualVerifier(gateway, model=model)
    return await verifier.verify(url, expectations)
