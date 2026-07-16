"""S10 验证引擎验证测试。"""

import textwrap
from unittest.mock import MagicMock, patch

import pytest

from praxis.models.gateway import JudgeResult
from praxis.models.verification import (
    FailureDetail,
    QualityPhase,
    VerificationResult,
    VerificationStatus,
    VerificationType,
)
from praxis.verification.computational import (
    LintVerifier,
    SchemaVerifier,
    SuiteTestVerifier,
    TypeCheckVerifier,
    run_computational,
)
from praxis.verification.gav import (
    ControlQuadrant,
    GAVController,
    GAVVerifyRequest,
    GAVVerifyResponse,
)
from praxis.verification.inferential import InferentialVerifier, run_inferential
from praxis.verification.registry import VerifierRegistry

# ── Task 9.1: 计算型验证 ───────────────────────────────────────────────────


class TestLintVerifier:
    """Lint 验证器测试。"""

    async def test_valid_code_passes(self) -> None:
        v = LintVerifier()
        result = await v.verify({"code": "x = 1\nprint(x)\n"})
        assert result.status == VerificationStatus.PASS
        assert len(result.failures) == 0

    async def test_syntax_error_fails(self) -> None:
        v = LintVerifier()
        result = await v.verify({"code": "def foo(\n"})
        assert result.status == VerificationStatus.FAIL
        assert len(result.failures) == 1
        assert result.failures[0].rule == "syntax_error"
        assert result.failures[0].line is not None

    async def test_missing_params(self) -> None:
        v = LintVerifier()
        result = await v.verify({})
        assert result.status == VerificationStatus.ERROR

    async def test_file_not_found(self) -> None:
        v = LintVerifier()
        result = await v.verify({"file": "/nonexistent/path.py"})
        assert result.status == VerificationStatus.FAIL
        assert result.failures[0].rule == "file_not_found"

    async def test_valid_file(self, tmp_path) -> None:
        f = tmp_path / "good.py"
        f.write_text("x = 1\n", encoding="utf-8")
        v = LintVerifier()
        result = await v.verify({"file": str(f)})
        assert result.status == VerificationStatus.PASS

    async def test_invalid_file(self, tmp_path) -> None:
        f = tmp_path / "bad.py"
        f.write_text("def foo(\n", encoding="utf-8")
        v = LintVerifier()
        result = await v.verify({"file": str(f)})
        assert result.status == VerificationStatus.FAIL
        assert result.failures[0].line is not None

    def test_parse_lint_output(self) -> None:
        output = textwrap.dedent("""\
            app.py:10:5: E302 expected 2 blank lines
            app.py:20:1: W291 trailing whitespace
        """)
        failures = LintVerifier.parse_lint_output(output)
        assert len(failures) == 2
        assert failures[0].file == "app.py"
        assert failures[0].line == 10
        assert failures[1].line == 20


class TestSchemaVerifier:
    """Schema 验证器测试。"""

    async def test_valid_data(self) -> None:
        v = SchemaVerifier()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        result = await v.verify({"data": {"name": "Alice", "age": 30}, "schema": schema})
        assert result.status == VerificationStatus.PASS

    async def test_invalid_data(self) -> None:
        v = SchemaVerifier()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        result = await v.verify({"data": {"age": 30}, "schema": schema})
        assert result.status == VerificationStatus.FAIL
        assert len(result.failures) == 1

    async def test_missing_schema(self) -> None:
        v = SchemaVerifier()
        result = await v.verify({"data": {}})
        assert result.status == VerificationStatus.ERROR


class TestTypeCheckVerifier:
    """类型检查验证器测试。"""

    async def test_valid_code(self) -> None:
        v = TypeCheckVerifier()
        result = await v.verify({"code": "x: int = 1\n"})
        assert result.status == VerificationStatus.PASS

    async def test_syntax_error(self) -> None:
        v = TypeCheckVerifier()
        result = await v.verify({"code": "def f(:\n"})
        assert result.status == VerificationStatus.FAIL


class TestSuiteTestVerifier:
    """测试套件验证器测试。"""

    async def test_verify_returns_metadata(self) -> None:
        v = SuiteTestVerifier()
        result = await v.verify({"path": "tests/"})
        assert result.metadata.get("command") is not None

    def test_parse_pytest_output_pass(self) -> None:
        result = SuiteTestVerifier.parse_pytest_output("5 passed\n", 0)
        assert result.status == VerificationStatus.PASS

    def test_parse_pytest_output_fail(self) -> None:
        output = "FAILED tests/test_foo.py::test_bar\n1 failed\n"
        result = SuiteTestVerifier.parse_pytest_output(output, 1)
        assert result.status == VerificationStatus.FAIL
        assert len(result.failures) == 1
        assert "test_bar" in result.failures[0].message


class TestRunComputational:
    """批量计算型验证测试。"""

    async def test_batch_run(self) -> None:
        results = await run_computational(
            [LintVerifier(), SchemaVerifier()],
            {
                "code": "x = 1\n",
                "data": {"name": "test"},
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        )
        assert len(results) == 2
        assert all(r.status == VerificationStatus.PASS for r in results)


# ── Task 9.2: 推理型验证 ───────────────────────────────────────────────────


class TestInferentialVerifier:
    """推理型验证测试。"""

    async def test_pass_verdict(self) -> None:
        gw = MagicMock()
        gw.config = MagicMock()
        gw.config.max_budget = None
        gw.config.default_model = "test-model"

        mock_judge = JudgeResult(
            verdict=True, confidence=0.9,
            reasoning="代码正确", raw_response="{}",
        )
        with patch("praxis.verification.inferential.judge", return_value=mock_judge):
            result = await run_inferential(
                gw, criteria="代码是否正确", content="def add(a,b): return a+b",
            )
            assert result.status == VerificationStatus.PASS
            assert result.score == 0.9

    async def test_fail_verdict(self) -> None:
        gw = MagicMock()
        gw.config = MagicMock()
        gw.config.max_budget = None
        gw.config.default_model = "test-model"

        mock_judge = JudgeResult(
            verdict=False, confidence=0.3,
            reasoning="逻辑错误", raw_response="{}",
        )
        with patch("praxis.verification.inferential.judge", return_value=mock_judge):
            result = await run_inferential(
                gw, criteria="代码是否正确", content="def add(a,b): return a-b",
            )
            assert result.status == VerificationStatus.FAIL
            assert "逻辑错误" in result.feedback

    async def test_error_handling(self) -> None:
        gw = MagicMock()
        gw.config = MagicMock()
        gw.config.max_budget = None
        gw.config.default_model = "test-model"

        with patch("praxis.verification.inferential.judge", side_effect=RuntimeError("LLM down")):
            result = await run_inferential(
                gw, criteria="test", content="test",
            )
            assert result.status == VerificationStatus.ERROR

    async def test_with_dimensions(self) -> None:
        gw = MagicMock()
        gw.config = MagicMock()
        gw.config.max_budget = None
        gw.config.default_model = "test-model"

        mock_judge = JudgeResult(
            verdict=True, confidence=0.85,
            reasoning="各维度均通过", raw_response="{}",
        )
        with patch("praxis.verification.inferential.judge", return_value=mock_judge):
            v = InferentialVerifier(gw)
            result = await v.verify(
                "代码质量评估", "some code",
                dimensions=["正确性", "完整性", "代码风格"],
            )
            assert result.passed


# ── Task 9.4: GAV 循环支持 ─────────────────────────────────────────────────


class TestGAVController:
    """GAV 循环测试。"""

    def test_all_pass(self) -> None:
        ctrl = GAVController()
        results = [
            VerificationResult(
                status=VerificationStatus.PASS,
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name="lint",
            ),
        ]
        request = GAVVerifyRequest(results=results)
        response = ctrl.evaluate(request)
        assert response.passed is True
        assert response.context_injection == ""

    def test_failure_returns_context(self) -> None:
        ctrl = GAVController()
        results = [
            VerificationResult(
                status=VerificationStatus.FAIL,
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name="lint",
                feedback="语法错误",
                failures=[
                    FailureDetail(
                        file="app.py", line=10,
                        message="unexpected EOF", rule="syntax_error",
                    ),
                ],
            ),
        ]
        request = GAVVerifyRequest(results=results)
        response = ctrl.evaluate(request)
        assert response.passed is False
        assert "app.py:10" in response.context_injection
        assert "unexpected EOF" in response.context_injection

    def test_select_quadrant(self) -> None:
        assert GAVController.select_quadrant(
            True, VerificationType.COMPUTATIONAL,
        ) == ControlQuadrant.FEEDFORWARD_COMPUTATIONAL
        assert GAVController.select_quadrant(
            False, VerificationType.INFERENTIAL,
        ) == ControlQuadrant.FEEDBACK_INFERENTIAL

    def test_format_for_context_pass(self) -> None:
        response = GAVVerifyResponse(passed=True, results=[])
        text = GAVController.format_for_context(response)
        assert "验证通过" in text

    def test_format_for_context_fail(self) -> None:
        response = GAVVerifyResponse(
            passed=False,
            results=[],
            context_injection="lint: syntax error",
            retry_hint="修复语法",
        )
        text = GAVController.format_for_context(response)
        assert "验证失败" in text
        assert "lint" in text


# ── Task 9.5: 验证器注册与质量左移 ─────────────────────────────────────────


class TestVerifierRegistry:
    """验证器注册与质量左移测试。"""

    def test_register_and_get(self) -> None:
        reg = VerifierRegistry()
        reg.register(LintVerifier())
        entry = reg.get("lint_ruff")
        assert entry is not None
        assert entry.verification_type == VerificationType.COMPUTATIONAL

    def test_unregister(self) -> None:
        reg = VerifierRegistry()
        reg.register(LintVerifier())
        reg.unregister("lint_ruff")
        assert reg.get("lint_ruff") is None

    def test_list_by_type(self) -> None:
        reg = VerifierRegistry()
        reg.register(LintVerifier())
        reg.register(SchemaVerifier())
        entries = reg.list_verifiers(verification_type=VerificationType.COMPUTATIONAL)
        assert len(entries) == 2

    def test_list_by_phase(self) -> None:
        reg = VerifierRegistry()
        reg.register(
            LintVerifier(),
            phases=[QualityPhase.PRE_INTEGRATION],
        )
        reg.register(
            SchemaVerifier(),
            phases=[QualityPhase.POST_INTEGRATION],
        )

        pre = reg.list_verifiers(phase=QualityPhase.PRE_INTEGRATION)
        assert len(pre) == 1
        assert pre[0].verifier.name == "lint_ruff"

    async def test_run_computational(self) -> None:
        reg = VerifierRegistry()
        reg.register(LintVerifier())
        reg.register(SchemaVerifier())

        results = await reg.run_computational({
            "code": "x = 1\n",
            "data": {},
            "schema": {"type": "object"},
        })
        assert len(results) == 2
        assert all(r.status == VerificationStatus.PASS for r in results)

    async def test_run_computational_with_phase_filter(self) -> None:
        reg = VerifierRegistry()
        reg.register(
            LintVerifier(),
            phases=[QualityPhase.PRE_INTEGRATION],
        )
        reg.register(
            SchemaVerifier(),
            phases=[QualityPhase.POST_INTEGRATION],
        )

        results = await reg.run_computational(
            {"code": "x = 1\n"},
            phase=QualityPhase.PRE_INTEGRATION,
        )
        assert len(results) == 1

    def test_get_phase_config(self) -> None:
        reg = VerifierRegistry()
        reg.register(LintVerifier(), phases=[QualityPhase.PRE_INTEGRATION])
        reg.register(TypeCheckVerifier(), phases=[QualityPhase.PRE_INTEGRATION])
        names = reg.get_phase_config(QualityPhase.PRE_INTEGRATION)
        assert set(names) == {"lint_ruff", "typecheck_mypy"}

    async def test_run_inferential_via_registry(self) -> None:
        """注册表持有网关时，可驱动推理型验证。"""
        gw = MagicMock()
        gw.config = MagicMock()
        gw.config.max_budget = None
        gw.config.default_model = "test-model"
        reg = VerifierRegistry(gateway=gw)

        mock_judge = JudgeResult(
            verdict=True, confidence=0.88, reasoning="符合标准", raw_response="{}",
        )
        with patch("praxis.verification.inferential.judge", return_value=mock_judge):
            result = await reg.run_inferential(criteria="是否正确", content="def f(): pass")
            assert result.status == VerificationStatus.PASS
            assert result.score == 0.88

    async def test_run_inferential_without_gateway_skips(self) -> None:
        """未配置网关时推理型验证返回 SKIP，而非抛错或静默丢失。"""
        reg = VerifierRegistry()
        result = await reg.run_inferential(criteria="x", content="y")
        assert result.status == VerificationStatus.SKIP

    async def test_run_visual_without_gateway_skips(self) -> None:
        reg = VerifierRegistry()
        result = await reg.run_visual(url="http://x", expectations="y")
        assert result.status == VerificationStatus.SKIP

    async def test_run_visual_skips_when_default_model_lacks_vision(self) -> None:
        gateway = MagicMock()
        gateway.supports_vision.return_value = False
        reg = VerifierRegistry(gateway=gateway, visual_enabled=True)

        result = await reg.run_visual(url="https://example.com", expectations="ok")

        assert result.status is VerificationStatus.SKIP
        assert "视觉能力" in result.feedback
        gateway.supports_vision.assert_called_once_with()

    async def test_run_visual_reports_missing_optional_dependency(self) -> None:
        gateway = MagicMock()
        gateway.supports_vision.return_value = True
        reg = VerifierRegistry(gateway=gateway, visual_enabled=True)

        with (
            patch("praxis.verification.registry.find_spec", return_value=None),
            pytest.raises(RuntimeError, match=r"praxis\[visual\]"),
        ):
            await reg.run_visual(url="https://example.com", expectations="ok")

    async def test_custom_verifier(self) -> None:
        class CustomVerifier:
            @property
            def name(self) -> str:
                return "custom_check"

            async def verify(self, target):
                return VerificationResult(
                    status=VerificationStatus.PASS,
                    verification_type=VerificationType.COMPUTATIONAL,
                    verifier_name=self.name,
                    feedback="自定义检查通过",
                )

        reg = VerifierRegistry()
        reg.register(CustomVerifier())
        results = await reg.run_computational({"code": "test"})
        assert len(results) == 1
        assert results[0].feedback == "自定义检查通过"


class TestVerifierConfigFlags:
    """S10 配置启停开关。"""

    async def test_disabled_types_skip(self) -> None:
        from praxis.config.schemas import VerificationConfig
        from praxis.verification.computational import LintVerifier

        reg = VerifierRegistry.from_config(
            VerificationConfig(
                computational_enabled=False,
                inferential_enabled=False,
                visual_enabled=False,
            ),
        )
        reg.register(LintVerifier())
        assert await reg.run_computational({"code": "x=1\n"}) == []
        r = await reg.run_inferential(criteria="x", content="y")
        assert r.status == VerificationStatus.SKIP
        r2 = await reg.run_visual(url="http://x", expectations="y")
        assert r2.status == VerificationStatus.SKIP
