"""计算型验证（Computational Verification）。

确定性、基于 CPU 的快速验证。Verifier Protocol 定义统一接口，
内置：测试套件、类型检查、Lint、Schema 校验。
"""

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import jsonschema

from praxis.exceptions import ToolPolicyViolationError, ToolTimeoutError
from praxis.models.verification import (
    FailureDetail,
    VerificationResult,
    VerificationStatus,
    VerificationType,
)
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric
from praxis.tools.policy import ToolPolicy
from praxis.tools.process import ProcessResult, ProcessRunner

log = get_logger("verification.computational")


@runtime_checkable
class Verifier(Protocol):
    """验证器协议。所有计算型验证器实现此接口。"""

    @property
    def name(self) -> str: ...

    async def verify(self, target: dict[str, Any]) -> VerificationResult: ...


class LintVerifier:
    """Lint 验证器——检查 Python 代码的语法和风格问题。

    通过 S5 工具系统的 run_command 执行外部 Lint 工具（ruff/flake8），
    解析结构化输出。
    """

    def __init__(
        self,
        tool: str = "ruff",
        *,
        runner: ProcessRunner | None = None,
        policy: ToolPolicy | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.tool = tool
        self.runner = runner or ProcessRunner()
        self.policy = policy
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"lint_{self.tool}"

    async def verify(self, target: dict[str, Any]) -> VerificationResult:
        """执行 Lint 验证。

        Args:
            target: {"file": "path/to/file.py"} 或
                    {"code": "source code", "file": "virtual.py"}

        Returns:
            包含行号级别失败详情的验证结果。
        """
        start = time.perf_counter()
        file_path = target.get("file", "")
        code = target.get("code", "")

        if not file_path and not code:
            return VerificationResult(
                status=(
                    VerificationStatus.SKIP
                    if set(target) == {"tool_outcomes"}
                    else VerificationStatus.ERROR
                ),
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name=self.name,
                feedback="缺少 file 或 code 参数",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        temporary: tempfile.TemporaryDirectory[str] | None = None
        try:
            if code:
                syntax_failures = self.check_syntax(str(code), file_path or "<inline>")
                if syntax_failures:
                    return VerificationResult(
                        status=VerificationStatus.FAIL,
                        verification_type=VerificationType.COMPUTATIONAL,
                        verifier_name=self.name,
                        failures=syntax_failures,
                        feedback=f"{len(syntax_failures)} 个问题",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
                temporary = tempfile.TemporaryDirectory(prefix="praxis-lint-")
                lint_path = Path(temporary.name) / (Path(file_path).name or "inline.py")
                lint_path.write_text(str(code), encoding="utf-8")
            else:
                lint_path = self.resolve_path(str(file_path))
            process = await self.runner.run_exec(
                [self.tool, "check", "--output-format", "concise", str(lint_path)],
                cwd=lint_path.parent,
                timeout=self.timeout,
                environment=dict(os.environ),
            )
            failures = self.parse_lint_output(process.stdout or process.stderr)
            status = self.process_status(process)
        except (OSError, ValueError, ToolPolicyViolationError, ToolTimeoutError) as exc:
            return self.error_result(str(exc), start)
        finally:
            if temporary is not None:
                temporary.cleanup()

        elapsed = (time.perf_counter() - start) * 1000

        emit_metric(
            "verification_computational",
            1.0,
            {"verifier": self.name, "status": status.value},
            "counter",
        )
        return VerificationResult(
            status=status,
            verification_type=VerificationType.COMPUTATIONAL,
            verifier_name=self.name,
            failures=failures,
            feedback=f"{len(failures)} 个问题" if failures else "通过",
            duration_ms=elapsed,
        )

    def resolve_path(self, file_path: str) -> Path:
        path = Path(file_path).expanduser().resolve()
        if self.policy is not None:
            return self.policy.check_path(path)
        return path

    @staticmethod
    def process_status(process: ProcessResult) -> VerificationStatus:
        if process.returncode == 0:
            return VerificationStatus.PASS
        if process.returncode == 1:
            return VerificationStatus.FAIL
        return VerificationStatus.ERROR

    def error_result(self, feedback: str, start: float) -> VerificationResult:
        return VerificationResult(
            status=VerificationStatus.ERROR,
            verification_type=VerificationType.COMPUTATIONAL,
            verifier_name=self.name,
            feedback=feedback,
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    @staticmethod
    def check_syntax(code: str, filename: str) -> list[FailureDetail]:
        """使用 compile() 检查 Python 语法。"""
        try:
            compile(code, filename, "exec")
            return []
        except SyntaxError as exc:
            return [
                FailureDetail(
                    file=filename,
                    line=exc.lineno,
                    column=exc.offset,
                    message=exc.msg,
                    severity="error",
                    rule="syntax_error",
                )
            ]

    async def run_lint(self, file_path: str) -> list[FailureDetail]:
        """通过外部 Lint 工具验证文件。

        需要 S5 工具系统支持，此处提供解析框架。
        实际执行由 S11 编排循环通过 S5 run_command 完成，
        结果通过 parse_lint_output 解析。
        """
        return self.check_syntax_from_file(file_path)

    @staticmethod
    def check_syntax_from_file(file_path: str) -> list[FailureDetail]:
        """从文件路径检查语法。"""
        try:
            with open(file_path, encoding="utf-8") as f:
                code = f.read()
            compile(code, file_path, "exec")
            return []
        except SyntaxError as exc:
            return [
                FailureDetail(
                    file=file_path,
                    line=exc.lineno,
                    column=exc.offset,
                    message=exc.msg,
                    severity="error",
                    rule="syntax_error",
                )
            ]
        except FileNotFoundError:
            return [
                FailureDetail(
                    file=file_path,
                    message=f"文件未找到: {file_path}",
                    severity="error",
                    rule="file_not_found",
                )
            ]

    @staticmethod
    def parse_lint_output(output: str) -> list[FailureDetail]:
        """解析 ruff/flake8 格式输出。

        格式：file.py:line:col: CODE message
        """
        failures: list[FailureDetail] = []
        for line in output.strip().splitlines():
            match = re.match(r"^(.*?):(\d+):(\d+):\s+([^\s]+)\s+(.*)$", line)
            if match is None:
                continue
            failures.append(FailureDetail(
                file=match.group(1).strip(),
                line=int(match.group(2)),
                column=int(match.group(3)),
                message=match.group(5).strip(),
                severity="warning",
                rule=match.group(4).strip(),
            ))
        return failures


class TypeCheckVerifier:
    """类型检查验证器——通过 mypy/pyright 检查类型。"""

    def __init__(
        self,
        tool: str = "pyright",
        *,
        runner: ProcessRunner | None = None,
        policy: ToolPolicy | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.tool = tool
        self.runner = runner or ProcessRunner()
        self.policy = policy
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"typecheck_{self.tool}"

    async def verify(self, target: dict[str, Any]) -> VerificationResult:
        """执行类型检查。

        Args:
            target: {"file": "path/to/file.py"} 或 {"code": "..."}

        Returns:
            验证结果。
        """
        start = time.perf_counter()
        file_path = target.get("file", "")
        code = target.get("code", "")

        if not file_path and not code:
            return VerificationResult(
                status=(
                    VerificationStatus.SKIP
                    if set(target) == {"tool_outcomes"}
                    else VerificationStatus.ERROR
                ),
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name=self.name,
                feedback="缺少 file 或 code 参数",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        temporary: tempfile.TemporaryDirectory[str] | None = None
        try:
            if code:
                temporary = tempfile.TemporaryDirectory(prefix="praxis-types-")
                checked_path = Path(temporary.name) / (Path(file_path).name or "inline.py")
                checked_path.write_text(str(code), encoding="utf-8")
            else:
                checked_path = Path(str(file_path)).expanduser().resolve()
                if self.policy is not None:
                    checked_path = self.policy.check_path(checked_path)
            command = [self.tool, "--outputjson", str(checked_path)]
            process = await self.runner.run_exec(
                command,
                cwd=checked_path.parent,
                timeout=self.timeout,
                environment=dict(os.environ),
            )
            failures = self.parse_output(process.stdout or process.stderr)
            if process.returncode == 0:
                status = VerificationStatus.PASS
            elif process.returncode == 1:
                status = VerificationStatus.FAIL
            else:
                status = VerificationStatus.ERROR
        except (OSError, ValueError, ToolPolicyViolationError, ToolTimeoutError) as exc:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name=self.name,
                feedback=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        finally:
            if temporary is not None:
                temporary.cleanup()

        elapsed = (time.perf_counter() - start) * 1000
        return VerificationResult(
            status=status,
            verification_type=VerificationType.COMPUTATIONAL,
            verifier_name=self.name,
            failures=failures,
            feedback=f"{len(failures)} 个问题" if failures else "通过",
            duration_ms=elapsed,
        )

    @staticmethod
    def parse_output(output: str) -> list[FailureDetail]:
        try:
            decoded: object = json.loads(output)
        except json.JSONDecodeError:
            return [FailureDetail(message=output[:4000], rule="typecheck_error")] if output else []
        if not isinstance(decoded, dict):
            return []
        payload = cast(dict[str, object], decoded)
        diagnostics_value = payload.get("generalDiagnostics", [])
        if not isinstance(diagnostics_value, list):
            return []
        diagnostics = cast(list[object], diagnostics_value)
        failures: list[FailureDetail] = []
        for diagnostic_value in diagnostics:
            if not isinstance(diagnostic_value, dict):
                continue
            diagnostic = cast(dict[str, object], diagnostic_value)
            range_value = diagnostic.get("range", {})
            range_mapping = (
                cast(dict[str, object], range_value) if isinstance(range_value, dict) else {}
            )
            start_value = range_mapping.get("start", {})
            start_mapping = (
                cast(dict[str, object], start_value) if isinstance(start_value, dict) else {}
            )
            line = start_mapping.get("line")
            character = start_mapping.get("character")
            failures.append(FailureDetail(
                file=str(diagnostic.get("file", "")),
                line=int(line) + 1 if isinstance(line, int) else None,
                column=int(character) + 1 if isinstance(character, int) else None,
                message=str(diagnostic.get("message", "类型检查失败")),
                severity=str(diagnostic.get("severity", "error")),
                rule=str(diagnostic.get("rule", "type_error")),
            ))
        return failures


class SchemaVerifier:
    """Schema 校验器——校验结构化输出的 JSON Schema 合规性。"""

    @property
    def name(self) -> str:
        return "schema"

    async def verify(self, target: dict[str, Any]) -> VerificationResult:
        """校验数据是否符合 JSON Schema。

        Args:
            target: {"data": ..., "schema": {...}}

        Returns:
            验证结果。
        """
        start = time.perf_counter()
        data = target.get("data")
        schema = target.get("schema")

        if schema is None:
            return VerificationResult(
                status=(
                    VerificationStatus.SKIP
                    if set(target) == {"tool_outcomes"}
                    else VerificationStatus.ERROR
                ),
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name=self.name,
                feedback="缺少 schema 参数",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        failures: list[FailureDetail] = []
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as exc:
            path_str = ".".join(str(p) for p in exc.absolute_path)
            failures.append(FailureDetail(
                message=exc.message,
                severity="error",
                rule=f"schema_violation:{path_str}" if path_str else "schema_violation",
            ))

        elapsed = (time.perf_counter() - start) * 1000
        status = VerificationStatus.PASS if not failures else VerificationStatus.FAIL
        return VerificationResult(
            status=status,
            verification_type=VerificationType.COMPUTATIONAL,
            verifier_name=self.name,
            failures=failures,
            feedback=f"{len(failures)} 个问题" if failures else "Schema 校验通过",
            duration_ms=elapsed,
        )


class SuiteTestVerifier:
    """测试套件验证器——通过 S5 执行测试命令，解析结果。"""

    def __init__(
        self,
        command: tuple[str, ...] = ("pytest", "--tb=short", "-q"),
        *,
        runner: ProcessRunner | None = None,
        policy: ToolPolicy | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.command = command
        self.runner = runner or ProcessRunner()
        self.policy = policy
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "test_suite"

    async def verify(self, target: dict[str, Any]) -> VerificationResult:
        """执行测试套件。

        Args:
            target: {"path": "tests/", "command": "override command"}

        Returns:
            验证结果。
        """
        start = time.perf_counter()
        if target and "path" not in target and "command" not in target:
            return VerificationResult(
                status=VerificationStatus.SKIP,
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name=self.name,
                feedback="目标不包含测试套件请求",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        test_path = str(target.get("path", "."))
        command_value: object = target.get("command", self.command)
        command = self.parse_command(command_value)
        if command is None:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name=self.name,
                feedback="测试命令必须是字符串参数列表",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        try:
            path = Path(test_path).expanduser().resolve()
            if self.policy is not None:
                path = self.policy.check_path(path)
            process = await self.runner.run_exec(
                command,
                cwd=path,
                timeout=self.timeout,
                environment=dict(os.environ),
            )
        except (OSError, ValueError, ToolPolicyViolationError, ToolTimeoutError) as exc:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name=self.name,
                feedback=str(exc),
                metadata={"path": test_path},
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        parsed = self.parse_pytest_output(
            "\n".join(item for item in (process.stdout, process.stderr) if item),
            process.returncode,
        )
        parsed.metadata = {
            "command": command,
            "path": str(path),
            "exit_code": process.returncode,
        }
        parsed.duration_ms = (time.perf_counter() - start) * 1000
        return parsed

    @staticmethod
    def parse_command(value: object) -> list[str] | None:
        if not isinstance(value, (list, tuple)):
            return None
        items = cast(list[object] | tuple[object, ...], value)
        if not items or not all(isinstance(item, str) and item for item in items):
            return None
        return [cast(str, item) for item in items]

    @staticmethod
    def parse_pytest_output(output: str, exit_code: int) -> VerificationResult:
        """解析 pytest 输出。

        Args:
            output: pytest 标准输出。
            exit_code: 退出码（0=全部通过）。

        Returns:
            结构化验证结果。
        """
        failures: list[FailureDetail] = []
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("FAILED"):
                parts = stripped.split("::", maxsplit=1)
                file_name = parts[0].replace("FAILED ", "").strip()
                test_name = parts[1].strip() if len(parts) > 1 else ""
                failures.append(FailureDetail(
                    file=file_name,
                    message=f"测试失败: {test_name}",
                    severity="error",
                    rule="test_failure",
                ))

        status = VerificationStatus.PASS if exit_code == 0 else VerificationStatus.FAIL
        return VerificationResult(
            status=status,
            verification_type=VerificationType.COMPUTATIONAL,
            verifier_name="test_suite",
            failures=failures,
            feedback=f"退出码: {exit_code}, {len(failures)} 个失败",
        )


async def run_computational(
    verifiers: list[Verifier],
    target: dict[str, Any],
) -> list[VerificationResult]:
    """批量执行计算型验证。

    Args:
        verifiers: 验证器列表。
        target: 验证目标。

    Returns:
        每个验证器的结果列表。
    """
    results: list[VerificationResult] = []
    for verifier in verifiers:
        result = await verifier.verify(target)
        invalid_result = (
            result.verifier_name != verifier.name
            or result.verification_type is not VerificationType.COMPUTATIONAL
            or (
                result.status is VerificationStatus.PASS
                and not result.feedback.strip()
            )
        )
        if invalid_result:
            result = VerificationResult(
                status=VerificationStatus.ERROR,
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name=verifier.name,
                feedback="验证器返回了无效或缺少证据的结果",
            )
        log.info(
            "计算型验证完成",
            verifier=verifier.name,
            status=result.status.value,
        )
        results.append(result)
    return results
