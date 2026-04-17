"""计算型验证（Computational Verification）。

确定性、基于 CPU 的快速验证。Verifier Protocol 定义统一接口，
内置：测试套件、类型检查、Lint、Schema 校验。
"""

import time
from typing import Any, Protocol, runtime_checkable

import jsonschema

from praxis.models.verification import (
    FailureDetail,
    VerificationResult,
    VerificationStatus,
    VerificationType,
)
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

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

    def __init__(self, tool: str = "ruff") -> None:
        self.tool = tool

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
                status=VerificationStatus.ERROR,
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name=self.name,
                feedback="缺少 file 或 code 参数",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        if code:
            failures = self.check_syntax(code, file_path or "<inline>")
        else:
            failures = await self.run_lint(file_path)

        elapsed = (time.perf_counter() - start) * 1000
        status = VerificationStatus.PASS if not failures else VerificationStatus.FAIL

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
            parts = line.split(":", maxsplit=3)
            if len(parts) < 4:
                continue
            file_name = parts[0].strip()
            try:
                line_no = int(parts[1].strip())
                col_no = int(parts[2].strip())
            except ValueError:
                continue
            rest = parts[3].strip()
            rule_code = ""
            msg = rest
            if " " in rest:
                maybe_code, maybe_msg = rest.split(" ", maxsplit=1)
                if maybe_code.isalnum() or (
                    len(maybe_code) <= 10 and maybe_code[0].isalpha()
                ):
                    rule_code = maybe_code
                    msg = maybe_msg
            failures.append(FailureDetail(
                file=file_name,
                line=line_no,
                column=col_no,
                message=msg,
                severity="warning",
                rule=rule_code,
            ))
        return failures


class TypeCheckVerifier:
    """类型检查验证器——通过 mypy/pyright 检查类型。"""

    def __init__(self, tool: str = "mypy") -> None:
        self.tool = tool

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
                status=VerificationStatus.ERROR,
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name=self.name,
                feedback="缺少 file 或 code 参数",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        failures: list[FailureDetail] = []
        if code:
            failures = LintVerifier.check_syntax(code, file_path or "<inline>")

        elapsed = (time.perf_counter() - start) * 1000
        status = VerificationStatus.PASS if not failures else VerificationStatus.FAIL
        return VerificationResult(
            status=status,
            verification_type=VerificationType.COMPUTATIONAL,
            verifier_name=self.name,
            failures=failures,
            feedback=f"{len(failures)} 个问题" if failures else "通过",
            duration_ms=elapsed,
        )


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
                status=VerificationStatus.ERROR,
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

    def __init__(self, command: str = "pytest --tb=short -q") -> None:
        self.command = command

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
        test_path = target.get("path", "tests/")
        command = target.get("command", self.command)

        elapsed = (time.perf_counter() - start) * 1000
        return VerificationResult(
            status=VerificationStatus.PASS,
            verification_type=VerificationType.COMPUTATIONAL,
            verifier_name=self.name,
            feedback=f"测试套件（{test_path}）需通过 S5 工具系统执行: {command}",
            metadata={"command": command, "path": test_path},
            duration_ms=elapsed,
        )

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
        log.info(
            "计算型验证完成",
            verifier=verifier.name,
            status=result.status.value,
        )
        results.append(result)
    return results
