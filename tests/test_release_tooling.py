"""Cross-platform checks for release and wheel-verification tooling."""

import io

from scripts.verify_wheel import write_console_output


def test_wheel_verifier_renders_unicode_on_legacy_windows_console() -> None:
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252")

    write_console_output("配置校验通过\n", stream=stream)
    stream.flush()

    rendered = buffer.getvalue().decode("cp1252")
    assert rendered.endswith("\n")
    assert "\\u" in rendered
