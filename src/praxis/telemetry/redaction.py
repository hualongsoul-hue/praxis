"""Shared immutable redaction policy for events, logs, audit records, and traces."""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    """Recursively redact secrets and bound untrusted observability values."""

    max_depth: int = 8
    max_collection_items: int = 100
    max_string_length: int = 16_384
    redacted_value: str = "[REDACTED]"
    truncated_suffix: str = "…[TRUNCATED]"

    def is_sensitive_key(self, key: str) -> bool:
        normalized = key.casefold().replace("-", "_")
        return any(
            marker in normalized
            for marker in (
                "api_key",
                "apikey",
                "authorization",
                "cookie",
                "credential",
                "password",
                "private_key",
                "secret",
                "token",
            )
        )

    def redact_text(
        self, value: str, *, max_length: int | None = None, truncate: bool = True,
    ) -> str:
        """Redact credentials; business bodies can explicitly retain their full length."""
        redacted = re.sub(
            r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+\-/=]+",
            self.redacted_value,
            value,
        )
        redacted = re.sub(
            r"(?i)\b(?:sk|key|token)-[a-z0-9_-]{8,}\b",
            self.redacted_value,
            redacted,
        )
        redacted = re.sub(
            r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*[^\s,;&]+",
            lambda match: f"{match.group(1)}={self.redacted_value}",
            redacted,
        )
        redacted = self.redact_urls(redacted)
        limit = self.max_string_length if max_length is None else max_length
        if truncate and len(redacted) > limit:
            redacted = redacted[:limit] + self.truncated_suffix
        return redacted

    def redact_urls(self, value: str) -> str:
        url_pattern = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

        def replace_url(match: re.Match[str]) -> str:
            raw_url = match.group(0)
            try:
                parsed = urlsplit(raw_url)
                hostname = parsed.hostname or ""
                port = f":{parsed.port}" if parsed.port is not None else ""
                netloc = f"{hostname}{port}"
                query = urlencode([
                    (
                        key,
                        self.redacted_value if self.is_sensitive_key(key) else item,
                    )
                    for key, item in parse_qsl(parsed.query, keep_blank_values=True)
                ])
                return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
            except ValueError:
                return "[REDACTED_URL]"

        return url_pattern.sub(replace_url, value)

    def redact(self, value: object, *, depth: int = 0) -> object:
        """Return a detached JSON-compatible, bounded, recursively redacted value."""
        if depth >= self.max_depth:
            return "[MAX_DEPTH]"
        if isinstance(value, BaseException):
            return {"error_type": type(value).__name__}
        if isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            items = list(mapping.items())[: self.max_collection_items]
            result = {
                str(key): (
                    self.redacted_value
                    if self.is_sensitive_key(str(key))
                    else self.redact(item, depth=depth + 1)
                )
                for key, item in items
            }
            if len(mapping) > self.max_collection_items:
                result["truncated_items"] = len(mapping) - self.max_collection_items
            return result
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = list(cast(Sequence[object], value))
            result = [
                self.redact(item, depth=depth + 1)
                for item in items[: self.max_collection_items]
            ]
            if len(items) > self.max_collection_items:
                result.append(f"[TRUNCATED_ITEMS:{len(items) - self.max_collection_items}]")
            return result
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, (bytes, bytearray)):
            return f"[BINARY:{len(value)} bytes]"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return self.redact_text(str(value))

    def summarize(self, value: object, *, max_length: int = 2048) -> str:
        """Serialize a redacted value to a bounded one-line JSON summary."""
        serialized = json.dumps(
            self.redact(value),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return self.redact_text(serialized, max_length=max_length)


DEFAULT_REDACTION_POLICY = RedactionPolicy()


def redact_observability_value(value: object) -> object:
    """Use the process-immutable default policy at every observability boundary."""
    return DEFAULT_REDACTION_POLICY.redact(value)


def summarize_observability_value(value: object, *, max_length: int = 2048) -> str:
    """Return a bounded redacted summary for public metadata fields."""
    return DEFAULT_REDACTION_POLICY.summarize(value, max_length=max_length)
