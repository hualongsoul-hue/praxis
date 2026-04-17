"""即时检索（Just-in-Time Retrieval）。

维护轻量级标识符（文件名、函数签名），
仅在需要时动态加载完整内容。
Few-shot 示例管理：按任务类型索引和按需注入。
"""

from typing import Any

from praxis.telemetry.logger import get_logger

log = get_logger("context.jit_retrieval")


class IdentifierEntry:
    """轻量级标识符条目。"""

    __slots__ = ("identifier", "kind", "source", "metadata")

    def __init__(
        self,
        identifier: str,
        kind: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.identifier = identifier
        self.kind = kind
        self.source = source
        self.metadata = metadata or {}


class FewShotExample:
    """Few-shot 示例。"""

    __slots__ = ("task_type", "user_message", "assistant_response", "tags")

    def __init__(
        self,
        task_type: str,
        user_message: str,
        assistant_response: str,
        tags: list[str] | None = None,
    ) -> None:
        self.task_type = task_type
        self.user_message = user_message
        self.assistant_response = assistant_response
        self.tags = tags or []


class JITRetriever:
    """即时检索管理器。

    维护标识符索引和 Few-shot 示例集，
    按需提供完整内容和上下文示例。
    """

    def __init__(self) -> None:
        self.identifiers: dict[str, IdentifierEntry] = {}
        self.examples: list[FewShotExample] = []
        self.content_loader: ContentLoader | None = None

    def register_identifier(
        self,
        identifier: str,
        kind: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """注册轻量标识符。

        Args:
            identifier: 标识符名称（如文件路径、函数签名）。
            kind: 类型（file, function, class, symbol）。
            source: 来源（如项目路径）。
            metadata: 附加元数据。
        """
        self.identifiers[identifier] = IdentifierEntry(
            identifier=identifier,
            kind=kind,
            source=source,
            metadata=metadata,
        )

    def unregister_identifier(self, identifier: str) -> bool:
        """移除标识符。"""
        if identifier in self.identifiers:
            del self.identifiers[identifier]
            return True
        return False

    def get_identifier_index(self) -> list[dict[str, str]]:
        """获取所有标识符的轻量索引。

        Returns:
            标识符摘要列表。
        """
        return [
            {"identifier": e.identifier, "kind": e.kind, "source": e.source}
            for e in self.identifiers.values()
        ]

    def get_identifiers_by_kind(self, kind: str) -> list[IdentifierEntry]:
        """按类型过滤标识符。"""
        return [e for e in self.identifiers.values() if e.kind == kind]

    def add_example(
        self,
        task_type: str,
        user_message: str,
        assistant_response: str,
        tags: list[str] | None = None,
    ) -> None:
        """添加 Few-shot 示例。

        Args:
            task_type: 任务类型（如 coding, debugging, review）。
            user_message: 用户消息示例。
            assistant_response: 助手响应示例。
            tags: 标签列表。
        """
        self.examples.append(FewShotExample(
            task_type=task_type,
            user_message=user_message,
            assistant_response=assistant_response,
            tags=tags,
        ))

    def get_examples_for_task(
        self,
        task_type: str,
        max_examples: int = 3,
    ) -> list[dict[str, Any]]:
        """获取指定任务类型的 Few-shot 示例（OpenAI 消息格式）。

        Args:
            task_type: 任务类型。
            max_examples: 最多返回示例数。

        Returns:
            消息格式的示例列表。
        """
        matching = [e for e in self.examples if e.task_type == task_type]
        selected = matching[:max_examples]

        messages: list[dict[str, Any]] = []
        for ex in selected:
            messages.append({"role": "user", "content": ex.user_message})
            messages.append({"role": "assistant", "content": ex.assistant_response})
        return messages

    def set_content_loader(self, loader: "ContentLoader") -> None:
        """设置内容加载器。"""
        self.content_loader = loader

    async def load_content(self, identifier: str) -> str | None:
        """按需加载标识符的完整内容。

        Args:
            identifier: 标识符名称。

        Returns:
            完整内容，未找到时返回 None。
        """
        entry = self.identifiers.get(identifier)
        if entry is None:
            return None

        if self.content_loader is None:
            log.warning("内容加载器未设置", identifier=identifier)
            return None

        content = await self.content_loader.load(entry.source, entry.identifier)
        log.info("内容已加载", identifier=identifier, kind=entry.kind)
        return content


class ContentLoader:
    """内容加载器协议。

    通过 S5 工具系统或文件系统按需加载完整内容。
    """

    def __init__(self, read_func: Any) -> None:
        self.read_func = read_func

    async def load(self, source: str, identifier: str) -> str | None:
        """加载完整内容。

        Args:
            source: 来源路径。
            identifier: 标识符。

        Returns:
            文件内容。
        """
        return await self.read_func(source, identifier)
