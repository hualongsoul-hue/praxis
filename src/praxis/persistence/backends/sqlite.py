"""SQLite 存储后端（默认）。

零配置启动，基于 SQLAlchemy ORM + aiosqlite 异步驱动。
"""

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Index, LargeBinary, String, delete, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy ORM 声明基类。"""


class KVEntry(Base):
    """键值存储 ORM 模型。"""

    __tablename__ = "kv_store"
    __table_args__ = (
        Index("idx_kv_namespace", "namespace"),
        Index("idx_kv_namespace_key", "namespace", "key"),
    )

    namespace: Mapped[str] = mapped_column(String, primary_key=True)
    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class SqliteBackend:
    """SQLite 异步存储后端，基于 SQLAlchemy ORM。"""

    def __init__(
        self,
        engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory

    @classmethod
    async def create(cls, path: str) -> "SqliteBackend":
        """创建并初始化 SQLite 后端。自动创建目录和数据表。"""
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return cls(engine, session_factory)

    async def save(self, namespace: str, key: str, data: bytes) -> None:
        async with self._session_factory() as session:
            existing = await session.get(KVEntry, (namespace, key))
            if existing:
                existing.value = data
                existing.updated_at = datetime.now(timezone.utc)
            else:
                session.add(KVEntry(namespace=namespace, key=key, value=data))
            await session.commit()

    async def load(self, namespace: str, key: str) -> bytes | None:
        async with self._session_factory() as session:
            entry = await session.get(KVEntry, (namespace, key))
            return bytes(entry.value) if entry else None

    async def delete(self, namespace: str, key: str) -> None:
        async with self._session_factory() as session:
            stmt = delete(KVEntry).where(
                KVEntry.namespace == namespace,
                KVEntry.key == key,
            )
            await session.execute(stmt)
            await session.commit()

    async def list_keys(
        self, namespace: str, prefix: str | None = None
    ) -> list[str]:
        async with self._session_factory() as session:
            stmt = select(KVEntry.key).where(KVEntry.namespace == namespace)
            if prefix:
                stmt = stmt.where(KVEntry.key.startswith(prefix))
            result = await session.execute(stmt)
            return [row[0] for row in result.all()]

    async def clear_namespace(self, namespace: str) -> int:
        async with self._session_factory() as session:
            stmt = delete(KVEntry).where(KVEntry.namespace == namespace)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount  # type: ignore[return-value]

    async def close(self) -> None:
        await self._engine.dispose()
