"""Small immutable mapping used by public configuration snapshots."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import cast


class FrozenMapping[KeyT, ValueT](Mapping[KeyT, ValueT]):
    """Tuple-backed mapping with no mutable storage exposed to callers."""

    def __init__(self, values: Mapping[KeyT, ValueT] | None = None) -> None:
        source = values or {}
        self.entries = tuple((key, value) for key, value in source.items())

    def __getitem__(self, key: KeyT) -> ValueT:
        for item_key, value in self.entries:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[KeyT]:
        for entry in self.entries:
            yield entry[0]

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self.entries)!r})"

    def __deepcopy__(self, memo: dict[int, object]) -> "FrozenMapping[KeyT, ValueT]":
        return FrozenMapping({
            deepcopy(key, memo): deepcopy(value, memo)
            for key, value in self.entries
        })


def freeze_value(value: object) -> object:
    """Recursively convert mappings and sequences into immutable equivalents."""
    if isinstance(value, FrozenMapping):
        return cast(object, value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return FrozenMapping({str(key): freeze_value(item) for key, item in mapping.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in cast(list[object] | tuple[object, ...], value))
    return value


def freeze_mapping[MapValueT](
    value: Mapping[str, MapValueT],
) -> FrozenMapping[str, MapValueT]:
    """Freeze a typed string-keyed mapping and all nested JSON containers."""
    frozen = freeze_value(value)
    return cast(FrozenMapping[str, MapValueT], frozen)


def thaw_value(value: object) -> object:
    """Convert immutable public values back to ordinary JSON-compatible containers."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): thaw_value(item) for key, item in mapping.items()}
    if isinstance(value, tuple):
        return [thaw_value(item) for item in cast(tuple[object, ...], value)]
    return value
