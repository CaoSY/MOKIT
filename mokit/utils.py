# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""General-purpose utility helpers for MOKIT."""

from collections.abc import Callable, Hashable, Iterable
from typing import Any, Protocol, overload


def _group_keys_by_value_id[KeyType, ValueType, IDType: Hashable](
    pairs: Iterable[tuple[KeyType, ValueType]],
    idfunc: Callable[[ValueType], IDType],
) -> dict[IDType, list[KeyType]]:
    """
    Group pair keys by the identity of their paired values.

    Args:
        pairs: Key/value pairs to group.
        idfunc: Callable defining each value's identity.

    Returns:
        A mapping from identities to keys, ordered by first occurrence. Keys within each
        group preserve sequence order.

    """
    groups: dict[IDType, list[KeyType]] = {}

    for key, value in pairs:
        identity = idfunc(value)
        group = groups.get(identity)
        if group is None:
            groups[identity] = [key]
        else:
            group.append(key)

    return groups


@overload
def group_by_id[ValueType, IDType: Hashable](
    values: Iterable[ValueType],
    keys: None = None,
    idfunc: Callable[[ValueType], IDType] = id,
) -> dict[IDType, list[int]]: ...


@overload
def group_by_id[KeyType, ValueType, IDType: Hashable](
    values: Iterable[ValueType],
    keys: Iterable[KeyType],
    idfunc: Callable[[ValueType], IDType] = id,
) -> dict[IDType, list[KeyType]]: ...


def group_by_id[KeyType, ValueType, IDType: Hashable](
    values: Iterable[ValueType],
    keys: Iterable[KeyType] | None = None,
    idfunc: Callable[[ValueType], IDType] = id,
) -> dict[IDType, list[int]] | dict[IDType, list[KeyType]]:
    """
    Group keys by values' identities.

    ``keys`` defaults to the indices of ``values``. When provided, ``keys`` is paired
    positionally with ``values``. The identity of each value is defined by
    ``idfunc(value)``. ``idfunc`` defaults to ``id``, so values are grouped by exact
    object identity; a custom ``idfunc`` changes what counts as the same identity.

    Groups are ordered by first occurrence, and members within each group preserve
    sequence order. This relies on Python's guarantee that dictionaries preserve
    insertion order.

    Args:
        values: Values whose derived identities determine the grouping.
        keys: Optional labels paired positionally with ``values``. Defaults to the
            indices of ``values``.
        idfunc: Callable defining each value's identity. Defaults to ``id``, grouping by
            exact object identity.

    Returns:
        A mapping from identities to positions or caller-provided keys, ordered by first
        occurrence. Keys within each group preserve sequence order.

    Raises:
        ValueError: If ``keys`` and ``values`` have different lengths.
        TypeError: If ``idfunc`` returns an unhashable value.
        Exception: Any exception raised by ``idfunc`` propagates unchanged.

    """
    if keys is None:
        return _group_keys_by_value_id(enumerate(values), idfunc)
    return _group_keys_by_value_id(zip(keys, values, strict=True), idfunc)


class _SourceFunc[SourceType, ResultType](Protocol):
    """The protocol for the callable passed to ``apply_fn`` without destinations."""

    def __call__(
        self,
        source: SourceType,
        *args: Any,
        **kwargs: Any,
    ) -> ResultType:
        """Call the function with a source value."""


class _DestFunc[SourceType, DestType, ResultType](Protocol):
    """The protocol for the callable passed to ``apply_fn`` with destinations."""

    def __call__(
        self,
        source: SourceType,
        *args: Any,
        dest: DestType,
        **kwargs: Any,
    ) -> ResultType:
        """Call the function with a source value and destination."""


@overload
def apply_fn[SourceType, ResultType](
    src: Iterable[SourceType],
    func: _SourceFunc[SourceType, ResultType],
    *args: Any,
    dest: None = None,
    **kwargs: Any,
) -> list[ResultType]: ...


@overload
def apply_fn[SourceType, DestType, ResultType](
    src: Iterable[SourceType],
    func: _DestFunc[SourceType, DestType, ResultType],
    *args: Any,
    dest: Iterable[DestType],
    **kwargs: Any,
) -> list[ResultType]: ...


def apply_fn[SourceType, DestType, ResultType](
    src: Iterable[SourceType],
    func: Callable[..., ResultType],
    *args: Any,
    dest: Iterable[DestType] | None = None,
    **kwargs: Any,
) -> list[ResultType]:
    """
    Apply a callable to each source value.

    When ``dest`` is ``None``, the callable is invoked without a destination argument.
    Otherwise, source and destination values are paired by iteration order and the
    callable is invoked with the destination as its ``dest`` keyword argument.

    Args:
        src: The source values passed as the callable's first argument.
        func: The callable applied to each source value.
        *args: Positional arguments forwarded to the callable.
        dest: Optional destination values paired with ``src``.
        **kwargs: Keyword arguments forwarded to the callable.

    Returns:
        The callable results in source iteration order.

    Raises:
        ValueError: If ``dest`` and ``src`` have different lengths.
        Exception: Any exception raised by the callable propagates unchanged.

    """
    if dest is None:
        return [func(src_val, *args, **kwargs) for src_val in src]
    else:
        return [func(src_val, *args, dest=dest_val, **kwargs) for (src_val, dest_val) in zip(src, dest, strict=True)]


__all__ = ["apply_fn", "group_by_id"]
