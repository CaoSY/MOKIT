# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""General-purpose utility helpers for MOKIT."""

from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any, Protocol, cast, overload


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
    callable receives the destination as its ``dest`` keyword argument. This supports
    in-place-style callables, but no in-place behavior is required or enforced: the
    callable may modify ``dest`` and return it, return another value, or return ``None``.
    The returned list contains exactly the callable's return values.

    Args:
        src: The source values passed as the callable's first argument.
        func: The callable applied to each source value.
        *args: Positional arguments forwarded to the callable.
        dest: Optional destination values paired with ``src``.
        **kwargs: Keyword arguments forwarded to the callable.

    Returns:
        The results returned by the callable, in source iteration order. When ``dest`` is
        provided, these are not necessarily the destination values.

    Raises:
        ValueError: If ``dest`` and ``src`` have different lengths.
        Exception: Any exception raised by the callable propagates unchanged.

    """
    if dest is None:
        return [func(src_val, *args, **kwargs) for src_val in src]
    else:
        return [func(src_val, *args, dest=dest_val, **kwargs) for (src_val, dest_val) in zip(src, dest, strict=True)]


def apply_fn_by_id[SrcType, SrcIDType: Hashable, DestType, DestIDType: Hashable, ResultType](
    src: Sequence[SrcType],
    func: Callable[..., ResultType],
    *args: Any,
    dest: Sequence[DestType] | None = None,
    src_idfunc: Callable[[SrcType], SrcIDType] = id,
    dest_idfunc: Callable[[DestType], DestIDType] = id,
    **kwargs: Any,
) -> list[ResultType]:
    """
    Apply a callable once per identity group and restore source order.

    Values sharing the same ``src_idfunc`` identity share one result. If ``dest`` is
    provided, ``dest_idfunc`` defines destination identities, and its group topology must
    match the source grouping. The callable receives one representative destination per
    group. Like ``apply_fn``, ``dest`` is forwarded as a keyword argument and the callable
    is not required to return it; the returned list contains the values returned by the
    callable.

    Args:
        src: Source values.
        func: Callable applied once per source identity group.
        *args: Positional arguments forwarded to ``func``.
        dest: Optional destination values with the same grouping topology as ``src``.
        src_idfunc: Callable defining each source value's identity. Defaults to ``id``.
        dest_idfunc: Callable defining each destination value's identity. Defaults to ``id``.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        Results aligned with the original ``src`` sequence.

    Raises:
        ValueError: If ``dest`` grouping does not match ``src`` grouping.

    """
    src_groups = group_by_id(src, idfunc=src_idfunc)

    if dest is None:
        unique_dest = None
    else:
        dest_groups = group_by_id(dest, idfunc=dest_idfunc)
        if tuple(src_groups.values()) != tuple(dest_groups.values()):
            raise ValueError("src and dest grouping differ")
        unique_dest = (dest[group[0]] for group in dest_groups.values())

    unique_src = (src[group[0]] for group in src_groups.values())

    unique_output = apply_fn(unique_src, func, *args, dest=unique_dest, **kwargs)

    output: list[ResultType | None] = [None] * len(src)
    for value, group in zip(unique_output, src_groups.values(), strict=True):
        for index in group:
            output[index] = value

    return cast(list[ResultType], output)


__all__ = ["apply_fn", "apply_fn_by_id", "group_by_id"]
