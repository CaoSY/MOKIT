# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""General-purpose utility helpers for MOKIT."""

from collections.abc import Callable, Iterable
from typing import Any, Protocol, overload


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


__all__ = ["apply_fn"]
