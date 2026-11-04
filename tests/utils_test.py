# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.utils`."""

import pytest

from mokit.utils import apply_fn

# ==============================================================================
# apply_fn
# ==============================================================================


def test_apply_fn_maps_source_values_without_destinations() -> None:
    """Test that apply_fn maps source values and forwards arguments."""
    src = (value for value in (1, 2, 3))
    calls: list[tuple[int, int, int]] = []

    def transform(source: int, scale: int, *, offset: int) -> int:
        """Record the arguments and apply the transformation."""
        calls.append((source, scale, offset))
        return source * scale + offset

    results = apply_fn(src, transform, 2, offset=1)

    assert results == [3, 5, 7]
    assert calls == [(1, 2, 1), (2, 2, 1), (3, 2, 1)]


def test_apply_fn_pairs_destinations_and_forwards_dest_keyword() -> None:
    """Test that apply_fn pairs destinations and forwards the dest keyword."""
    src = ["a", "b"]
    dest = (value for value in (10, 20))

    def transform(source: str, *, dest: int, scale: int) -> str:
        """Combine a source, destination, and scale factor."""
        return f"{source}{dest * scale}"

    results = apply_fn(src, transform, dest=dest, scale=2)

    assert results == ["a20", "b40"]


def test_apply_fn_returns_empty_list_for_empty_sources() -> None:
    """Test that apply_fn returns an empty list for empty sources."""

    def identity(value: int) -> int:
        """Return the value unchanged."""
        return value

    assert apply_fn([], identity) == []
    assert apply_fn([], identity, dest=None) == []
    assert apply_fn([], identity, dest=[]) == []


def test_apply_fn_rejects_destination_length_mismatch() -> None:
    """Test that apply_fn rejects mismatched source and destination lengths."""

    def combine(source: int, *, dest: int) -> int:
        """Add a source and destination value."""
        return source + dest

    src = (value for value in (1, 2))
    dest = (value for value in (10,))

    with pytest.raises(ValueError):
        apply_fn(src, combine, dest=dest)


def test_apply_fn_propagates_exceptions() -> None:
    """Test that apply_fn propagates callable exceptions."""

    def fail(_value: int) -> int:
        """Raise an exception."""
        raise ValueError("failure")

    with pytest.raises(ValueError, match="failure"):
        apply_fn([1], fail)


def test_apply_fn_preserves_generic_result_types() -> None:
    """Test that apply_fn supports non-tensor source and result types."""

    def describe(value: int) -> tuple[int, str]:
        """Return a tuple containing the value and its string form."""
        return value, str(value)

    assert apply_fn([1, 2], describe) == [(1, "1"), (2, "2")]
