# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.utils`."""

from typing import Any

import pytest

from mokit.utils import apply_fn, group_by_id

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


# ==============================================================================
# group_by_id
# ==============================================================================


def test_group_by_id_groups_indices_by_object_identity() -> None:
    """Test that indices are grouped by exact object identity."""
    first = object()
    second = object()
    third = object()

    groups = group_by_id([first, second, first, third, second])

    assert list(groups) == [id(first), id(second), id(third)]
    assert groups == {
        id(first): [0, 2],
        id(second): [1, 4],
        id(third): [3],
    }


def test_group_by_id_preserves_first_occurrence_order() -> None:
    """Test that group and member order follow first occurrence."""
    values = ["b", "a", "b", "c", "a", "d"]

    groups = group_by_id(values, idfunc=lambda value: value)

    assert list(groups) == ["b", "a", "c", "d"]
    assert groups == {
        "b": [0, 2],
        "a": [1, 4],
        "c": [3],
        "d": [5],
    }


def test_group_by_id_keeps_equal_but_distinct_objects_separate() -> None:
    """Test that equal objects with different identities form separate groups."""
    left: list[int] = []
    right: list[int] = []

    groups = group_by_id([left, right, left])

    assert list(groups) == [id(left), id(right)]
    assert groups == {
        id(left): [0, 2],
        id(right): [1],
    }


def test_group_by_id_groups_caller_keys_by_object_identity() -> None:
    """Test that caller-provided keys are grouped by object identity."""
    first = object()
    second = object()

    groups = group_by_id([first, second, first], ["left", "middle", "right"])

    assert list(groups) == [id(first), id(second)]
    assert groups == {
        id(first): ["left", "right"],
        id(second): ["middle"],
    }


def test_group_by_id_uses_custom_idfunc() -> None:
    """Test that a custom idfunc defines the grouping identity."""
    groups = group_by_id([1, 2, 3, 4], idfunc=lambda value: value % 2)

    assert list(groups) == [1, 0]
    assert groups == {1: [0, 2], 0: [1, 3]}


def test_group_by_id_uses_custom_idfunc_with_caller_keys() -> None:
    """Test that custom identity grouping also applies to caller keys."""
    groups = group_by_id(
        [1, 2, 3, 4],
        ["a", "b", "c", "d"],
        idfunc=lambda value: value % 2,
    )

    assert list(groups) == [1, 0]
    assert groups == {1: ["a", "c"], 0: ["b", "d"]}


def test_group_by_id_handles_empty_inputs() -> None:
    """Test that empty values produce no groups."""
    assert group_by_id([]) == {}
    assert group_by_id([], keys=[]) == {}


@pytest.mark.parametrize(
    ("values", "keys"),
    [
        pytest.param([1], [1, 2], id="keys_too_long"),
        pytest.param([1, 2], [1], id="keys_too_short"),
    ],
)
def test_group_by_id_rejects_length_mismatch(values: list[int], keys: list[int]) -> None:
    """Test that values and keys must have equal lengths."""
    with pytest.raises(ValueError):
        group_by_id(values, keys)


def test_group_by_id_rejects_unhashable_identity() -> None:
    """Test that an idfunc returning an unhashable value is rejected."""

    def unhashable_idfunc(_value: int) -> Any:
        """Return an unhashable identity."""
        return []

    with pytest.raises(TypeError):
        group_by_id([1], idfunc=unhashable_idfunc)
