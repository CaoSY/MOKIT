# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.utils`."""

from typing import Any

import pytest

from mokit.utils import apply_fn, apply_fn_by_id, group_by_id

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


def test_apply_fn_allows_in_place_callables_returning_none() -> None:
    """Test that apply_fn accepts a destination callable whose result is None."""
    destinations: list[dict[str, int]] = [{"value": 0}, {"value": 0}]

    def assign(source: int, *, dest: dict[str, int]) -> None:
        """Assign the source value to the destination."""
        dest["value"] = source

    results = apply_fn([1, 2], assign, dest=destinations)

    assert results == [None, None]
    assert [destination["value"] for destination in destinations] == [1, 2]


def test_apply_fn_returns_callable_results_not_destinations() -> None:
    """Test that apply_fn returns callable values rather than destination objects."""
    destinations: list[dict[str, int]] = [{"value": 0}, {"value": 0}]

    def assign_and_return_source(source: int, *, dest: dict[str, int]) -> int:
        """Assign the source value and return the source."""
        dest["value"] = source
        return source

    results = apply_fn([1, 2], assign_and_return_source, dest=destinations)

    assert results == [1, 2]
    assert results[0] is not destinations[0]
    assert [destination["value"] for destination in destinations] == [1, 2]


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


# ==============================================================================
# apply_fn_by_id
# ==============================================================================


def test_apply_fn_by_id_applies_once_per_identity_and_restores_order() -> None:
    """Test that the callable runs once per identity group and output order is restored."""
    first = object()
    second = object()
    third = object()
    calls: list[object] = []

    def transform(value: object) -> int:
        """Record the representative value and return its call number."""
        calls.append(value)
        return len(calls)

    result = apply_fn_by_id([first, second, first, third, second], transform)

    assert calls == [first, second, third]
    assert result == [1, 2, 1, 3, 2]


def test_apply_fn_by_id_reuses_result_for_aliases() -> None:
    """Test that aliased source positions share the exact same result object."""
    first = object()
    second = object()

    def transform(_value: object) -> object:
        """Return a fresh result object."""
        return object()

    result = apply_fn_by_id([first, second, first], transform)

    assert result[0] is result[2]
    assert result[0] is not result[1]


def test_apply_fn_by_id_pairs_dest_groups_and_restores_order() -> None:
    """Test that destination representatives are paired by matching group topology."""
    first = object()
    second = object()
    third = object()
    dest_first = object()
    dest_second = object()
    dest_third = object()
    calls: list[tuple[object, object]] = []

    def transform(source: object, *, dest: object) -> tuple[object, object]:
        """Record and return the source/destination group representatives."""
        calls.append((source, dest))
        return source, dest

    result = apply_fn_by_id(
        [first, second, first, third, second],
        transform,
        dest=[dest_first, dest_second, dest_first, dest_third, dest_second],
    )

    assert calls == [(first, dest_first), (second, dest_second), (third, dest_third)]
    assert result == [
        (first, dest_first),
        (second, dest_second),
        (first, dest_first),
        (third, dest_third),
        (second, dest_second),
    ]


class _DestHolder:
    """A destination object with a mutable inner field."""

    def __init__(self, inner: object) -> None:
        """Store the inner value."""
        self.inner = inner


def test_apply_fn_by_id_updates_dest_fields_without_replacing_instances() -> None:
    """Test that destination instances and aliasing survive field mutation."""
    first = object()
    second = object()
    src = [first, second, first]
    dest_first = _DestHolder(None)
    dest_second = _DestHolder(None)
    dest = [dest_first, dest_second, dest_first]

    def transform(source: object, *, dest: _DestHolder) -> _DestHolder:
        """Assign the source value to the destination's inner field."""
        dest.inner = source
        return dest

    result = apply_fn_by_id(src, transform, dest=dest)

    assert dest[0] is dest_first
    assert dest[1] is dest_second
    assert dest[2] is dest_first
    assert dest_first.inner is first
    assert dest_second.inner is second
    assert result == [dest_first, dest_second, dest_first]
    assert result[0] is result[2]


def test_apply_fn_by_id_forwards_args_and_kwargs() -> None:
    """Test that positional and keyword arguments reach the callable."""
    values = [object(), object()]
    calls: list[tuple[int, int]] = []

    def transform(_value: object, scale: int, *, offset: int) -> int:
        """Record and return the forwarded arguments."""
        calls.append((scale, offset))
        return scale + offset

    result = apply_fn_by_id(values, transform, 2, offset=1)

    assert calls == [(2, 1), (2, 1)]
    assert result == [3, 3]


def test_apply_fn_by_id_uses_separate_identity_functions() -> None:
    """
    Test that source and destination identities use their own callables.

    The source is grouped by parity, giving position groups ``[0, 2]`` and
    ``[1, 3]``. The destination is grouped by divisibility by three, which has the
    same topology and therefore pairs with the source groups. The test verifies
    both mismatch directions: only applying parity to the destination gives ``[0, 1]``
    and ``[2, 3]``, and only applying divisibility by three to the source gives a
    different topology. The call trackers verify that each identity function is used only on
    its own sequence.

    """
    src = [1, 2, 3, 4]
    dest = [3, 1, 6, 2]
    calls: list[tuple[int, int]] = []
    src_id_calls: list[int] = []
    dest_id_calls: list[int] = []

    def transform(source: int, *, dest: int) -> tuple[int, int]:
        """Record and return the source/destination group representatives."""
        calls.append((source, dest))
        return source, dest

    def parity(value: int) -> int:
        """Return the value's parity."""
        return value % 2

    def divisible_by_three(value: int) -> bool:
        """Return whether the value is divisible by three."""
        return value % 3 == 0

    def src_idfunc(value: int) -> int:
        """Group source values by parity."""
        src_id_calls.append(value)
        return parity(value)

    def dest_idfunc(value: int) -> bool:
        """Group destination values by divisibility by three."""
        dest_id_calls.append(value)
        return divisible_by_three(value)

    src_parity = group_by_id(src, idfunc=parity)
    dest_parity = group_by_id(dest, idfunc=parity)
    assert tuple(src_parity.values()) != tuple(dest_parity.values())

    src_div3 = group_by_id(src, idfunc=divisible_by_three)
    dest_div3 = group_by_id(dest, idfunc=divisible_by_three)
    assert tuple(src_div3.values()) != tuple(dest_div3.values())

    result = apply_fn_by_id(
        src,
        transform,
        dest=dest,
        src_idfunc=src_idfunc,
        dest_idfunc=dest_idfunc,
    )

    assert src_id_calls == src
    assert dest_id_calls == dest
    assert calls == [(1, 3), (2, 1)]
    assert result == [(1, 3), (2, 1), (1, 3), (2, 1)]


def test_apply_fn_by_id_rejects_mismatched_dest_grouping() -> None:
    """Test that destination grouping must match source grouping."""
    first = object()
    second = object()
    dest_first = object()
    dest_second = object()
    dest_third = object()

    def transform(source: object, *, dest: object) -> tuple[object, object]:
        """Return the paired representatives."""
        return source, dest

    with pytest.raises(ValueError, match="grouping differ"):
        apply_fn_by_id(
            [first, second, first],
            transform,
            dest=[dest_first, dest_second, dest_third],
        )


def test_apply_fn_by_id_handles_empty_inputs() -> None:
    """Test that empty source and destination sequences produce no output."""
    calls: list[object] = []

    def transform(value: object) -> int:
        """Record the value and return a constant."""
        calls.append(value)
        return 1

    assert apply_fn_by_id([], transform) == []
    assert apply_fn_by_id([], transform, dest=[]) == []
    assert calls == []
