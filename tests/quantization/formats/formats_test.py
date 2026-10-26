# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.quantization.formats`."""

import dataclasses
from pathlib import Path
from typing import cast

import pytest
import torch

from mokit.quantization.formats import (
    Granularity,
    PerBlock,
    PerDim,
    PerSlice,
    PerTensor,
    QuantizationRoundingMode,
)

# ==============================================================================
# Granularity
# ==============================================================================


def test_granularity_is_a_frozen_slotted_dataclass() -> None:
    """Test that Granularity is a frozen, slotted dataclass."""
    assert dataclasses.is_dataclass(Granularity)
    assert not hasattr(Granularity(), "__dict__")
    attribute = "attribute"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(Granularity(), attribute, 1)


@pytest.mark.parametrize(
    "granularity",
    [PerTensor(), PerSlice(axis=0), PerDim(dim=1), PerBlock(32)],
)
def test_granularity_subclasses(granularity: Granularity) -> None:
    """Test that every granularity variant is a Granularity."""
    assert isinstance(granularity, Granularity)


# ==============================================================================
# PerSlice
# ==============================================================================

# ------------------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------------------


def test_per_slice_construction() -> None:
    """Test that PerSlice accepts positional and keyword axes."""
    assert PerSlice(1).axis == 1
    assert PerSlice(axis=1).axis == 1


# ==============================================================================
# PerDim
# ==============================================================================

# ------------------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------------------


def test_per_dim_construction() -> None:
    """Test that PerDim accepts positional and keyword dims."""
    assert PerDim(0).dim == 0
    assert PerDim(dim=0).dim == 0


# ==============================================================================
# PerBlock
# ==============================================================================

# ------------------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------------------


def test_per_block_construction() -> None:
    """Test that PerBlock packs variadic integers into a tuple."""
    assert PerBlock(32).block_size == (32,)
    assert PerBlock(32, 64).block_size == (32, 64)
    assert PerBlock().block_size == ()


def test_per_block_rejects_non_integers() -> None:
    """Test that PerBlock rejects non-integer block sizes."""
    with pytest.raises(TypeError):
        PerBlock(32, cast(int, "64"))


# ------------------------------------------------------------------------------
# Class-level
# ------------------------------------------------------------------------------


def test_per_block_is_frozen() -> None:
    """Test that PerBlock cannot be mutated."""
    field = "block_size"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(PerBlock(32), field, (64,))


# ------------------------------------------------------------------------------
# __len__
# ------------------------------------------------------------------------------


def test_per_block_length() -> None:
    """Test that len reports the number of dimensions."""
    assert len(PerBlock()) == 0
    assert len(PerBlock(32, 64, 16)) == 3


# ------------------------------------------------------------------------------
# __iter__
# ------------------------------------------------------------------------------


def test_per_block_iteration() -> None:
    """Test that PerBlock iterates over its per-dimension sizes."""
    assert list(PerBlock(32, 64, 16)) == [32, 64, 16]
    assert 64 in PerBlock(32, 64, 16)


# ------------------------------------------------------------------------------
# __getitem__
# ------------------------------------------------------------------------------


def test_per_block_indexing() -> None:
    """Test that PerBlock indexes its per-dimension sizes."""
    block = PerBlock(32, 64, 16)
    assert block[0] == 32
    assert block[-1] == 16


def test_per_block_slicing_returns_per_block() -> None:
    """Test that slicing a PerBlock returns a new PerBlock."""
    sliced = PerBlock(32, 64, 16)[1:]
    assert isinstance(sliced, PerBlock)
    assert sliced == PerBlock(64, 16)


# ------------------------------------------------------------------------------
# __eq__ / __hash__
# ------------------------------------------------------------------------------


def test_per_block_equality_and_hash() -> None:
    """Test that equal PerBlock values compare and hash equal."""
    assert PerBlock(32, 64) == PerBlock(32, 64)
    assert PerBlock(32, 64) != PerBlock(64, 32)
    assert hash(PerBlock(32, 64)) == hash(PerBlock(32, 64))


# ==============================================================================
# Serialization
# ==============================================================================

# ------------------------------------------------------------------------------
# Safe-global registration
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [QuantizationRoundingMode, PerTensor, PerSlice, PerDim, PerBlock],
)
def test_class_is_registered_as_safe_global(cls: type) -> None:
    """Test that every serializable format class is registered for weights-only loading."""
    assert cls in torch.serialization.get_safe_globals(), f"{cls.__name__} is not registered as a safe global"


# ------------------------------------------------------------------------------
# Weights-only round trip
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [PerTensor(), PerSlice(1), PerDim(0), PerBlock(32, 32), QuantizationRoundingMode.RNE],
)
def test_ser_des_round_trips(value: object, tmp_path: Path) -> None:
    """Test that formats survive a weights-only round trip."""
    path = tmp_path / "value.pt"
    torch.save(value, path)
    assert torch.load(path, weights_only=True) == value
