# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.quantization.formats.mx_formats`."""

import dataclasses
from pathlib import Path
from typing import cast

import pytest
import torch

from mokit.quantization.formats import (
    MXFormat,
    MXScaleCalculationMode,
    PerBlock,
    PerTensor,
    QuantizationRoundingMode,
)
from mokit.quantization.formats.mx_formats import BlockMXFormat

MX_FORMAT_CLASSES: list[type[MXFormat]] = [MXFormat, BlockMXFormat]


@pytest.mark.parametrize("cls", MX_FORMAT_CLASSES)
@pytest.mark.parametrize(
    ("dtype", "expected_max"),
    [
        (torch.float8_e4m3fn, 448.0),
        (torch.float8_e4m3fnuz, 240.0),
        (torch.float8_e5m2, 57344.0),
        (torch.float4_e2m1fn_x2, 6.0),
        (torch.int8, 127.0),
    ],
)
def test_elem_dtype_max_is_derived(cls: type[MXFormat], dtype: torch.dtype, expected_max: float) -> None:
    """Test that elem_dtype_max is derived from elem_dtype when unset."""
    assert cls(elem_dtype=dtype).elem_dtype_max == expected_max


@pytest.mark.parametrize("cls", MX_FORMAT_CLASSES)
def test_explicit_elem_dtype_max(cls: type[MXFormat]) -> None:
    """Test that an explicit elem_dtype_max overrides the derived value."""
    assert cls(elem_dtype_max=100.0).elem_dtype_max == 100.0


@pytest.mark.parametrize("cls", MX_FORMAT_CLASSES)
def test_elem_dtype_max_must_be_positive(cls: type[MXFormat]) -> None:
    """Test that a non-positive elem_dtype_max is rejected."""
    with pytest.raises(ValueError, match="positive"):
        cls(elem_dtype_max=0.0)


@pytest.mark.parametrize("cls", MX_FORMAT_CLASSES)
def test_unsupported_elem_dtype(cls: type[MXFormat]) -> None:
    """Test that an unsupported element dtype is rejected."""
    with pytest.raises(ValueError, match="Supported element dtypes"):
        cls(elem_dtype=torch.float16)


@pytest.mark.parametrize("cls", MX_FORMAT_CLASSES)
def test_invalid_rounding_mode(cls: type[MXFormat]) -> None:
    """Test that a non-QuantizationRoundingMode round mode is rejected."""
    with pytest.raises(ValueError, match="rounding_mode"):
        cls(rounding_mode=cast(QuantizationRoundingMode, "rne"))


@pytest.mark.parametrize("cls", MX_FORMAT_CLASSES)
def test_scale_dtype_is_fixed(cls: type[MXFormat]) -> None:
    """Test that a scale dtype other than E8M0 is rejected."""
    with pytest.raises(ValueError, match="fixed at E8M0"):
        cls(scale_dtype=torch.float32)


@pytest.mark.parametrize("cls", MX_FORMAT_CLASSES)
def test_invalid_scale_mode(cls: type[MXFormat]) -> None:
    """Test that a non-MXScaleCalculationMode scale mode is rejected."""
    with pytest.raises(ValueError, match="scale_mode"):
        cls(scale_mode=cast(MXScaleCalculationMode, "rceil"))


@pytest.mark.parametrize("cls", MX_FORMAT_CLASSES)
def test_granularity_must_be_per_block(cls: type[MXFormat]) -> None:
    """Test that a granularity other than PerBlock is rejected."""
    with pytest.raises(ValueError, match="must be of type PerBlock"):
        cls(granularity=cast(PerBlock, PerTensor()))


@pytest.mark.parametrize("cls", MX_FORMAT_CLASSES)
@pytest.mark.parametrize("block_size", [(0,), (-1,), (), (32, 0)])
def test_granularity_sizes_must_be_positive(cls: type[MXFormat], block_size: tuple[int, ...]) -> None:
    """Test that block sizes must be non-empty and positive."""
    with pytest.raises(ValueError, match="positive"):
        cls(granularity=PerBlock(*block_size))


@pytest.mark.parametrize(
    ("cls", "granularity"),
    [(MXFormat, PerBlock(10)), (BlockMXFormat, PerBlock(10, 10))],
    ids=["MXFormat", "BlockMXFormat"],
)
def test_granularity_is_hardware_agnostic(cls: type[MXFormat], granularity: PerBlock) -> None:
    """Test that block sizes need not be multiples of the default block size."""
    assert cls(granularity=granularity).granularity == granularity


def test_block_mx_format_defaults_to_a_square_block() -> None:
    """Test that BlockMXFormat defaults to a two-dimensional square block."""
    assert BlockMXFormat().granularity == PerBlock(32, 32)


@pytest.mark.parametrize("block_size", [(32,), (32, 64), (32, 32, 32)])
def test_block_mx_format_requires_a_square_block(block_size: tuple[int, ...]) -> None:
    """Test that BlockMXFormat rejects blocks that are not two-dimensional squares."""
    with pytest.raises(ValueError, match="square quantization block"):
        BlockMXFormat(granularity=PerBlock(*block_size))


def test_block_mx_format_accepts_a_square_block() -> None:
    """Test that BlockMXFormat accepts a two-dimensional square block."""
    assert BlockMXFormat(granularity=PerBlock(64, 64)).granularity == PerBlock(64, 64)


@pytest.mark.parametrize("cls", [MXFormat, BlockMXFormat])
def test_is_a_frozen_slotted_dataclass(cls: type) -> None:
    """Test that MXFormat and BlockMXFormat are frozen, slotted dataclasses."""
    assert dataclasses.is_dataclass(cls)
    assert not hasattr(cls(), "__dict__")
    attribute = "attribute"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(cls(), attribute, 1)


@pytest.mark.parametrize("cls", [MXScaleCalculationMode, MXFormat, BlockMXFormat])
def test_class_is_registered_as_safe_global(cls: type) -> None:
    """Test that every serializable MX format class is registered for weights-only loading."""
    assert cls in torch.serialization.get_safe_globals(), f"{cls.__name__} is not registered as a safe global"


@pytest.mark.parametrize(
    "mx_format",
    [MXFormat(), MXFormat(granularity=PerBlock(64, 64)), BlockMXFormat()],
)
def test_ser_des_round_trips(mx_format: MXFormat, tmp_path: Path) -> None:
    """Test that MX formats survive a weights-only round trip."""
    path = tmp_path / "mx_format.pt"
    torch.save(mx_format, path)
    assert torch.load(path, weights_only=True) == mx_format
