# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.quantization.formats.float_formats`."""

import dataclasses
from pathlib import Path
from typing import cast

import pytest
import torch

from mokit.quantization.formats import (
    FloatFormat,
    FloatQuantConfig,
    Granularity,
    PerBlock,
    PerDim,
    PerSlice,
    PerTensor,
    QuantizationRoundingMode,
)

FLOAT_CLASSES: list[type[FloatFormat]] = [FloatFormat, FloatQuantConfig]


@pytest.mark.parametrize("cls", FLOAT_CLASSES)
def test_unsupported_elem_dtype(cls: type[FloatFormat]) -> None:
    """Test that an element dtype outside the reduced precision group is rejected."""
    with pytest.raises(ValueError, match="elem_dtype"):
        cls(elem_dtype=torch.float16)


def test_invalid_rounding_mode() -> None:
    """Test that a non-QuantizationRoundingMode round mode is rejected."""
    with pytest.raises(ValueError, match="rounding_mode"):
        FloatQuantConfig(rounding_mode=cast(QuantizationRoundingMode, "rne"))


@pytest.mark.parametrize("cls", FLOAT_CLASSES)
def test_mx_scale_dtype_is_rejected(cls: type[FloatFormat]) -> None:
    """Test that the MX scale dtype is rejected in favor of MXFormat."""
    with pytest.raises(ValueError, match="use MXFormat instead"):
        cls(scale_dtype=torch.float8_e8m0fnu)


@pytest.mark.parametrize("cls", FLOAT_CLASSES)
def test_scale_dtype_must_be_high_precision(cls: type[FloatFormat]) -> None:
    """Test that a scale dtype outside the high precision group is rejected."""
    with pytest.raises(ValueError, match="scale_dtype"):
        cls(scale_dtype=torch.float8_e5m2)


@pytest.mark.parametrize("cls", FLOAT_CLASSES)
@pytest.mark.parametrize(
    "granularity",
    [PerTensor(), PerSlice(axis=1), PerDim(dim=0), PerBlock(32, 32)],
)
def test_accepts_any_granularity(cls: type[FloatFormat], granularity: Granularity) -> None:
    """Test that float format classes accept every granularity variant."""
    assert cls(granularity=granularity).granularity == granularity


@pytest.mark.parametrize("cls", FLOAT_CLASSES)
def test_is_a_frozen_slotted_dataclass(cls: type) -> None:
    """Test that float format classes are frozen, slotted dataclasses."""
    assert dataclasses.is_dataclass(cls)
    assert not hasattr(cls(), "__dict__")
    attribute = "attribute"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(cls(), attribute, 1)


@pytest.mark.parametrize("cls", FLOAT_CLASSES)
def test_class_is_registered_as_safe_global(cls: type) -> None:
    """Test that every serializable float format class is registered for weights-only loading."""
    assert cls in torch.serialization.get_safe_globals(), f"{cls.__name__} is not registered as a safe global"


@pytest.mark.parametrize(
    "float_format",
    [
        FloatFormat(),
        FloatFormat(scale_dtype=torch.float16, granularity=PerBlock(32)),
        FloatQuantConfig(),
        FloatQuantConfig(
            scale_dtype=torch.float16,
            granularity=PerBlock(32),
            rounding_mode=QuantizationRoundingMode.STOCHASTIC,
        ),
    ],
)
def test_ser_des_round_trips(float_format: FloatFormat, tmp_path: Path) -> None:
    """Test that float formats survive a weights-only round trip."""
    path = tmp_path / "float_format.pt"
    torch.save(float_format, path)
    assert torch.load(path, weights_only=True) == float_format
