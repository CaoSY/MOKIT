# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.quantization.formats.float_formats`."""

from pathlib import Path
from typing import cast

import pytest
import torch

from mokit.quantization.formats import (
    FloatFormat,
    Granularity,
    PerBlock,
    PerDim,
    PerSlice,
    PerTensor,
    QuantizationRoundingMode,
)


def test_unsupported_elem_dtype() -> None:
    """Test that an element dtype outside the reduced precision group is rejected."""
    with pytest.raises(ValueError, match="elem_dtype"):
        FloatFormat(elem_dtype=torch.float16)


def test_invalid_rounding_mode() -> None:
    """Test that a non-QuantizationRoundingMode round mode is rejected."""
    with pytest.raises(ValueError, match="rounding_mode"):
        FloatFormat(rounding_mode=cast(QuantizationRoundingMode, "rne"))


def test_mx_scale_dtype_is_rejected() -> None:
    """Test that the MX scale dtype is rejected in favor of MXFormat."""
    with pytest.raises(ValueError, match="use MXFormat instead"):
        FloatFormat(scale_dtype=torch.float8_e8m0fnu)


def test_scale_dtype_must_be_high_precision() -> None:
    """Test that a scale dtype outside the high precision group is rejected."""
    with pytest.raises(ValueError, match="scale_dtype"):
        FloatFormat(scale_dtype=torch.float8_e5m2)


@pytest.mark.parametrize(
    "granularity",
    [PerTensor(), PerSlice(axis=1), PerDim(dim=0), PerBlock(32, 32)],
)
def test_accepts_any_granularity(granularity: Granularity) -> None:
    """Test that FloatFormat accepts every granularity variant."""
    assert FloatFormat(granularity=granularity).granularity == granularity


def test_class_is_registered_as_safe_global() -> None:
    """Test that FloatFormat is registered for weights-only loading."""
    assert FloatFormat in torch.serialization.get_safe_globals(), "FloatFormat is not registered as a safe global"


@pytest.mark.parametrize(
    "float_format",
    [FloatFormat(), FloatFormat(scale_dtype=torch.float16, granularity=PerBlock(32))],
)
def test_ser_des_round_trips(float_format: FloatFormat, tmp_path: Path) -> None:
    """Test that float formats survive a weights-only round trip."""
    path = tmp_path / "float_format.pt"
    torch.save(float_format, path)
    assert torch.load(path, weights_only=True) == float_format
