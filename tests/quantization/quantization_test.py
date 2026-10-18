# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.quantization`."""

import pytest
import torch

from mokit.quantization import (
    FP4_DTYPES,
    FP8_DTYPES,
    FP16_DTYPES,
    FP32_DTYPES,
    HIGH_PRECISION_DTYPES,
    REDUCED_PRECISION_DTYPES,
)


@pytest.mark.parametrize(
    "group",
    [
        FP32_DTYPES,
        FP16_DTYPES,
        FP8_DTYPES,
        FP4_DTYPES,
        HIGH_PRECISION_DTYPES,
        REDUCED_PRECISION_DTYPES,
    ],
    ids=["fp32", "fp16", "fp8", "fp4", "high_precision", "reduced_precision"],
)
def test_dtype_group_contains_only_dtypes(group: tuple[torch.dtype, ...]) -> None:
    """Test that a dtype group contains only torch dtypes."""
    assert isinstance(group, tuple), f"Expected a tuple of torch dtypes, got {type(group).__name__}"
    assert all(isinstance(dtype, torch.dtype) for dtype in group), f"Group contains non-dtype entries: {group}"
