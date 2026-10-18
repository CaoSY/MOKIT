# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization utilities."""

from typing import Final

import torch

FP32_DTYPES: Final[tuple[torch.dtype, ...]] = (torch.float32,)
FP16_DTYPES: Final[tuple[torch.dtype, ...]] = (torch.float16, torch.bfloat16)
FP8_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)
FP4_DTYPES: Final[tuple[torch.dtype, ...]] = (torch.float4_e2m1fn_x2,)

HIGH_PRECISION_DTYPES: Final[tuple[torch.dtype, ...]] = (*FP32_DTYPES, *FP16_DTYPES)
REDUCED_PRECISION_DTYPES: Final[tuple[torch.dtype, ...]] = (*FP8_DTYPES, *FP4_DTYPES)
