# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Float quantization format definitions."""

from dataclasses import dataclass
from typing import Final

import torch

from mokit.core import MokitConfigBase
from mokit.quantization import HIGH_PRECISION_DTYPES, REDUCED_PRECISION_DTYPES
from mokit.quantization.formats import Granularity, PerTensor, QuantizationRoundingMode

DEFAULT_GRANULARITY: Final[Granularity] = PerTensor()


@dataclass(frozen=True, slots=True, kw_only=True)
class FloatFormat(MokitConfigBase):
    """
    A floating-point quantization format.

    Attributes:
        elem_dtype: Data type of the quantized elements. Must be one of ``REDUCED_PRECISION_DTYPES``.
            Defaults to ``torch.float8_e4m3fn``.

        round_mode: Rounding mode used when casting high-precision values to ``elem_dtype``. See
            ``QuantizationRoundingMode`` for the available modes. Defaults to ``QuantizationRoundingMode.RNE``.

        scale_dtype: Data type of the scale. Must be one of ``HIGH_PRECISION_DTYPES``;
            ``torch.float8_e8m0fnu`` is rejected, since that is an MX scale. Defaults to ``torch.float32``.

        granularity: Granularity of the quantization. Defaults to ``DEFAULT_GRANULARITY``.

    """

    elem_dtype: Final[torch.dtype] = torch.float8_e4m3fn
    round_mode: Final[QuantizationRoundingMode] = QuantizationRoundingMode.RNE
    scale_dtype: Final[torch.dtype] = torch.float32
    granularity: Final[Granularity] = DEFAULT_GRANULARITY

    def __post_init__(self) -> None:
        """Validate the field values against the float format sets."""
        if self.elem_dtype not in REDUCED_PRECISION_DTYPES:
            raise ValueError(f"elem_dtype must be one of {REDUCED_PRECISION_DTYPES}, got {self.elem_dtype}")

        if not isinstance(self.round_mode, QuantizationRoundingMode):
            raise ValueError(f"round_mode must be of type QuantizationRoundingMode, got {type(self.round_mode)}")

        if self.scale_dtype is torch.float8_e8m0fnu:
            raise ValueError(f"If the scale's dtype is {torch.float8_e8m0fnu}, use MXFormat instead.")
        elif self.scale_dtype not in HIGH_PRECISION_DTYPES:
            raise ValueError(f"scale_dtype only supports {HIGH_PRECISION_DTYPES}, got {self.scale_dtype}")
