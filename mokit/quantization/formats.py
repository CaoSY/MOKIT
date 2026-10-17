# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization format definitions."""

from dataclasses import dataclass
from typing import Final

import torch

from mokit.core import MokitConfigBase
from mokit.quantization import FP4_DTYPES, FP8_DTYPES

MX_ELEM_DTYPES: Final[tuple[torch.dtype, ...]] = (*FP8_DTYPES, *FP4_DTYPES)
MX_SCALE_DTYPE: Final[torch.dtype] = torch.float8_e8m0fnu
MX_DEFAULT_BLOCK_SIZE: Final[int] = 32


@dataclass(frozen=True, kw_only=True, slots=True)
class MXFormat(MokitConfigBase):
    """
    An OCP Microscaling (MX) format.

    See https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf/
    """

    elem_dtype: Final[torch.dtype] = torch.float8_e4m3fn
    scale_dtype: Final[torch.dtype] = MX_SCALE_DTYPE
    block_size: Final[int] = MX_DEFAULT_BLOCK_SIZE

    def __post_init__(self) -> None:
        """Validate the field values against the MX format set."""
        if self.elem_dtype not in MX_ELEM_DTYPES:
            raise ValueError(f"Supported element dtypes are {MX_ELEM_DTYPES}, got {self.elem_dtype}")

        if self.scale_dtype is not MX_SCALE_DTYPE:
            raise ValueError(f"The scale dtype is fixed at E8M0 by the MX spec, got {self.scale_dtype}")
