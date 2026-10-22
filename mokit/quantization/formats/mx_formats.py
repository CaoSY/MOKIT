# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""OCP Microscaling (MX) format definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import Final

import torch

from mokit.core import MokitConfigBase
from mokit.quantization import FP4_DTYPES, FP8_DTYPES
from mokit.quantization.formats import PerBlock, QuantizationRoundingMode
from mokit.serialization import torch_add_safe_global

MX_ELEM_DTYPES: Final[tuple[torch.dtype, ...]] = (*FP8_DTYPES, *FP4_DTYPES, torch.int8)
MX_SCALE_DTYPE: Final[torch.dtype] = torch.float8_e8m0fnu
MX_DEFAULT_BLOCK_SIZE: Final[int] = 32
MX_DEFAULT_GRANULARITY: Final[PerBlock] = PerBlock(MX_DEFAULT_BLOCK_SIZE)


@torch_add_safe_global
class MXScaleCalculationMode(Enum):
    r"""
    Methods for calculating MX block scaling.

    Attributes:
        FLOOR: Recommended by `OCP Microscaling Formats (MX) Specification v1.0
            <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf/>`_.
            Uses :math:`X = 2^{\lfloor \log_2(\max|v|) - \text{max\_exp} \rfloor}`.
            Overflows for large values and performs poorly for gradient quantization.

        RCEIL: Applies ceil to the ratio of ``max_abs(v)`` and ``Amax(DType)``. See "Computing scaling and conversion
            factors for FP8 with UE8M0 scales" in the `NVIDIA cuBLAS block quantization documentation
            <https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization>`_.

        CEIL: Uses :math:`X = 2^{\lceil \log_2(\max|v|) - \text{max\_exp} \rceil}`.
            Avoids overflow, but small values may shift to 0 because of the large scaling factor.

        EVEN: Trade-off between FLOOR and CEIL, using
            :math:`X = 2^{\lfloor \log_2(\operatorname{round}(\max|v|)) - \text{max\_exp} \rfloor}`.

    """

    FLOOR = "floor"
    RCEIL = "rceil"
    CEIL = "ceil"
    EVEN = "even"


@torch_add_safe_global
@dataclass(frozen=True, kw_only=True, slots=True)
class MXFormat(MokitConfigBase):
    """
    An OCP Microscaling (MX) format.

    See the `OCP Microscaling Formats (MX) Specification v1.0
    <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf/>`_.

    This class describes the format only; conversion settings live in :class:`MXQuantConfig`.

    Attributes:
        elem_dtype: Data type of the quantized elements. Must be one of ``MX_ELEM_DTYPES``. Defaults to
            ``torch.float8_e4m3fn``.

        scale_dtype: Data type of the per-block scale. Fixed at E8M0 by the MX spec. Defaults to ``MX_SCALE_DTYPE``.

        granularity: Granularity of the quantization blocks. Must be a ``PerBlock`` with positive
            per-dimension sizes. Defaults to ``MX_DEFAULT_GRANULARITY``.

    """

    elem_dtype: Final[torch.dtype] = torch.float8_e4m3fn

    scale_dtype: Final[torch.dtype] = MX_SCALE_DTYPE

    granularity: PerBlock = MX_DEFAULT_GRANULARITY

    def __post_init__(self) -> None:
        """Validate the format fields."""
        if self.elem_dtype not in MX_ELEM_DTYPES:
            raise ValueError(f"Supported element dtypes are {MX_ELEM_DTYPES}, got {self.elem_dtype}")

        if self.scale_dtype is not MX_SCALE_DTYPE:
            raise ValueError(f"The scale dtype is fixed at E8M0 by the MX spec, got {self.scale_dtype}")

        if not isinstance(self.granularity, PerBlock):
            raise ValueError(f"MXFormat.granularity must be of type PerBlock, got {type(self.granularity)}")
        elif not self.granularity or any(size <= 0 for size in self.granularity):
            raise ValueError(f"Each dim of a quantization block must be positive, got {self.granularity}")


@torch_add_safe_global
@dataclass(frozen=True, slots=True, kw_only=True)
class MXQuantConfig(MXFormat):
    """
    Conversion settings for an MX format.

    The format fields are inherited from :class:`MXFormat`.

    Attributes:
        elem_dtype_max: Reference maximum used when calculating scales. A ``float`` for floating-point
            dtypes and an ``int`` for integer dtypes. Defaults to ``None``, in which case it is derived
            from ``elem_dtype``.

        rounding_mode: Rounding mode used when casting high-precision values to ``elem_dtype``. See
            ``QuantizationRoundingMode`` for the available modes. Defaults to ``QuantizationRoundingMode.RNE``.

        scale_mode: Method used to calculate the per-block scale. See ``MXScaleCalculationMode`` for the
            available methods. Defaults to ``MXScaleCalculationMode.RCEIL``.

    """

    elem_dtype_max: Final[float | int | None] = None

    rounding_mode: Final[QuantizationRoundingMode] = QuantizationRoundingMode.RNE

    scale_mode: Final[MXScaleCalculationMode] = MXScaleCalculationMode.RCEIL

    def __post_init__(self) -> None:
        """Validate the conversion settings and derive ``elem_dtype_max`` if unset."""
        # Not super(): a zero-arg super here leaves the generated frozen __setattr__ bound to the
        # pre-clone class, so assignment raises TypeError instead of FrozenInstanceError.
        MXFormat.__post_init__(self)

        if not isinstance(self.rounding_mode, QuantizationRoundingMode):
            raise ValueError(f"rounding_mode must be a QuantizationRoundingMode, got {type(self.rounding_mode)}")

        if self.elem_dtype_max is None:
            if self.elem_dtype is torch.float4_e2m1fn_x2:
                dtype_max = 6.0
            elif self.elem_dtype.is_floating_point:
                dtype_max = torch.finfo(self.elem_dtype).max
            else:
                dtype_max = torch.iinfo(self.elem_dtype).max
            object.__setattr__(self, "elem_dtype_max", dtype_max)
        elif self.elem_dtype_max <= 0:
            raise ValueError(f"elem_dtype_max must be positive, got {self.elem_dtype_max}")

        if not isinstance(self.scale_mode, MXScaleCalculationMode):
            raise ValueError(f"scale_mode must be a MXScaleCalculationMode, got {type(self.scale_mode)}")


BLOCK_MX_DEFAULT_GRANULARITY: Final[PerBlock] = PerBlock(MX_DEFAULT_BLOCK_SIZE, MX_DEFAULT_BLOCK_SIZE)


@torch_add_safe_global
@dataclass(frozen=True, slots=True, kw_only=True)
class BlockMXFormat(MXFormat):
    """
    An MX format restricted to square, two-dimensional quantization blocks.

    Requires ``granularity`` to have exactly two equal dimensions, e.g. ``PerBlock(32, 32)``.
    Defaults to ``BLOCK_MX_DEFAULT_GRANULARITY``.

    The other attributes are inherited from :class:`MXFormat`, where they are documented.
    """

    granularity: Final[PerBlock] = BLOCK_MX_DEFAULT_GRANULARITY

    def __post_init__(self) -> None:
        """Validate the square block constraint on top of the base checks."""
        # Not super(): a zero-arg super here leaves the generated frozen __setattr__ bound to the
        # pre-clone class, so assignment raises TypeError instead of FrozenInstanceError.
        MXFormat.__post_init__(self)

        _validate_square_block(self.granularity)


def _validate_square_block(granularity: PerBlock) -> None:
    """Validate that a granularity is a two-dimensional square block."""
    if len(granularity) != 2 or granularity[0] != granularity[1]:
        raise ValueError(f"BlockMXFormat only supports a square quantization block, got {granularity}")


@torch_add_safe_global
@dataclass(frozen=True, slots=True, kw_only=True)
class BlockMXQuantConfig(BlockMXFormat, MXQuantConfig):
    """
    Conversion settings for an MX format with square, two-dimensional blocks.

    Combines the square-block constraint from :class:`BlockMXFormat` with the conversion
    settings from :class:`MXQuantConfig`. The default block size comes from
    :class:`BlockMXFormat`.
    """

    def __post_init__(self) -> None:
        """Validate the conversion settings and the square block constraint."""
        # BlockMXFormat is first in the MRO so its BLOCK_MX_DEFAULT_GRANULARITY default wins,
        # but its __post_init__ would skip MXQuantConfig's checks. Validate the conversion
        # settings once, then validate the block shape.
        MXQuantConfig.__post_init__(self)

        _validate_square_block(self.granularity)
