# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization format definitions."""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Final, overload

from mokit.core import MokitConfigBase


class QuantizationRoundingMode(Enum):
    """
    Rounding modes for casting high-precision values to reduced-precision values.

    Attributes:
        RNE: Round to nearest, ties to even.

        STOCHASTIC: Round up or down at random, with probability proportional to the distance to each neighbour.

    """

    RNE = "round-to-nearest-even"
    STOCHASTIC = "stochastic"


@dataclass(frozen=True, slots=True, kw_only=True)
class Granularity(MokitConfigBase):
    """Base class for representing the granularity of quantization."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PerTensor(Granularity):
    """
    Per-tensor quantization granularity.

    Calculates the quantization parameter based on the entire tensor.
    """


@dataclass(frozen=True, slots=True)
class PerSlice(Granularity):
    """
    Per-slice quantization granularity.

    Calculates different quantization parameters using slices indexed by ``axis``.

    Examples:
        * input_tensor shape [M, N], axis 0 -> scale_shape [M, 1]
        * input_tensor shape [M, N], axis 1 -> scale_shape [1, N]
        * input_tensor shape [M, K, N], axis 1 -> scale_shape [1, K, 1]

    Attributes:
        axis: The axis indexing the slices, on which reduction is performed.

    """

    axis: Final[int]


@dataclass(frozen=True, slots=True)
class PerDim(Granularity):
    """
    Dim-wise quantization granularity.

    Calculates different quantization parameters using values in ``dim``.

    Examples:
        * input_tensor shape [M, N], dim 0 -> scale_shape [1, N]
        * input_tensor shape [M, N], dim 1 -> scale_shape [M, 1]
        * input_tensor shape [M, K, N], dim 1 -> scale_shape [M, 1, N]

    Attributes:
        dim: The dim which is reduced across, all other dims are kept.

    """

    dim: Final[int]


@dataclass(frozen=True, slots=True, init=False)
class PerBlock(Granularity):
    """
    (Multidimensional) per-block quantization granularity.

    Calculates different quantization parameters based on tiles given by ``block_size``.
    Iterates and indexes over ``block_size``, one value per dimension. Slicing returns a new
    ``PerBlock``.

    Examples:
        Below we assume reduction are always performed on trailing dimensions.

        * block_size has shape [X]
            - input_tensor shape [A] -> [A // X]
            - input_tensor shape [A, B] -> [A, B // X]
            - input_tensor shape [A, B, C] -> [A, B, C // X]
            - and so on
        * block_size has shape [X, Y]
            - input_tensor shape [A] -> scaling undefined
            - input_tensor shape [A, B] -> scale shape [A // X, B // Y]
            - input_tensor shape [A, B, C] -> scale shape [A, B // X, C // Y]
            - input_tensor shape [A, B, C, D] -> scale shape [A, B, C // X, D // Y]
            - and so on
        * and so on

    Attributes:
        block_size: The size of each quantization block, one value per dimension.

    """

    block_size: Final[tuple[int, ...]]

    def __init__(self, *block_size: int) -> None:
        """Set the block size from one value per dimension."""
        if not all(isinstance(x, int) for x in block_size):
            raise TypeError(f"block_size must be ints, e.g. PerBlock(32, 32), got {block_size}")
        object.__setattr__(self, "block_size", block_size)

    def __iter__(self) -> Iterator[int]:
        """Iterate over the block size of each dimension."""
        return iter(self.block_size)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> PerBlock: ...

    def __getitem__(self, index: int | slice) -> int | PerBlock:
        """Return the block size of a dimension, or a new ``PerBlock`` for a slice."""
        if isinstance(index, slice):
            return PerBlock(*self.block_size[index])
        return self.block_size[index]

    def __len__(self) -> int:
        """Return the number of dimensions."""
        return len(self.block_size)


from mokit.quantization.formats.mx_formats import (  # noqa: E402
    BlockMXFormat,
    MXFormat,
    MXScaleCalculationMode,
)

__all__ = [
    "BlockMXFormat",
    "Granularity",
    "MXFormat",
    "MXScaleCalculationMode",
    "PerBlock",
    "PerDim",
    "PerSlice",
    "PerTensor",
    "QuantizationRoundingMode",
]
