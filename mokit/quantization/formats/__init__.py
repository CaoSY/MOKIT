# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization format definitions."""

from enum import Enum


class QuantizationRoundingMode(Enum):
    """
    Rounding modes for casting high-precision values to reduced-precision values.

    Attributes:
        RNE: Round to nearest, ties to even.

        STOCHASTIC: Round up or down at random, with probability proportional to the distance to each neighbour.

    """

    RNE = "round-to-nearest-even"
    STOCHASTIC = "stochastic"


from mokit.quantization.formats.mx_formats import (  # noqa: E402
    MXFormat,
    MXScaleCalculationMode,
)

__all__ = [
    "MXFormat",
    "MXScaleCalculationMode",
    "QuantizationRoundingMode",
]
