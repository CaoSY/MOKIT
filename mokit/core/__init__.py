# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Core utilities for MOKIT."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class MokitConfigBase:
    """The base class for all MOKIT config classes."""


class MokitTensorBase(torch.Tensor):
    """The base class for all MOKIT tensor classes."""


from mokit.core.wrapper_tensor_base import MokitWrapperTensorBase  # noqa: E402

__all__ = ["MokitConfigBase", "MokitTensorBase", "MokitWrapperTensorBase"]
