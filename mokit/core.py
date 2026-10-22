# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Core utilities for MOKIT."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch._ops import OpOverload


@dataclass(frozen=True, slots=True, kw_only=True)
class MokitConfigBase:
    """The base class for all MOKIT config classes."""


class MokitWrapperTensorBase(torch.Tensor):
    """The base class for MOKIT wrapper tensor subclasses."""

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch torch functions for MOKIT tensors."""
        if kwargs is None:
            kwargs = {}

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]
        cls,
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch torch operations for MOKIT tensors."""
        if kwargs is None:
            kwargs = {}

        arg_types = tuple(type(arg) for arg in args)
        kwarg_types = {param: type(arg) for param, arg in kwargs.items()}
        raise NotImplementedError(
            f"{cls.__name__} dispatch: attempting to run unimplemented operator/function: "
            f"{func=}, {types=}, {arg_types=}, {kwarg_types=}"
        )
