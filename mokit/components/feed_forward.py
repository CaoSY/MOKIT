# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Feed-forward network implementations."""

from collections.abc import Callable
from typing import ClassVar, Self

import torch
from torch.types import Device


class FeedForward(torch.nn.Module):
    r"""
    A feed-forward network with an activation between two linear projections.

    This module computes
    :math:`y = W_\text{down} \phi(W_\text{up} x + b_\text{up}) + b_\text{down}`,
    where :math:`\phi` is the activation.

    Args:
        in_features: Size of each input sample.
        hidden_features: Size of the hidden dimension.
        out_features: Size of each output sample.
        activation: The activation module type or factory. It is called with
            no arguments, and the resulting module is applied between the two
            projections. To configure the activation, pass a factory such as
            ``functools.partial(torch.nn.SiLU, inplace=True)``.
            Default: ``torch.nn.SiLU``
        bias: A boolean sets the bias of both projections; a two-element tuple
            sets the bias of ``up_proj`` and ``down_proj`` separately.
            Default: ``True``
        device: The device of the parameters. Default: ``None``
        dtype: The dtype of the parameters. Default: ``None``

    Shape:
        - Input: :math:`(*, H_\text{in})`, where :math:`H_\text{in} =
          \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})`, where all dimensions except the
          last match the input and :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        in_features: Size of each input sample.
        hidden_features: Size of the hidden dimension.
        out_features: Size of each output sample.
        up_proj: The projection from ``in_features`` to ``hidden_features``.
            Its weight is initialized from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = 1 / \text{in\_features}`, and its bias, when present,
            is initialized in the same range.
        activation: The activation module applied between the projections.
        down_proj: The projection from ``hidden_features`` to ``out_features``.
            Its weight is initialized from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = 1 / \text{hidden\_features}`, and its bias, when
            present, is initialized in the same range.

    Examples::

        >>> m = FeedForward(20, 64, 30)
        >>> input = torch.randn(8, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([8, 30])

    """

    __constants__: ClassVar[list[str]] = ["in_features", "hidden_features", "out_features"]

    def __init__(
        self: Self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.SiLU,
        bias: bool | tuple[bool, bool] = True,
        device: Device = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the feed-forward network.

        Args:
            in_features: Size of each input sample.
            hidden_features: Size of the hidden dimension.
            out_features: Size of each output sample.
            activation: The activation module type or factory.
            bias: A boolean sets the bias of both projections; a two-element
                tuple sets the bias of ``up_proj`` and ``down_proj``
                separately. Default: ``True``
            device: The device of the parameters.
            dtype: The dtype of the parameters.

        Raises:
            ValueError: If ``bias`` is a tuple whose length is not 2.

        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        up_bias, down_bias = (bias, bias) if isinstance(bias, bool) else bias
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.up_proj = torch.nn.Linear(in_features, hidden_features, bias=up_bias, **factory_kwargs)
        self.activation = activation()
        self.down_proj = torch.nn.Linear(hidden_features, out_features, bias=down_bias, **factory_kwargs)

    def forward(self: Self, input: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Args:
            input: The input tensor of shape ``(..., in_features)``.

        Returns:
            The output tensor of shape ``(..., out_features)``.

        """
        return self.down_proj(self.activation(self.up_proj(input)))

    def extra_repr(self: Self) -> str:
        """
        Return the extra representation of the module.

        Returns:
            A string describing the module configuration.

        """
        return (
            f"in_features={self.in_features}, hidden_features={self.hidden_features}, out_features={self.out_features}"
        )
