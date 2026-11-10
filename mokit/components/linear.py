# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Linear layer implementations."""

import math
from typing import ClassVar, Self

import torch
from torch.nn import init
from torch.types import Device


class BatchedLinear(torch.nn.Module):
    r"""
    Apply an affine linear transformation with a separate weight per batch.

    This module computes :math:`y_b = x_b A_b^T + b_b` for each batch element
    :math:`b`. The weight and bias are independent for each batch element.

    Args:
        batches: Number of independent weight matrices.
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        device: The device of the parameters. Default: ``None``
        dtype: The dtype of the parameters. Default: ``None``

    Shape:
        - Input: :math:`(B, *, H_\text{in})`, where :math:`B` must equal
          ``batches`` and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(B, *, H_\text{out})`, where all dimensions except
          the batch and last dimensions match the input and
          :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: The learnable weights of the module. They are stored as a
            flattened tensor of shape
            :math:`(\text{batches} \cdot \text{out\_features}, \text{in\_features})`
            so that initialization follows ``torch.nn.Linear``. The values
            are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`,
            where :math:`k = \frac{1}{\text{in\_features}}`.
        bias: The learnable bias of the module. It is stored as a flattened
            tensor of shape
            :math:`(\text{batches} \cdot \text{out\_features})`. If
            :attr:`bias` is ``True``, the values are initialized from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`.

    Note:
        The ``weight`` and ``bias`` parameters are stored in flattened form.
        ``weight`` has shape
        :math:`(\text{batches} \cdot \text{out\_features}, \text{in\_features})`
        and ``bias`` has shape
        :math:`(\text{batches} \cdot \text{out\_features})`. When FSDP2
        shards these parameters, it therefore sees a 2D weight and a 1D bias
        whose ``dim0`` is ``batches * out_features``, rather than the
        conceptual 3D weight and 2D bias. This requires a little extra care
        when inspecting or sharding the parameters, but it provides more
        flexibility when applying ``dim0`` sharding, because a shard can
        split the flattened ``(batches, out_features)`` axis at any row
        boundary, not only at batch boundaries.

    Examples::

        >>> m = BatchedLinear(4, 20, 30)
        >>> input = torch.randn(4, 128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([4, 128, 30])

    """

    __constants__: ClassVar[list[str]] = ["batches", "in_features", "out_features"]
    batches: int
    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self: Self,
        batches: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Device = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the batched linear layer.

        Args:
            batches: Number of independent weight matrices.
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If ``False``, the layer will not learn an additive bias.
            device: The device of the parameters.
            dtype: The dtype of the parameters.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.batches = batches
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((batches * out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(batches * out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self: Self) -> None:
        """
        Reset parameters based on their initialization used in ``__init__``.

        The weight initialization matches ``torch.nn.Linear`` for each batch element.
        """
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self: Self, input: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Args:
            input: The input tensor of shape ``(batches, ..., in_features)``.

        Returns:
            The output tensor of shape ``(batches, ..., out_features)``.

        Raises:
            ValueError: If ``input`` has fewer than two dimensions or its
                first dimension is not ``batches``.

        """
        if input.ndim < 2 or input.shape[0] != self.batches:
            raise ValueError(
                f"BatchedLinear expects input of shape (batches, ..., in_features) "
                f"with batches={self.batches}, got {tuple(input.shape)}"
            )

        flattened_input = torch.reshape(input, (self.batches, -1, input.size(-1)))
        weight_t = torch.reshape(
            self.weight,
            (self.batches, self.out_features, self.in_features),
        ).transpose(-2, -1)

        if self.bias is None:
            flattened_output = torch.bmm(flattened_input, weight_t)
        else:
            bias = torch.reshape(self.bias, (self.batches, self.out_features)).unsqueeze(1)
            flattened_output = torch.baddbmm(bias, flattened_input, weight_t)
        return torch.reshape(flattened_output, (*input.shape[:-1], self.out_features))

    def extra_repr(self: Self) -> str:
        """
        Return the extra representation of the module.

        Returns:
            A string describing the module configuration.

        """
        return (
            f"batches={self.batches}, in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )
