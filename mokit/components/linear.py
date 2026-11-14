# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Linear layer implementations."""

import math
from typing import ClassVar, Self, cast

import torch
import torch.nn.functional as F
from torch.nn import init
from torch.types import Device


class BatchedLinear(torch.nn.Module):
    r"""
    Apply an affine linear transformation with a separate weight per batch.

    This module computes :math:`y_b = x_b A_b^T + b_b` for each batch element
    :math:`b`. The weight and bias are independent for each batch element.

    Args:
        batches: Number of independent weight matrices, must be > 0.
        in_features: Size of each input sample, must be > 0.
        out_features: Size of each output sample, must be > 0.
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
            batches: Number of independent weight matrices, must be > 0.
            in_features: Size of each input sample, must be > 0.
            out_features: Size of each output sample, must be > 0.
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

        Raises:
            ValueError: If ``in_features`` is not positive.

        """
        if self.in_features <= 0:
            raise ValueError(f"in_features must be > 0, but got (in_features={self.in_features})")

        bound = 1 / math.sqrt(self.in_features)
        init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
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


class GroupedLinear(torch.nn.Module):
    r"""
    An affine linear layer with a separate weight per group.

    This module computes :math:`y_g = x_g A_g^T + b_g` for each group
    :math:`g`. It is the linear primitive for Mixture-of-Experts layers, where
    each group is an expert and tokens are sorted into per-expert blocks.

    The grouped GEMM operands are passed directly to
    :func:`torch.nn.functional.grouped_mm` and must satisfy its requirements.

    Args:
        num_groups: Number of groups (independent weight matrices).
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        device: The device of the parameters. Default: ``None``
        dtype: The dtype of the parameters. Default: ``None``

    Shape:
        - Input, 2D: :math:`(M, H_\text{in})` together with ``offs`` of shape
          :math:`(G,)`, where :math:`M` is the total number of rows and
          :math:`G = \text{num\_groups}`. Group ``g`` owns rows
          ``[offs[g - 1], offs[g])``, with the lower bound of group 0 equal to
          0.
        - Input, 3D: :math:`(G, M_g, H_\text{in})`, where every group has the
          same number of rows and ``offs`` must be ``None``.
        - Output: :math:`(M, H_\text{out})` for 2D input, or
          :math:`(G, M_g, H_\text{out})` for 3D input, where
          :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: The learnable weights, stored as a flattened 2D tensor of
            shape :math:`(G \cdot H_\text{out}, H_\text{in})`; FSDP2 therefore
            sees ``dim0`` as ``num_groups * out_features``. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{H_\text{in}}`. When
            :math:`H_\text{in} = 0`, the bound is zero.
        bias: The learnable bias, stored as a flattened 1D tensor of shape
            :math:`(G \cdot H_\text{out})`, whose ``dim0`` is also
            ``num_groups * out_features``. It is initialized in the same range
            as the weight, or is ``None`` when ``bias=False``.

    Examples::

        >>> m = GroupedLinear(3, 20, 30)
        >>> input = torch.randn(10, 20)
        >>> offs = torch.tensor([2, 5, 10], dtype=torch.int32)
        >>> output = m(input, offs=offs)
        >>> print(output.size())
        torch.Size([10, 30])

    """

    __constants__: ClassVar[list[str]] = ["num_groups", "in_features", "out_features"]

    num_groups: int
    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self: Self,
        num_groups: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Device = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the grouped linear layer.

        Args:
            num_groups: Number of groups (independent weight matrices).
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If ``False``, the layer will not learn an additive bias.
            device: The device of the parameters.
            dtype: The dtype of the parameters.

        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((num_groups * out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(num_groups * out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self: Self) -> None:
        r"""
        Reset the learnable parameters.

        The weight, and the bias when present, are initialized from a uniform
        distribution over :math:`[-\sqrt{k}, \sqrt{k}]`, where
        :math:`k = 1 / H_\text{in}`. If :math:`H_\text{in} = 0`, the bound is
        zero.

        """
        bound = 0.0 if self.in_features == 0 else 1 / math.sqrt(self.in_features)
        init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(
        self: Self,
        input: torch.Tensor,
        *,
        offs: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Run the forward pass.

        Args:
            input: The input tensor. A 2D input of shape
                ``(M, in_features)`` requires ``offs``. A 3D input of shape
                ``(num_groups, M_g, in_features)`` requires ``offs`` to be
                ``None``.
            offs: Optional offsets of shape ``(num_groups,)``. Required for
                2D input and forbidden for 3D input.
            out_dtype: Optional output dtype. When provided, the returned
                tensor has this dtype; otherwise it has ``input.dtype``.

        The grouped GEMM operands are passed directly to
        :func:`torch.nn.functional.grouped_mm` and must satisfy its
        requirements.

        Returns:
            The output tensor of shape ``(M, out_features)`` for 2D input, or
            ``(num_groups, M_g, out_features)`` for 3D input. Its dtype is
            ``out_dtype`` when provided, otherwise ``input.dtype``. For 2D
            input, rows past ``offs[-1]`` are left uninitialized; callers must
            mask or discard them.

        Note:
            TODO: When ``grouped_mm`` supports ``bias`` with autograd, pass
            the per-group bias directly to it and remove the manual addition.

        """
        weight_t = torch.reshape(
            self.weight,
            (self.num_groups, self.out_features, self.in_features),
        ).transpose(-2, -1)

        out = F.grouped_mm(input, weight_t, offs=offs, out_dtype=out_dtype)

        if self.bias is not None:
            bias = torch.reshape(self.bias, (self.num_groups, self.out_features))

            # In-place is safe: grouped_mm's backward does not save its output.
            if input.ndim == 3:
                out.add_(bias[:, None, :])
            else:
                # grouped_mm already rejected a 2D input without offs.
                # We cast below only to pass pyrefly check.
                offs = cast(torch.Tensor, offs)
                row_group = torch.bucketize(
                    torch.arange(input.shape[0], device=bias.device),
                    offs,
                    # True today because grouped_mm requires int32 offs; kept
                    # in sync if int64 offsets are supported later.
                    out_int32=(offs.dtype is torch.int32),
                    right=True,
                )
                # The sentinel row is only selected by rows past offs[-1],
                # whose outputs are unspecified, so its values do not matter.
                bias = torch.cat([bias, bias.new_empty(1, self.out_features)])
                out.add_(bias[row_group])

        return out

    def extra_repr(self: Self) -> str:
        """
        Return the extra representation of the module.

        Returns:
            A string describing the module configuration.

        """
        return (
            f"num_groups={self.num_groups}, in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )
