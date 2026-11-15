# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Feed-forward network implementations."""

from collections.abc import Callable
from typing import ClassVar, Self

import torch
from torch.types import Device

from mokit.components.linear import GroupedLinear
from mokit.types import DType


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


class GatedFeedForward(torch.nn.Module):
    r"""
    A gated feed-forward network with a fused gate and up projection.

    This module computes
    :math:`y = W_\text{down} (\phi(W_\text{gate} x + b_\text{gate}) \odot
    (W_\text{up} x + b_\text{up})) + b_\text{down}`,
    where :math:`\phi` is the activation. The gate and up projections are
    stored as one weight of shape :math:`(2 H_\text{hidden}, H_\text{in})`, so
    a single matrix multiplication produces both halves.

    Args:
        in_features: Size of each input sample.
        hidden_features: Size of each of the two parallel projections. The
            fused projection has ``2 * hidden_features`` output features.
        out_features: Size of each output sample.
        activation: The activation module type or factory. It is called with
            no arguments, and the resulting module is applied to the gate
            half. To configure the activation, pass a factory such as
            ``functools.partial(torch.nn.SiLU, inplace=True)``.
            Default: ``torch.nn.SiLU``
        bias: A boolean sets the bias of both ``gate_up_proj`` and
            ``down_proj``; a two-element tuple sets them separately.
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
        hidden_features: Size of each of the two parallel projections.
        out_features: Size of each output sample.
        gate_up_proj: The fused projection from ``in_features`` to
            ``2 * hidden_features``. The first ``hidden_features`` outputs are
            the gate half and the remaining outputs are the up half. Its
            weight is initialized from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = 1 / \text{in\_features}`, and its bias, when present,
            is initialized in the same range.
        activation: The activation module applied to the gate half.
        down_proj: The projection from ``hidden_features`` to ``out_features``.
            Its weight is initialized from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = 1 / \text{hidden\_features}`, and its bias, when
            present, is initialized in the same range.

    Note:
        TODO: Add serialization support that concatenates ``gate_proj`` and
        ``up_proj`` into ``gate_up_proj`` when loading and splits it when
        saving.

    Examples::

        >>> m = GatedFeedForward(20, 64, 30)
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
        Initialize the gated feed-forward network.

        Args:
            in_features: Size of each input sample.
            hidden_features: Size of each of the two parallel projections.
            out_features: Size of each output sample.
            activation: The activation module type or factory.
            bias: A boolean sets the bias of both ``gate_up_proj`` and
                ``down_proj``; a two-element tuple sets them separately.
                Default: ``True``
            device: The device of the parameters.
            dtype: The dtype of the parameters.

        Raises:
            ValueError: If ``bias`` is a tuple whose length is not 2.

        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        gate_up_bias, down_bias = (bias, bias) if isinstance(bias, bool) else bias
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.gate_up_proj = torch.nn.Linear(in_features, 2 * hidden_features, bias=gate_up_bias, **factory_kwargs)
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
        gate, up = self.gate_up_proj(input).tensor_split(2, dim=-1)
        return self.down_proj(self.activation(gate) * up)

    def extra_repr(self: Self) -> str:
        """
        Return the extra representation of the module.

        Returns:
            A string describing the module configuration.

        """
        return (
            f"in_features={self.in_features}, hidden_features={self.hidden_features}, out_features={self.out_features}"
        )


class GroupedFeedForward(torch.nn.Module):
    r"""
    A feed-forward network with a separate weight per group.

    This module computes
    :math:`y_g = W_\text{down}^{(g)} \phi(W_\text{up}^{(g)} x_g + b_\text{up}^{(g)}) + b_\text{down}^{(g)}`
    for each group :math:`g`, where :math:`\phi` is the activation.

    Args:
        num_groups: Number of groups (independent expert weights).
        in_features: Size of each input sample.
        hidden_features: Size of each group's hidden dimension.
        out_features: Size of each output sample.
        activation: The activation module type or factory. It is called with
            no arguments, and the resulting module is applied to the hidden
            projection. It does not receive ``offs``, so it cannot skip rows
            past ``offs[-1]``. To configure the activation, pass a factory
            such as ``functools.partial(torch.nn.SiLU, inplace=True)``.
            Default: ``torch.nn.SiLU``
        bias: A boolean sets the bias of both projections; a two-element tuple
            sets the bias of ``up_proj`` and ``down_proj`` separately.
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
          :math:`H_\text{in} = \text{in\_features}` and
          :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        num_groups: Number of groups.
        in_features: Size of each input sample.
        hidden_features: Size of each group's hidden dimension.
        out_features: Size of each output sample.
        up_proj: The ``GroupedLinear`` projection from ``in_features`` to
            ``hidden_features``.
        activation: The activation module applied to the hidden projection.
            It does not receive ``offs``.
        down_proj: The ``GroupedLinear`` projection from ``hidden_features``
            to ``out_features``.

    Note:
        For 2D input, rows past ``offs[-1]`` are left uninitialized. The
        activation is applied to the entire tensor produced by ``up_proj``,
        including those tail rows, because it does not receive ``offs`` and
        therefore cannot skip them. Callers must mask or discard the tail rows.

    Warning:
        The grouped GEMM kernels used by the two projections have alignment
        and stride requirements. Since the intermediate tensor produced by
        ``up_proj`` is passed directly to ``down_proj``, choose the dimension
        sizes so that both projections satisfy those requirements, and consult
        the kernel's documentation for the exact constraints.

    Examples::

        >>> m = GroupedFeedForward(3, 20, 64, 30)
        >>> input = torch.randn(10, 20)
        >>> offs = torch.tensor([2, 5, 10], dtype=torch.int32)
        >>> output = m(input, offs=offs)
        >>> print(output.size())
        torch.Size([10, 30])

    """

    __constants__: ClassVar[list[str]] = ["num_groups", "in_features", "hidden_features", "out_features"]

    def __init__(
        self: Self,
        num_groups: int,
        in_features: int,
        hidden_features: int,
        out_features: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.SiLU,
        bias: bool | tuple[bool, bool] = True,
        device: Device = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the grouped feed-forward network.

        Args:
            num_groups: Number of groups (independent expert weights).
            in_features: Size of each input sample.
            hidden_features: Size of each group's hidden dimension.
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
        self.num_groups = num_groups
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.up_proj = GroupedLinear(num_groups, in_features, hidden_features, bias=up_bias, **factory_kwargs)
        self.activation = activation()
        self.down_proj = GroupedLinear(num_groups, hidden_features, out_features, bias=down_bias, **factory_kwargs)

    def forward(
        self: Self,
        input: torch.Tensor,
        *,
        offs: torch.Tensor | None = None,
        out_dtype: DType | tuple[DType, DType] = None,
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
            out_dtype: Optional output dtype for the projections. A scalar or
                ``None`` applies to both; a two-element tuple sets ``up_proj``
                and ``down_proj`` separately. ``None`` follows each
                projection's input dtype.

        Returns:
            The output tensor of shape ``(M, out_features)`` for 2D input, or
            ``(num_groups, M_g, out_features)`` for 3D input. The output dtype
            is the dtype requested for ``down_proj`` when provided; otherwise
            it is the dtype produced by the activation. For 2D input, rows
            past ``offs[-1]`` are left uninitialized; callers must mask or
            discard them.

        """
        up_out_dtype, down_out_dtype = out_dtype if isinstance(out_dtype, tuple) else (out_dtype, out_dtype)
        hidden = self.activation(self.up_proj(input, offs=offs, out_dtype=up_out_dtype))
        return self.down_proj(hidden, offs=offs, out_dtype=down_out_dtype)

    def extra_repr(self: Self) -> str:
        """
        Return the extra representation of the module.

        Returns:
            A string describing the module configuration.

        """
        return (
            f"num_groups={self.num_groups}, in_features={self.in_features}, "
            f"hidden_features={self.hidden_features}, out_features={self.out_features}"
        )


class GatedGroupedFeedForward(torch.nn.Module):
    r"""
    A gated feed-forward network with a separate fused weight per group.

    This module computes
    :math:`y_g = W_\text{down}^{(g)} (\phi(W_\text{gate}^{(g)} x_g + b_\text{gate}^{(g)}) \odot
    (W_\text{up}^{(g)} x_g + b_\text{up}^{(g)})) + b_\text{down}^{(g)}`
    for each group :math:`g`, where :math:`\phi` is the activation. The gate
    and up projections are stored as one weight per group, so a single grouped
    matrix multiplication produces both halves.

    Args:
        num_groups: Number of groups (independent expert weights).
        in_features: Size of each input sample.
        hidden_features: Size of each of the two parallel projections per
            group. The fused projection has ``2 * hidden_features`` output
            features.
        out_features: Size of each output sample.
        activation: The activation module type or factory. It is called with
            no arguments, and the resulting module is applied to the gate half.
            It does not receive ``offs``, so it cannot skip rows past
            ``offs[-1]``. To configure the activation, pass a factory such as
            ``functools.partial(torch.nn.SiLU, inplace=True)``.
            Default: ``torch.nn.SiLU``
        bias: A boolean sets the bias of both ``gate_up_proj`` and
            ``down_proj``; a two-element tuple sets them separately.
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
          :math:`H_\text{in} = \text{in\_features}` and
          :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        num_groups: Number of groups.
        in_features: Size of each input sample.
        hidden_features: Size of each of the two parallel projections.
        out_features: Size of each output sample.
        gate_up_proj: The ``GroupedLinear`` projection from ``in_features`` to
            ``2 * hidden_features`` per group. The first ``hidden_features``
            outputs of each group are the gate half and the remaining outputs
            are the up half.
        activation: The activation module applied to the gate half. It does
            not receive ``offs``.
        down_proj: The ``GroupedLinear`` projection from ``hidden_features``
            to ``out_features``.

    Note:
        For 2D input, rows past ``offs[-1]`` are left uninitialized. The
        activation is applied to the gate half of the tensor produced by
        ``gate_up_proj``, including those tail rows, because it does not
        receive ``offs`` and therefore cannot skip them. Callers must mask or
        discard the tail rows.

    Warning:
        The grouped GEMM kernels used by the two projections have alignment
        and stride requirements. Since the intermediate tensor produced by
        ``gate_up_proj`` is passed directly to ``down_proj``, choose the
        dimension sizes so that both projections satisfy those requirements,
        and consult the kernel's documentation for the exact constraints.

    Examples::

        >>> m = GatedGroupedFeedForward(3, 20, 64, 30)
        >>> input = torch.randn(10, 20)
        >>> offs = torch.tensor([2, 5, 10], dtype=torch.int32)
        >>> output = m(input, offs=offs)
        >>> print(output.size())
        torch.Size([10, 30])

    """

    __constants__: ClassVar[list[str]] = ["num_groups", "in_features", "hidden_features", "out_features"]

    def __init__(
        self: Self,
        num_groups: int,
        in_features: int,
        hidden_features: int,
        out_features: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.SiLU,
        bias: bool | tuple[bool, bool] = True,
        device: Device = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the gated grouped feed-forward network.

        Args:
            num_groups: Number of groups (independent expert weights).
            in_features: Size of each input sample.
            hidden_features: Size of each of the two parallel projections.
            out_features: Size of each output sample.
            activation: The activation module type or factory.
            bias: A boolean sets the bias of both ``gate_up_proj`` and
                ``down_proj``; a two-element tuple sets them separately.
                Default: ``True``
            device: The device of the parameters.
            dtype: The dtype of the parameters.

        Raises:
            ValueError: If ``bias`` is a tuple whose length is not 2.

        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        gate_up_bias, down_bias = (bias, bias) if isinstance(bias, bool) else bias
        self.num_groups = num_groups
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.gate_up_proj = GroupedLinear(num_groups, in_features, 2 * hidden_features, gate_up_bias, **factory_kwargs)
        self.activation = activation()
        self.down_proj = GroupedLinear(num_groups, hidden_features, out_features, bias=down_bias, **factory_kwargs)

    def forward(
        self: Self,
        input: torch.Tensor,
        *,
        offs: torch.Tensor | None = None,
        out_dtype: DType | tuple[DType, DType] = None,
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
            out_dtype: Optional output dtype for the projections. A scalar or
                ``None`` applies to both; a two-element tuple sets
                ``gate_up_proj`` and ``down_proj`` separately. ``None``
                follows each projection's input dtype.

        Returns:
            The output tensor of shape ``(M, out_features)`` for 2D input, or
            ``(num_groups, M_g, out_features)`` for 3D input. The output dtype
            is the dtype requested for ``down_proj`` when provided; otherwise
            it is the dtype produced by the activation. For 2D input, rows
            past ``offs[-1]`` are left uninitialized; callers must mask or
            discard them.

        """
        gate_up_out_dtype, down_out_dtype = out_dtype if isinstance(out_dtype, tuple) else (out_dtype, out_dtype)
        gate, up = self.gate_up_proj(input, offs=offs, out_dtype=gate_up_out_dtype).tensor_split(2, dim=-1)
        return self.down_proj(self.activation(gate) * up, offs=offs, out_dtype=down_out_dtype)

    def extra_repr(self: Self) -> str:
        """
        Return the extra representation of the module.

        Returns:
            A string describing the module configuration.

        """
        return (
            f"num_groups={self.num_groups}, in_features={self.in_features}, "
            f"hidden_features={self.hidden_features}, out_features={self.out_features}"
        )
