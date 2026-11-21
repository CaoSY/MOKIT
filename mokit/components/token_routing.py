# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Token routing components."""

from collections.abc import Callable
from typing import ClassVar, Self, cast

import torch
import torch.nn.functional as F
from torch.types import Device

from mokit.types import DType


class FP32SoftmaxScore(torch.nn.Module):
    """A softmax score module that computes in float32."""

    def forward(self: Self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply a softmax over the last dimension in float32.

        Args:
            input: The input tensor.

        Returns:
            The softmax scores in float32.

        """
        return F.softmax(input, dim=-1, dtype=torch.float32)


class TopKExpertSelector(torch.nn.Module):
    r"""
    Select the top-k experts from a score matrix.

    The selector operates on scores, not logits; scoring and normalization are
    handled by the caller. When a selection bias is enabled, it is added only to
    the scores used for selection. Returned weights are always gathered from the
    unbiased ``scores``.

    Args:
        top_k: Number of experts selected for each token.
        num_experts: Number of experts in the score dimension.
        bias: Whether to create a selection-only bias and an expert-count buffer.
            The bias is applied here but is updated by an optimizer-step hook,
            not by this module.
        stable_tie: Whether to use a stable descending sort. If ``True``, the
            indices are returned in descending selection-score order and ties
            favor the lower expert index. If ``False``, ``torch.topk`` is used
            with ``sorted=False``: selection is faster, but index order and tie
            behavior are unspecified.
        device: The device of the bias and expert-count buffers.
        dtype: The dtype of the selection bias. Counts are always int64.

    Shape:
        - Input: ``scores`` of shape ``(*, num_experts)``.
        - Output: ``(topk_weights, topk_indices)``, both of shape
          ``(*, top_k)``.

    Attributes:
        top_k: Number of experts selected for each token.
        num_experts: Number of experts in the score dimension.
        stable_tie: Whether selection uses a stable descending sort.
        bias: Selection-only bias buffer of shape ``(num_experts,)``, or
            ``None`` when disabled.
        expert_count: Non-persistent int64 count buffer of shape
            ``(num_experts,)``, or ``None`` when disabled.

    Note:
        ``expert_count`` is accumulated during every forward with gradients
        enabled. Evaluation and inference must run under
        ``torch.inference_mode()`` or ``torch.no_grad()`` to avoid polluting the
        training counts and to skip the unnecessary expert-count computation.

        The count reduction, bias update, and reset are not implemented here.
        TODO: add the optimizer-step hook that reduces the counts across ranks
        and updates the selection bias.

        Registered buffers are visible to distributed wrappers. Vanilla DDP
        broadcasts buffers by default, including non-persistent buffers;
        exclude ``expert_count`` from that broadcast or disable buffer
        broadcasting. Megatron-style DDP and FSDP2 do not broadcast registered
        buffers on every forward.

        TODO: add expert capacity control, capacity-based token dropping, and
        token masking. Until then, every score row is treated as a valid token,
        and every selected index is assumed to be a real expert. Padding rows
        from token masking and ``-1`` entries in dense routing or top-k index
        tensors from capacity/drop layouts are unsupported.

    """

    __constants__: ClassVar[list[str]] = ["top_k", "num_experts", "stable_tie"]

    def __init__(
        self: Self,
        top_k: int,
        num_experts: int,
        bias: bool = False,
        stable_tie: bool = True,
        device: Device = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize the top-k expert selector.

        Args:
            top_k: Number of experts selected for each token.
            num_experts: Number of experts in the score dimension.
            bias: Whether to create a selection-only bias and an expert-count buffer.
            stable_tie: Whether to use a stable sort.
            device: The device of the bias and expert-count buffers.
            dtype: The dtype of the selection bias.

        Raises:
            ValueError: If ``top_k`` is not between 1 and ``num_experts``.

        """
        if not (0 < top_k <= num_experts):
            raise ValueError(f"top_k must be between 1 and num_experts ({num_experts}), got {top_k}")

        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.top_k = top_k
        self.num_experts = num_experts
        if bias:
            self.register_buffer("bias", torch.zeros(num_experts, **factory_kwargs))
            self.register_buffer(
                "expert_count",
                torch.zeros_like(cast(torch.Tensor, self.bias), dtype=torch.int64),
                persistent=False,
            )
        else:
            self.register_buffer("bias", None)
            self.register_buffer("expert_count", None, persistent=False)
        self.stable_tie = stable_tie

    def forward(self: Self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select the top-k experts for each token.

        When the instance attribute ``self.stable_tie`` is ``True`` and a
        selection bias is enabled, the indices follow descending biased-score
        order, with ties favoring the lower expert index. The weights are
        gathered from the unbiased ``scores`` in that same index order, so they
        may not be descending by their own values. Without a bias,
        ``self.stable_tie=True`` returns scores in descending order.

        When ``self.stable_tie`` is ``False``, the selector uses
        ``torch.topk`` with ``sorted=False``. Selection is faster, but the
        outputs are not sorted, their order is unspecified, and tie selection
        is not guaranteed to be stable across invocations.

        Args:
            scores: The score matrix of shape ``(*, num_experts)``.

        Returns:
            A tuple ``(topk_weights, topk_indices)``, both of shape
            ``(*, top_k)``. The weights are gathered from the unbiased scores.

        Side Effects:
            When ``self.expert_count`` is not ``None`` and gradients are
            enabled, the selected expert counts are accumulated into that
            buffer in place. This side effect does not affect the returned
            tensors.

        """
        if self.stable_tie:
            selection_scores = scores if self.bias is None else scores + self.bias
            topk_indices = torch.argsort(selection_scores, dim=-1, descending=True, stable=True)[..., : self.top_k]
            topk_weights = torch.gather(scores, -1, topk_indices)
        elif self.bias is None:
            topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1, largest=True, sorted=False)
        else:
            selection_scores = scores + self.bias
            _, topk_indices = torch.topk(selection_scores, self.top_k, dim=-1, largest=True, sorted=False)
            topk_weights = torch.gather(scores, -1, topk_indices)

        if self.expert_count is not None and torch.is_grad_enabled():
            with torch.no_grad():
                flattened_topk_indices = topk_indices.flatten()
                self.expert_count.scatter_add_(
                    0,
                    flattened_topk_indices,
                    torch.ones_like(flattened_topk_indices, dtype=self.expert_count.dtype),
                )

        return topk_weights, topk_indices

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.

        Returns:
            A string describing the module configuration.

        """
        return (
            f"top_k={self.top_k}, num_experts={self.num_experts}, "
            f"bias={self.bias is not None}, stable_tie={self.stable_tie}"
        )


class TopKTokenRouter(torch.nn.Module):
    r"""
    A token-choice top-k router.

    The router projects tokens through a gate, applies the configured score
    module, selects the top-k experts, and optionally normalizes the selected
    weights. Scoring, selection, and normalization are separate modules.

    Args:
        num_experts: Number of experts.
        in_features: Size of each input sample.
        top_k: Number of experts selected for each token.
        bias: Whether the gate projection has a bias. Default: ``True``
        score: The score module type or factory. It is applied to the gate
            logits. Default: ``FP32SoftmaxScore``
        selector: The selector module type or factory. It is constructed with
            ``top_k``, ``num_experts``, and ``device``. Pass a factory or
            ``functools.partial`` to enable selector options such as ``bias``
            and ``stable_tie``. Default: ``TopKExpertSelector``
        normalization: The normalization module type or factory. It is applied
            to the selected top-k weights. Default: ``torch.nn.Identity``
        device: The device of the parameters. Default: ``None``
        dtype: The dtype of the gate parameters. Default: ``None``

    Shape:
        - Input: :math:`(*, H_\text{in})`, where :math:`H_\text{in} =
          \text{in\_features}`.
        - Output: a tuple ``(topk_weights, topk_indices)``, both of shape
          :math:`(*, k)`, where :math:`k = \text{top\_k}`.

    Attributes:
        num_experts: Number of experts.
        in_features: Size of each input sample.
        top_k: Number of experts selected for each token.
        gate_proj: The projection from ``in_features`` to ``num_experts``.
        score: The score module applied to the gate logits.
        selector: The module that selects and weights the top-k experts.
        normalization: The module applied to the selected weights.

    Note:
        The selector is constructed with ``top_k``; the router also stores it
        for introspection and representation. Weight dtype, index order, and tie
        handling depend on the configured score, selector, and normalization
        modules; see :class:`TopKExpertSelector` for the default selection
        semantics.

        TODO: add auxiliary-loss support. The router currently discards the
        full score matrix and routing map needed for load balancing.

    """

    __constants__: ClassVar[list[str]] = ["num_experts", "in_features", "top_k"]

    def __init__(
        self: Self,
        num_experts: int,
        in_features: int,
        top_k: int,
        bias: bool = True,
        score: Callable[..., torch.nn.Module] = FP32SoftmaxScore,
        selector: Callable[..., torch.nn.Module] = TopKExpertSelector,
        normalization: Callable[..., torch.nn.Module] = torch.nn.Identity,
        device: Device = None,
        dtype: DType = None,
    ) -> None:
        """
        Initialize the top-k token router.

        Args:
            num_experts: Number of experts.
            in_features: Size of each input sample.
            top_k: Number of experts selected for each token.
            bias: Whether the gate projection has a bias. Default: ``True``
            score: The score module type or factory. It is applied to the gate
                logits. Default: ``FP32SoftmaxScore``
            selector: The selector module type or factory. It is constructed with
                ``top_k``, ``num_experts``, and ``device``. Pass a factory or
                ``functools.partial`` to enable selector options such as
                ``bias`` and ``stable_tie``. Default: ``TopKExpertSelector``
            normalization: The normalization module type or factory. It is
                applied to the selected top-k weights. Default: ``torch.nn.Identity``
            device: The device of the parameters. Default: ``None``
            dtype: The dtype of the gate parameters. Default: ``None``

        Raises:
            ValueError: If ``top_k`` is not between 1 and ``num_experts``.

        """
        if not (0 < top_k <= num_experts):
            raise ValueError(f"top_k must be between 1 and num_experts ({num_experts}), got {top_k}")

        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_experts = num_experts
        self.in_features = in_features
        self.top_k = top_k
        self.gate_proj = torch.nn.Linear(in_features, num_experts, bias=bias, **factory_kwargs)
        self.score = score()
        self.selector = selector(top_k, num_experts, device=device)
        self.normalization = normalization()

    def forward(self: Self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select the top-k experts for each token.

        Args:
            input: The input tensor of shape ``(*, in_features)``.

        Returns:
            A tuple ``(topk_weights, topk_indices)``. Both tensors have shape
            ``(*, top_k)``. Weight dtype follows the score and normalization
            modules; the defaults produce float32. Index order and tie handling
            depend on the selector's ``stable_tie`` option.

        """
        logits = self.gate_proj(input)
        scores = self.score(logits)
        topk_weights, topk_indices = self.selector(scores)
        topk_weights = self.normalization(topk_weights)
        return topk_weights, topk_indices

    def extra_repr(self: Self) -> str:
        """
        Return the extra representation of the module.

        Returns:
            A string describing the module configuration.

        """
        return f"num_experts={self.num_experts}, in_features={self.in_features}, top_k={self.top_k}"
