# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Token routing components."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Self, cast

import torch
import torch.nn.functional as F
from torch.types import Device

from mokit.core import MokitConfigBase
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
    the scores used for selection. Returned scores are always gathered from the
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
        - Output: ``(topk_scores, topk_indices)``, both of shape
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
        order, with ties favoring the lower expert index. The selected scores
        are gathered from the unbiased ``scores`` in that same index order, so
        they may not be descending by their own values. Without a bias,
        ``self.stable_tie=True`` returns scores in descending order.

        When ``self.stable_tie`` is ``False``, the selector uses
        ``torch.topk`` with ``sorted=False``. Selection is faster, but the
        outputs are not sorted, their order is unspecified, and tie selection
        is not guaranteed to be stable across invocations.

        Args:
            scores: The score matrix of shape ``(*, num_experts)``.

        Returns:
            A tuple ``(topk_scores, topk_indices)``, both of shape
            ``(*, top_k)``. The scores are gathered from the unbiased scores.

        Side Effects:
            When ``self.expert_count`` is not ``None`` and gradients are
            enabled, the selected expert counts are accumulated into that
            buffer in place. This side effect does not affect the returned
            tensors.

        """
        if self.stable_tie:
            selection_scores = scores if self.bias is None else scores + self.bias
            topk_indices = torch.argsort(selection_scores, dim=-1, descending=True, stable=True)[..., : self.top_k]
            topk_scores = torch.gather(scores, -1, topk_indices)
        elif self.bias is None:
            topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1, largest=True, sorted=False)
        else:
            selection_scores = scores + self.bias
            _, topk_indices = torch.topk(selection_scores, self.top_k, dim=-1, largest=True, sorted=False)
            topk_scores = torch.gather(scores, -1, topk_indices)

        if self.expert_count is not None and torch.is_grad_enabled():
            with torch.no_grad():
                flattened_topk_indices = topk_indices.flatten()
                self.expert_count.scatter_add_(
                    0,
                    flattened_topk_indices,
                    torch.ones_like(flattened_topk_indices, dtype=self.expert_count.dtype),
                )

        return topk_scores, topk_indices

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

    The router projects tokens through a gate, computes scores, selects the top-k
    experts, normalizes the selected scores, and returns the ``topk_scores`` and
    ``topk_indices``. The default normalization is a no-op. ``topk_scores`` serve
    as the weighting scores for the combine stage. Scoring, selection, and
    normalization are separate modules.

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
            to the selected top-k scores. Default: ``torch.nn.Identity``
        device: The device of the parameters. Default: ``None``
        dtype: The dtype of the gate parameters. Default: ``None``

    Shape:
        - Input: :math:`(*, H_\text{in})`, where :math:`H_\text{in} =
          \text{in\_features}`.
        - Output: a tuple ``(topk_scores, topk_indices)``, both of shape
          :math:`(*, k)`, where :math:`k = \text{top\_k}`.

    Attributes:
        num_experts: Number of experts.
        in_features: Size of each input sample.
        top_k: Number of experts selected for each token.
        gate_proj: The projection from ``in_features`` to ``num_experts``.
        score: The score module applied to the gate logits.
        selector: The module that selects the top-k experts and returns their scores.
        normalization: The module applied to the selected scores.

    Note:
        The selector is constructed with ``top_k``; the router also stores it
        for introspection and representation. Score dtype, index order, and tie
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
                applied to the selected top-k scores. Default: ``torch.nn.Identity``
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
            A tuple ``(topk_scores, topk_indices)``. Both tensors have shape
            ``(*, top_k)``. Score dtype follows the score and normalization
            modules; the defaults produce float32. Index order and tie handling
            depend on the selector's ``stable_tie`` option.

        """
        logits = self.gate_proj(input)
        scores = self.score(logits)
        topk_scores, topk_indices = self.selector(scores)
        topk_scores = self.normalization(topk_scores)
        return topk_scores, topk_indices

    def extra_repr(self: Self) -> str:
        """
        Return the extra representation of the module.

        Returns:
            A string describing the module configuration.

        """
        return f"num_experts={self.num_experts}, in_features={self.in_features}, top_k={self.top_k}"


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenDispatchState(MokitConfigBase):
    """
    State produced by :meth:`LocalTokenDispatcher.dispatch`.

    ``N`` is the number of dispatched rows.

    Attributes:
        expert_scores: Selected expert scores in dispatched order, shape ``(N,)``.
        token_indices: Source token index for each dispatched row, shape ``(N,)``.
        input_leading_dims: Leading dimensions of the original token tensor.

    """

    expert_scores: torch.Tensor
    token_indices: torch.Tensor
    input_leading_dims: torch.Size


class LocalTokenDispatcher(torch.nn.Module):
    """
    A local, dropless token dispatcher.

    ``dispatch`` groups routed tokens by expert and records the state needed to
    reverse the permutation. ``combine`` weights expert outputs by their
    routing scores and sums them back to token order.

    Args:
        num_experts: Number of experts.
        stable_permutation: Whether ``dispatch`` uses a stable expert-major sort. ``None`` follows
            ``torch.are_deterministic_algorithms_enabled()`` at construction time. Default: ``None``

    Shape:
        - ``dispatch`` input: ``tokens`` of shape ``(*, H)``, with
          ``expert_scores`` and ``expert_indices`` of shape ``(*, k)``.
        - ``dispatch`` output: dispatched tokens of shape ``(N, H)`` and
          offsets of shape ``(num_experts,)``, where ``N = M * k`` and
          ``M = prod(*)`` is the number of input tokens.
        - ``combine`` input: expert outputs of shape ``(N, H_out)``.
        - ``combine`` output: ``(*, H_out)``, restoring the leading dimensions
          of ``tokens``.

    Attributes:
        num_experts: Number of experts.
        stable_permutation: Whether the expert-major permutation is stable.

    """

    __constants__: ClassVar[list[str]] = ["num_experts", "stable_permutation", "_expert_indices_sort_dtype"]

    def __init__(self: Self, num_experts: int, stable_permutation: bool | None = None) -> None:
        """
        Initialize the local token dispatcher.

        Args:
            num_experts: Number of experts.
            stable_permutation: Whether to use a stable expert-major sort.
                ``None`` follows the global deterministic-algorithms setting at
                construction time. Default: ``None``

        """
        super().__init__()
        self.num_experts = num_experts
        self.stable_permutation = (
            torch.are_deterministic_algorithms_enabled() if stable_permutation is None else stable_permutation
        )
        # Use the narrowest dtype that can represent expert IDs
        # [0, num_experts) as the argsort key for performance. The original
        # int64 expert_indices is still needed for the offset histogram.
        if num_experts <= 2**8:
            self._expert_indices_sort_dtype = torch.uint8
        elif num_experts <= 2**15:
            self._expert_indices_sort_dtype = torch.int16
        elif num_experts <= 2**31:
            self._expert_indices_sort_dtype = torch.int32
        else:
            self._expert_indices_sort_dtype = torch.int64

    def forward(self: Self) -> None:
        """
        Disable ``forward`` on purpose.

        The dispatcher exposes separate :meth:`dispatch` and :meth:`combine`
        operations and must not be called as a single forward module.

        Raises:
            NotImplementedError: Always.

        """
        raise NotImplementedError("LocalTokenDispatcher.forward is disabled; use dispatch() and combine() instead.")

    def dispatch(
        self: Self,
        tokens: torch.Tensor,
        expert_scores: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, TokenDispatchState]:
        """
        Group routed tokens by expert and return the dispatch state.

        Args:
            tokens: Token tensor of shape ``(*, H)``.
            expert_scores: Selected expert scores of shape ``(*, k)``, with the
                same leading dimensions as ``tokens``.
            expert_indices: Selected expert indices of shape ``(*, k)``, with
                the same leading dimensions as ``tokens``.

        Returns:
            A tuple ``(dispatched_tokens, offs, state)``.
            ``dispatched_tokens`` has shape ``(N, H)`` and is ordered by expert,
            where ``N`` is the number of routed rows. ``offs`` is an int32
            cumulative end offset per expert, shape ``(num_experts,)``.
            Expert 0 owns the half-open rows ``[0, offs[0])``; expert
            ``e > 0`` owns ``[offs[e - 1], offs[e])``. The final entry is
            ``N``. This is the end-offset layout expected by
            :func:`torch.nn.functional.grouped_mm`. ``state`` is the
            :class:`TokenDispatchState` consumed by :meth:`combine`.

        """
        input_leading_dims = tokens.shape[:-1]
        tokens = tokens.reshape(-1, tokens.size(-1))
        tokens_per_expert = expert_indices.size(-1)

        expert_scores = expert_scores.reshape(-1)
        expert_indices = expert_indices.reshape(-1)

        # Sort the narrow key, not the int64 expert_indices, for better performance.
        perm_indices = torch.argsort(expert_indices.to(self._expert_indices_sort_dtype), stable=self.stable_permutation)
        # Write int32 token indices directly to save bandwidth and memory.
        # index_select and scatter_add_ accept int32.
        token_indices = torch.empty_like(perm_indices, dtype=torch.int32)
        torch.floor_divide(perm_indices, tokens_per_expert, out=token_indices)
        repeated_tokens = tokens.index_select(0, token_indices)

        permuted_expert_scores = expert_scores[perm_indices]

        # Fixed-bin histc computes expert counts without a host sync;
        # cumsum(dtype=int32) matches the end-offset contract above.
        offs = torch.histc(expert_indices, self.num_experts, 0, self.num_experts).cumsum(0, dtype=torch.int32)

        token_dispatch_state = TokenDispatchState(
            expert_scores=permuted_expert_scores,
            token_indices=token_indices,
            input_leading_dims=input_leading_dims,
        )

        return repeated_tokens, offs, token_dispatch_state

    def combine(self: Self, tokens: torch.Tensor, state: TokenDispatchState) -> torch.Tensor:
        """
        Weight expert outputs by their scores and sum them back to token order.

        The element-wise multiplication of the routing scores and the expert
        outputs uses both operands' original dtypes; no extra cast is inserted.
        The product is then cast to ``tokens.dtype``. The reduction therefore
        adds ``tokens.dtype`` values, but the intermediate dtype used inside the
        reduction kernel is unspecified and subject to other controls.

        Args:
            tokens: Expert outputs in dispatched order, shape ``(N, H_out)``.
            state: The state returned by the matching :meth:`dispatch` call.

        Returns:
            The combined tensor of shape ``(*input_leading_dims, H_out)`` and
            dtype ``tokens.dtype``. Each token is the sum of its selected expert
            outputs weighted by the corresponding routing scores.

        """
        weighted_tokens = state.expert_scores.unsqueeze(-1) * tokens
        weighted_tokens = weighted_tokens.to(tokens.dtype)

        reduced_tokens = weighted_tokens.new_zeros((state.input_leading_dims.numel(), weighted_tokens.size(-1)))
        reduced_tokens.scatter_add_(
            0,
            state.token_indices.unsqueeze(1).expand(-1, weighted_tokens.size(-1)),
            weighted_tokens,
        )

        reduced_tokens = reduced_tokens.reshape(*state.input_leading_dims, weighted_tokens.size(-1))

        return reduced_tokens

    def extra_repr(self: Self) -> str:
        """
        Return the extra representation of the module.

        Returns:
            A string describing the module configuration.

        """
        return f"num_experts={self.num_experts}, stable_permutation={self.stable_permutation}"
