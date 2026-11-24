# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.components.token_routing`."""

import dataclasses
from functools import partial
from typing import cast

import pytest
import torch

from mokit.components.token_routing import (
    FP32SoftmaxScore,
    LocalTokenDispatcher,
    TokenDispatchState,
    TopKExpertSelector,
    TopKTokenRouter,
)

# ==============================================================================
# FP32SoftmaxScore
# ==============================================================================


# ------------------------------------------------------------------------------
# forward
# ------------------------------------------------------------------------------


def test_fp32_softmax_score_forward_matches_reference() -> None:
    """Test that forward applies softmax over the last dimension in float32."""
    module = FP32SoftmaxScore()
    scores = torch.randn(2, 3, 4, dtype=torch.float64)

    output = module(scores)

    expected = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
    assert output.dtype is torch.float32
    torch.testing.assert_close(output, expected)


# ==============================================================================
# TopKExpertSelector
# ==============================================================================


# ------------------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------------------


def test_topk_expert_selector_init_stores_configuration() -> None:
    """Test that __init__ stores the configuration and defaults."""
    selector = TopKExpertSelector(2, 4)

    assert selector.top_k == 2
    assert selector.num_experts == 4
    assert selector.stable_tie is True
    assert selector.bias is None
    assert selector.expert_count is None
    assert selector.__constants__ == ["top_k", "num_experts", "stable_tie"]


def test_topk_expert_selector_init_registers_bias_and_count() -> None:
    """Test that bias creates persistent bias and non-persistent count buffers."""
    selector = TopKExpertSelector(2, 4, bias=True)
    assert selector.bias is not None
    assert selector.expert_count is not None

    assert torch.equal(selector.bias, torch.zeros(4))
    assert selector.bias.dtype is torch.float32
    assert selector.expert_count.dtype is torch.int64
    assert torch.equal(selector.expert_count, torch.zeros(4, dtype=torch.int64))
    assert "bias" in selector.state_dict()
    assert "expert_count" not in selector.state_dict()
    assert dict(selector.named_parameters()) == {}


def test_topk_expert_selector_init_without_bias_has_no_count() -> None:
    """Test that disabling the bias leaves both optional buffers as None."""
    selector = TopKExpertSelector(2, 4, bias=False)

    assert selector.bias is None
    assert selector.expert_count is None
    assert selector.state_dict() == {}


def test_topk_expert_selector_init_respects_bias_dtype() -> None:
    """Test that the selection bias uses the requested dtype."""
    selector = TopKExpertSelector(2, 4, bias=True, dtype=torch.float64)
    assert selector.bias is not None
    assert selector.expert_count is not None

    assert selector.bias.dtype is torch.float64
    assert selector.expert_count.dtype is torch.int64


@pytest.mark.parametrize("top_k", [0, -1, 5], ids=["zero", "negative", "too_large"])
def test_topk_expert_selector_init_rejects_invalid_top_k(top_k: int) -> None:
    """Test that top_k must be between one and num_experts."""
    with pytest.raises(ValueError, match="top_k"):
        TopKExpertSelector(top_k, 4)


def test_topk_expert_selector_init_stores_stable_tie() -> None:
    """Test that stable_tie is stored and exposed through repr."""
    selector = TopKExpertSelector(2, 4, stable_tie=False)

    assert selector.stable_tie is False
    assert "stable_tie=False" in selector.extra_repr()


# ------------------------------------------------------------------------------
# forward
# ------------------------------------------------------------------------------


def _stable_reference(scores: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the expected stable top-k scores and indices."""
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :top_k]
    return torch.gather(scores, -1, indices), indices


@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 5)], ids=["2d", "3d"])
def test_topk_expert_selector_forward_stable_matches_reference(shape: tuple[int, ...]) -> None:
    """Test that stable selection matches the stable argsort reference."""
    selector = TopKExpertSelector(2, 5)
    scores = torch.randn(*shape)

    selected_scores, indices = selector(scores)

    expected_scores, expected_indices = _stable_reference(scores, 2)
    assert torch.equal(selected_scores, expected_scores)
    assert torch.equal(indices, expected_indices)


def test_topk_expert_selector_forward_stable_selects_largest_scores() -> None:
    """Test that stable selection chooses the largest scores in descending order."""
    selector = TopKExpertSelector(2, 4)
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])

    selected_scores, indices = selector(scores)

    assert indices.tolist() == [[3, 2], [0, 1]]
    assert torch.equal(selected_scores, torch.tensor([[3.0, 2.0], [3.0, 2.0]]))


def test_topk_expert_selector_forward_stable_breaks_ties_by_expert_index() -> None:
    """Test that equal scores select the lower expert index."""
    selector = TopKExpertSelector(2, 4)
    scores = torch.tensor([[0.0, 1.0, 1.0, 0.0]])

    selected_scores, indices = selector(scores)

    assert indices.tolist() == [[1, 2]]
    assert torch.equal(selected_scores, torch.tensor([[1.0, 1.0]]))


def test_topk_expert_selector_forward_supports_top_k_one() -> None:
    """Test that top_k=1 selects one expert per token."""
    selector = TopKExpertSelector(1, 4)
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    selected_scores, indices = selector(scores)

    assert indices.tolist() == [[3]]
    assert torch.equal(selected_scores, torch.tensor([[3.0]]))


def test_topk_expert_selector_forward_supports_all_experts() -> None:
    """Test that selecting every expert returns all indices in descending order."""
    selector = TopKExpertSelector(4, 4)
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    selected_scores, indices = selector(scores)

    assert indices.tolist() == [[3, 2, 1, 0]]
    assert torch.equal(selected_scores, torch.tensor([[3.0, 2.0, 1.0, 0.0]]))


def test_topk_expert_selector_forward_stable_bias_uses_biased_selection() -> None:
    """Test that bias changes selection but not the gathered scores."""
    selector = TopKExpertSelector(2, 3, bias=True)
    assert selector.bias is not None
    with torch.no_grad():
        selector.bias.copy_(torch.tensor([0.0, 1.0, 0.0]))
    scores = torch.tensor([[0.9, 0.2, 0.1]])

    selected_scores, indices = selector(scores)

    assert indices.tolist() == [[1, 0]]
    assert torch.equal(selected_scores, torch.tensor([[0.2, 0.9]]))


def test_topk_expert_selector_forward_stable_handles_empty_input() -> None:
    """Test that empty scores produce empty outputs."""
    selector = TopKExpertSelector(2, 4)

    selected_scores, indices = selector(torch.empty(0, 4))

    assert selected_scores.shape == (0, 2)
    assert indices.shape == (0, 2)
    assert selected_scores.dtype is torch.float32
    assert indices.dtype is torch.int64


def test_topk_expert_selector_forward_preserves_score_dtype() -> None:
    """Test that the selector does not cast scores for its own convenience."""
    selector = TopKExpertSelector(2, 4)
    scores = torch.randn(3, 4, dtype=torch.float64)

    selected_scores, indices = selector(scores)

    assert selected_scores.dtype is torch.float64
    assert indices.dtype is torch.int64


def test_topk_expert_selector_forward_backward_reaches_scores() -> None:
    """Test that selected scores carry gradients to the score tensor."""
    selector = TopKExpertSelector(2, 4)
    scores = torch.randn(3, 4, requires_grad=True)

    selected_scores, indices = selector(scores)
    selected_scores.sum().backward()

    assert not indices.requires_grad
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad) == 6


def test_topk_expert_selector_forward_bias_has_no_gradient() -> None:
    """Test that the selection-only bias does not participate in autograd."""
    selector = TopKExpertSelector(2, 4, bias=True)
    assert selector.bias is not None
    scores = torch.randn(3, 4, requires_grad=True)

    selected_scores, _ = selector(scores)
    selected_scores.sum().backward()

    assert selector.bias.requires_grad is False
    assert scores.grad is not None


# ------------------------------------------------------------------------------
# forward, unstable selection
# ------------------------------------------------------------------------------


def test_topk_expert_selector_forward_unstable_matches_topk() -> None:
    """Test that unstable selection returns the same selected scores as topk."""
    selector = TopKExpertSelector(2, 4, stable_tie=False)
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])

    selected_scores, indices = selector(scores)

    expected_values = torch.topk(scores, 2, dim=-1).values
    assert indices.shape == (2, 2)
    assert torch.equal(selected_scores, torch.gather(scores, -1, indices))
    assert torch.equal(selected_scores.sort(dim=-1, descending=True).values, expected_values)


def test_topk_expert_selector_forward_unstable_bias_matches_topk() -> None:
    """Test that unstable selection uses biased scores but returns unbiased scores."""
    selector = TopKExpertSelector(2, 3, bias=True, stable_tie=False)
    assert selector.bias is not None
    with torch.no_grad():
        selector.bias.copy_(torch.tensor([0.0, 1.0, 0.0]))
    scores = torch.tensor([[0.9, 0.2, 0.1]])

    selected_scores, indices = selector(scores)

    selection_scores = scores + selector.bias
    expected_values = torch.topk(selection_scores, 2, dim=-1).values
    assert torch.equal(selected_scores, torch.gather(scores, -1, indices))
    assert torch.equal(
        torch.gather(selection_scores, -1, indices).sort(dim=-1, descending=True).values,
        expected_values,
    )


# ------------------------------------------------------------------------------
# expert_count
# ------------------------------------------------------------------------------


def test_topk_expert_selector_counts_experts_when_grad_enabled() -> None:
    """Test that expert counts match the selected indices."""
    selector = TopKExpertSelector(2, 4, bias=True)
    assert selector.expert_count is not None
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])

    _, indices = selector(scores)

    expected = torch.bincount(indices.flatten(), minlength=4)
    assert torch.equal(selector.expert_count, expected)
    assert selector.expert_count.sum() == 4


def test_topk_expert_selector_accumulates_counts_across_forward_calls() -> None:
    """Test that counts accumulate across grad-enabled forward calls."""
    selector = TopKExpertSelector(2, 4, bias=True)
    assert selector.expert_count is not None
    scores = torch.randn(3, 4)

    selector(scores)
    selector(scores)

    assert selector.expert_count.sum() == 12


def test_topk_expert_selector_skips_counts_without_grad() -> None:
    """Test that no_grad and inference_mode skip expert counting."""
    selector = TopKExpertSelector(2, 4, bias=True)
    assert selector.expert_count is not None
    scores = torch.randn(3, 4)

    with torch.no_grad():
        selector(scores)
    assert selector.expert_count.sum() == 0

    with torch.inference_mode():
        selector(scores)
    assert selector.expert_count.sum() == 0


def test_topk_expert_selector_counts_without_bias() -> None:
    """Test that disabled bias also disables expert counting."""
    selector = TopKExpertSelector(2, 4, bias=False)

    selector(torch.randn(3, 4))

    assert selector.expert_count is None


# ------------------------------------------------------------------------------
# extra_repr
# ------------------------------------------------------------------------------


def test_topk_expert_selector_extra_repr_reports_configuration() -> None:
    """Test that extra_repr reports all selector options."""
    selector = TopKExpertSelector(2, 4, bias=True, stable_tie=False)

    assert selector.extra_repr() == "top_k=2, num_experts=4, bias=True, stable_tie=False"


# ==============================================================================
# TopKTokenRouter
# ==============================================================================


# ------------------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------------------


def test_topk_token_router_init_stores_configuration_and_defaults() -> None:
    """Test that __init__ stores the configuration and builds default modules."""
    router = TopKTokenRouter(4, 8, 2)

    assert router.num_experts == 4
    assert router.in_features == 8
    assert router.gate_proj.in_features == 8
    assert router.gate_proj.out_features == 4
    assert router.gate_proj.bias is not None
    assert isinstance(router.score, FP32SoftmaxScore)
    assert isinstance(router.selector, TopKExpertSelector)
    assert isinstance(router.normalization, torch.nn.Identity)
    assert router.selector.top_k == 2
    assert router.top_k == 2
    assert router.__constants__ == ["num_experts", "in_features", "top_k"]


def test_topk_token_router_init_respects_dtype() -> None:
    """Test that the gate uses the requested dtype while selector state stays float32."""
    router = TopKTokenRouter(4, 8, 2, dtype=torch.float64, selector=partial(TopKExpertSelector, bias=True))
    selector = cast(TopKExpertSelector, router.selector)
    assert selector.bias is not None
    assert selector.expert_count is not None
    assert router.gate_proj.bias is not None

    assert router.gate_proj.weight.dtype is torch.float64
    assert router.gate_proj.bias.dtype is torch.float64
    assert selector.bias.dtype is torch.float32
    assert selector.expert_count.dtype is torch.int64


@pytest.mark.parametrize("top_k", [0, -1, 5], ids=["zero", "negative", "too_large"])
def test_topk_token_router_init_rejects_invalid_top_k(top_k: int) -> None:
    """Test that the router validates top_k independently of the selector."""
    with pytest.raises(ValueError, match="top_k"):
        TopKTokenRouter(4, 8, top_k)


# ------------------------------------------------------------------------------
# forward
# ------------------------------------------------------------------------------


def _router_reference(router: TopKTokenRouter, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the expected unnormalized top-k result for the default router."""
    logits = router.gate_proj(input)
    scores = torch.nn.functional.softmax(logits.float(), dim=-1)
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., : router.selector.top_k]
    return torch.gather(scores, -1, indices), indices


@pytest.mark.parametrize("shape", [(3, 8), (2, 3, 8)], ids=["2d", "3d"])
def test_topk_token_router_forward_matches_reference(shape: tuple[int, ...]) -> None:
    """Test that forward matches the gate/softmax/stable-top-k reference."""
    router = TopKTokenRouter(4, 8, 2)
    input = torch.randn(*shape)

    selected_scores, indices = router(input)

    expected_scores, expected_indices = _router_reference(router, input)
    assert selected_scores.dtype is torch.float32
    assert indices.dtype is torch.int64
    torch.testing.assert_close(selected_scores, expected_scores)
    assert torch.equal(indices, expected_indices)


def test_topk_token_router_forward_uses_custom_modules() -> None:
    """Test that score, selector, and normalization are injectable."""
    router = TopKTokenRouter(
        4,
        8,
        2,
        score=torch.nn.Identity,
        selector=partial(TopKExpertSelector, stable_tie=False),
        normalization=partial(torch.nn.Softmax, dim=-1),
    )
    input = torch.randn(3, 8)

    selected_scores, indices = router(input)

    logits = router.gate_proj(input)
    assert indices.shape == (3, 2)
    torch.testing.assert_close(selected_scores, torch.gather(logits, -1, indices).softmax(dim=-1))


def test_topk_token_router_forward_handles_empty_input() -> None:
    """Test that an empty input produces empty outputs."""
    router = TopKTokenRouter(4, 8, 2)

    selected_scores, indices = router(torch.empty(0, 8))

    assert selected_scores.shape == (0, 2)
    assert indices.shape == (0, 2)


def test_topk_token_router_forward_rejects_wrong_input_width() -> None:
    """Test that the gate rejects an input whose last dimension is not in_features."""
    router = TopKTokenRouter(4, 8, 2)

    with pytest.raises(RuntimeError):
        router(torch.randn(2, 7))


def test_topk_token_router_forward_backward_computes_gradients() -> None:
    """Test that router scores carry gradients back through selection."""
    router = TopKTokenRouter(4, 8, 2)
    input = torch.randn(3, 8, requires_grad=True)

    selected_scores, indices = router(input)
    selected_scores.sum().backward()

    assert not indices.requires_grad
    assert input.grad is not None
    assert router.gate_proj.weight.grad is not None
    assert router.gate_proj.bias.grad is not None
    assert torch.isfinite(input.grad).all()
    assert torch.isfinite(router.gate_proj.weight.grad).all()


def test_topk_token_router_forward_counts_selected_experts() -> None:
    """Test that selector state accumulates through the router."""
    router = TopKTokenRouter(4, 8, 2, selector=partial(TopKExpertSelector, bias=True))
    selector = cast(TopKExpertSelector, router.selector)
    assert selector.expert_count is not None
    input = torch.randn(3, 8)

    _, indices = router(input)

    assert selector.expert_count.sum() == 6
    assert torch.equal(selector.expert_count, torch.bincount(indices.flatten(), minlength=4))


# ------------------------------------------------------------------------------
# extra_repr
# ------------------------------------------------------------------------------


def test_topk_token_router_extra_repr_reports_configuration() -> None:
    """Test that extra_repr reports the router configuration."""
    router = TopKTokenRouter(4, 8, 2)

    assert router.extra_repr() == "num_experts=4, in_features=8, top_k=2"


# ==============================================================================
# TokenDispatchState
# ==============================================================================


# ------------------------------------------------------------------------------
# Dataclass
# ------------------------------------------------------------------------------


def test_token_dispatch_state_stores_fields() -> None:
    """Test that the state stores the tensors and leading dimensions."""
    expert_scores = torch.tensor([0.5, 0.25])
    token_indices = torch.tensor([0, 1], dtype=torch.int32)
    input_leading_dims = torch.Size([2])

    state = TokenDispatchState(
        expert_scores=expert_scores,
        token_indices=token_indices,
        input_leading_dims=input_leading_dims,
    )

    assert state.expert_scores is expert_scores
    assert state.token_indices is token_indices
    assert state.input_leading_dims == input_leading_dims


def test_token_dispatch_state_is_frozen_and_slotted() -> None:
    """Test that the state is a frozen slotted dataclass."""
    assert dataclasses.is_dataclass(TokenDispatchState)
    state = TokenDispatchState(
        expert_scores=torch.zeros(1),
        token_indices=torch.zeros(1, dtype=torch.int32),
        input_leading_dims=torch.Size([1]),
    )
    assert not hasattr(state, "__dict__")
    attribute = "expert_scores"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(state, attribute, torch.ones(1))


# ==============================================================================
# LocalTokenDispatcher
# ==============================================================================


# ------------------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------------------


def test_local_token_dispatcher_init_stores_configuration() -> None:
    """Test that __init__ stores the configuration and the current determinism flag."""
    stable_permutation = torch.are_deterministic_algorithms_enabled()
    dispatcher = LocalTokenDispatcher(4)

    assert dispatcher.num_experts == 4
    assert dispatcher.stable_permutation is stable_permutation
    assert dispatcher.__constants__ == ["num_experts", "stable_permutation", "_expert_indices_sort_dtype"]


@pytest.mark.parametrize("stable_permutation", [True, False], ids=["stable", "unstable"])
def test_local_token_dispatcher_init_respects_stable_permutation(stable_permutation: bool) -> None:
    """Test that an explicit stable_permutation is stored unchanged."""
    dispatcher = LocalTokenDispatcher(4, stable_permutation=stable_permutation)

    assert dispatcher.stable_permutation is stable_permutation


def test_local_token_dispatcher_init_follows_deterministic_algorithms() -> None:
    """Test that a None stable_permutation follows the deterministic-algorithms flag."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        dispatcher = LocalTokenDispatcher(4)
    finally:
        torch.use_deterministic_algorithms(previous)

    assert dispatcher.stable_permutation is True


@pytest.mark.parametrize(
    ("num_experts", "expected_dtype"),
    [
        pytest.param(2**8, torch.uint8, id="uint8_max"),
        pytest.param(2**8 + 1, torch.int16, id="int16_min"),
        pytest.param(2**15, torch.int16, id="int16_max"),
        pytest.param(2**15 + 1, torch.int32, id="int32_min"),
        pytest.param(2**31, torch.int32, id="int32_max"),
        pytest.param(2**31 + 1, torch.int64, id="int64"),
    ],
)
def test_local_token_dispatcher_init_selects_sort_dtype(num_experts: int, expected_dtype: torch.dtype) -> None:
    """Test that the argsort key uses the narrowest dtype that fits expert IDs."""
    dispatcher = LocalTokenDispatcher(num_experts)

    assert dispatcher._expert_indices_sort_dtype is expected_dtype


# ------------------------------------------------------------------------------
# forward
# ------------------------------------------------------------------------------


def test_local_token_dispatcher_forward_is_disabled() -> None:
    """Test that forward cannot be used instead of dispatch and combine."""
    dispatcher = LocalTokenDispatcher(4)

    with pytest.raises(NotImplementedError, match=r"dispatch\(\) and combine\(\)"):
        dispatcher()


# ------------------------------------------------------------------------------
# dispatch
# ------------------------------------------------------------------------------


_DISPATCH_EXPERT_INDICES = ((0, 1), (2, 2), (1, 3), (0, 0))


def _dispatch_inputs(
    device: torch.device,
    leading_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create tokens, scores, and expert indices with duplicate experts."""
    top_k = len(_DISPATCH_EXPERT_INDICES[0])
    expert_indices = torch.tensor(_DISPATCH_EXPERT_INDICES, dtype=torch.int64, device=device)
    return (
        torch.randn(*leading_shape, 3, device=device),
        torch.randn(*leading_shape, top_k, device=device),
        expert_indices.reshape(*leading_shape, top_k),
    )


def _dispatch_reference(
    tokens: torch.Tensor,
    expert_scores: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the expected dispatch outputs with a stable Python sort."""
    top_k = expert_indices.size(-1)
    flat_indices = expert_indices.reshape(-1).tolist()
    perm = sorted(range(len(flat_indices)), key=flat_indices.__getitem__)
    perm_tensor = torch.tensor(perm, dtype=torch.int64, device=tokens.device)
    counts = torch.bincount(expert_indices.reshape(-1), minlength=num_experts)
    offs = counts.cumsum(0).to(torch.int32)
    token_indices = (perm_tensor // top_k).to(torch.int32)
    dispatched = tokens.reshape(-1, tokens.size(-1)).index_select(0, token_indices)
    permuted_expert_scores = expert_scores.reshape(-1)[perm_tensor]
    return dispatched, offs, token_indices, permuted_expert_scores


@pytest.mark.requires_accelerator
@pytest.mark.parametrize("leading_shape", [(4,), (2, 2)], ids=["1d", "2d"])
def test_local_token_dispatcher_dispatch_matches_reference(
    leading_shape: tuple[int, ...],
    device: torch.device,
) -> None:
    """Test that dispatch matches a stable expert-major reference."""
    dispatcher = LocalTokenDispatcher(4, stable_permutation=True)
    tokens, expert_scores, expert_indices = _dispatch_inputs(device, leading_shape)

    dispatched, offs, state = dispatcher.dispatch(tokens, expert_scores, expert_indices)

    expected_tokens, expected_offs, expected_token_indices, expected_scores = _dispatch_reference(
        tokens,
        expert_scores,
        expert_indices,
        dispatcher.num_experts,
    )
    assert torch.equal(dispatched, expected_tokens)
    assert torch.equal(offs, expected_offs)
    assert torch.equal(state.token_indices, expected_token_indices)
    assert torch.equal(state.expert_scores, expected_scores)
    assert state.input_leading_dims == tokens.shape[:-1]


@pytest.mark.requires_accelerator
def test_local_token_dispatcher_dispatch_returns_int32_offsets_and_state(device: torch.device) -> None:
    """Test that dispatch returns int32 offsets and a usable state."""
    dispatcher = LocalTokenDispatcher(4, stable_permutation=True)
    tokens, expert_scores, expert_indices = _dispatch_inputs(device, (4,))

    dispatched, offs, state = dispatcher.dispatch(tokens, expert_scores, expert_indices)

    assert offs.dtype is torch.int32
    assert offs.shape == (4,)
    assert offs[-1].item() == dispatched.size(0)
    assert torch.all(offs[1:] >= offs[:-1])
    assert state.expert_scores.dtype is expert_scores.dtype
    assert state.token_indices.dtype is torch.int32
    assert state.token_indices.shape == (dispatched.size(0),)
    assert torch.all(state.token_indices >= 0)
    assert torch.all(state.token_indices < tokens.numel() // tokens.size(-1))
    assert state.input_leading_dims == torch.Size([4])


@pytest.mark.requires_accelerator
def test_local_token_dispatcher_dispatch_handles_empty_tokens(device: torch.device) -> None:
    """Test that an empty token tensor produces empty rows and zero offsets."""
    dispatcher = LocalTokenDispatcher(4, stable_permutation=True)
    tokens = torch.empty(0, 3, device=device)
    expert_scores = torch.empty(0, 2, device=device)
    expert_indices = torch.empty(0, 2, dtype=torch.int64, device=device)

    dispatched, offs, state = dispatcher.dispatch(tokens, expert_scores, expert_indices)

    assert dispatched.shape == (0, 3)
    assert torch.equal(offs, torch.zeros(4, dtype=torch.int32, device=device))
    assert state.token_indices.shape == (0,)
    assert state.token_indices.dtype is torch.int32
    assert state.expert_scores.shape == (0,)
    assert state.input_leading_dims == torch.Size([0])


@pytest.mark.requires_accelerator
def test_local_token_dispatcher_dispatch_supports_top_k_one(device: torch.device) -> None:
    """Test that top_k=1 groups every token exactly once."""
    dispatcher = LocalTokenDispatcher(4, stable_permutation=True)
    tokens = torch.arange(4, dtype=torch.float32, device=device).unsqueeze(1)
    expert_scores = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)
    expert_indices = torch.tensor([[0], [2], [1], [3]], dtype=torch.int64, device=device)

    dispatched, offs, state = dispatcher.dispatch(tokens, expert_scores, expert_indices)

    assert torch.equal(offs, torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device))
    assert torch.equal(dispatched[:, 0], torch.tensor([0.0, 2.0, 1.0, 3.0], device=device))
    assert torch.equal(state.token_indices, torch.tensor([0, 2, 1, 3], dtype=torch.int32, device=device))


@pytest.mark.requires_accelerator
def test_local_token_dispatcher_dispatch_unstable_groups_tokens(device: torch.device) -> None:
    """Test that an unstable permutation still groups tokens by expert."""
    dispatcher = LocalTokenDispatcher(4, stable_permutation=False)
    tokens, expert_scores, expert_indices = _dispatch_inputs(device, (4,))

    dispatched, offs, state = dispatcher.dispatch(tokens, expert_scores, expert_indices)

    expected_offs = torch.bincount(expert_indices.reshape(-1), minlength=4).cumsum(0).to(torch.int32)
    assert torch.equal(offs, expected_offs)

    top_k = expert_indices.size(-1)
    flat_indices = expert_indices.reshape(-1).tolist()
    starts = [0, *offs[:-1].tolist()]
    for expert, start in enumerate(starts):
        end = offs[expert].item()
        expected_tokens = sorted(route // top_k for route, selected in enumerate(flat_indices) if selected == expert)
        assert sorted(state.token_indices[start:end].tolist()) == expected_tokens
    assert torch.equal(dispatched, tokens.index_select(0, state.token_indices))


@pytest.mark.requires_accelerator
@pytest.mark.parametrize("num_experts", [256, 257], ids=["uint8", "int16"])
def test_local_token_dispatcher_dispatch_supports_sort_dtype_boundary(
    num_experts: int,
    device: torch.device,
) -> None:
    """Test that dispatch handles the largest expert ID at each key-dtype boundary."""
    dispatcher = LocalTokenDispatcher(num_experts, stable_permutation=True)
    tokens = torch.arange(3, dtype=torch.float32, device=device).unsqueeze(1)
    expert_scores = torch.tensor([0.1, 0.2, 0.3], device=device)
    expert_indices = torch.tensor([[0], [num_experts - 1], [num_experts - 1]], dtype=torch.int64, device=device)

    dispatched, offs, state = dispatcher.dispatch(tokens, expert_scores, expert_indices)

    assert offs.dtype is torch.int32
    assert offs[0].item() == 1
    assert offs[-1].item() == 3
    assert torch.equal(dispatched[:, 0], torch.tensor([0.0, 1.0, 2.0], device=device))
    assert torch.equal(state.token_indices, torch.tensor([0, 1, 2], dtype=torch.int32, device=device))


# ------------------------------------------------------------------------------
# combine
# ------------------------------------------------------------------------------


def _combine_reference(
    expert_scores: torch.Tensor,
    tokens: torch.Tensor,
    token_indices: torch.Tensor,
    input_leading_dims: torch.Size,
) -> torch.Tensor:
    """Compute the expected weighted reduction with index_add."""
    weighted_tokens = (expert_scores.unsqueeze(-1) * tokens).to(tokens.dtype)
    reduced_tokens = weighted_tokens.new_zeros((input_leading_dims.numel(), tokens.size(-1)))
    reduced_tokens.index_add_(0, token_indices.to(torch.int64), weighted_tokens)
    return reduced_tokens.reshape(*input_leading_dims, tokens.size(-1))


@pytest.mark.parametrize(
    "input_leading_dims",
    [pytest.param(torch.Size([4]), id="1d"), pytest.param(torch.Size([2, 2]), id="2d")],
)
def test_local_token_dispatcher_combine_matches_reference(
    input_leading_dims: torch.Size,
    device: torch.device,
) -> None:
    """Test that combine weights and sums expert outputs in token order."""
    dispatcher = LocalTokenDispatcher(4, stable_permutation=True)
    token_indices = torch.tensor([0, 2, 1, 3, 0, 2, 3, 1], dtype=torch.int32, device=device)
    expert_scores = torch.rand(token_indices.numel(), device=device)
    expert_tokens = torch.randn(token_indices.numel(), 3, device=device)
    state = TokenDispatchState(
        expert_scores=expert_scores,
        token_indices=token_indices,
        input_leading_dims=input_leading_dims,
    )

    combined = dispatcher.combine(expert_tokens, state)

    expected = _combine_reference(expert_scores, expert_tokens, token_indices, input_leading_dims)
    assert combined.shape == (*input_leading_dims, 3)
    torch.testing.assert_close(combined, expected)


def test_local_token_dispatcher_combine_sums_multiple_routes(device: torch.device) -> None:
    """Test that routes for the same token are summed after weighting."""
    dispatcher = LocalTokenDispatcher(1, stable_permutation=True)
    state = TokenDispatchState(
        expert_scores=torch.tensor([0.5, 0.25], device=device),
        token_indices=torch.tensor([0, 0], dtype=torch.int32, device=device),
        input_leading_dims=torch.Size([1]),
    )
    expert_tokens = torch.tensor([[2.0], [4.0]], device=device)

    combined = dispatcher.combine(expert_tokens, state)

    assert torch.equal(combined, torch.tensor([[2.0]], device=device))


def test_local_token_dispatcher_combine_preserves_expert_output_dtype(device: torch.device) -> None:
    """Test that the output dtype follows the expert outputs, not the scores."""
    dispatcher = LocalTokenDispatcher(2, stable_permutation=True)
    token_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
    expert_scores = torch.tensor([0.5, 0.25, 0.125, 0.75], dtype=torch.float32, device=device)
    expert_tokens = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.bfloat16, device=device)
    state = TokenDispatchState(
        expert_scores=expert_scores,
        token_indices=token_indices,
        input_leading_dims=torch.Size([2]),
    )

    combined = dispatcher.combine(expert_tokens, state)

    expected = _combine_reference(expert_scores, expert_tokens, token_indices, state.input_leading_dims)
    assert combined.dtype is torch.bfloat16
    torch.testing.assert_close(combined, expected)


def test_local_token_dispatcher_combine_handles_empty_tokens(device: torch.device) -> None:
    """Test that combine returns an empty reduction for an empty state."""
    dispatcher = LocalTokenDispatcher(4, stable_permutation=True)
    state = TokenDispatchState(
        expert_scores=torch.empty(0, device=device),
        token_indices=torch.empty(0, dtype=torch.int32, device=device),
        input_leading_dims=torch.Size([0, 3]),
    )
    expert_tokens = torch.empty(0, 2, device=device)

    combined = dispatcher.combine(expert_tokens, state)

    assert combined.shape == (0, 3, 2)
    assert combined.dtype is expert_tokens.dtype


# ------------------------------------------------------------------------------
# dispatch and combine
# ------------------------------------------------------------------------------


@pytest.mark.requires_accelerator
def test_local_token_dispatcher_round_trip_matches_weighted_sum(device: torch.device) -> None:
    """Test that combining dispatched expert outputs reproduces the weighted sum."""
    dispatcher = LocalTokenDispatcher(4, stable_permutation=True)
    tokens, expert_scores, expert_indices = _dispatch_inputs(device, (4,))
    dispatched, _, state = dispatcher.dispatch(tokens, expert_scores, expert_indices)
    expert_outputs = torch.randn(dispatched.size(0), 2, device=device)

    combined = dispatcher.combine(expert_outputs, state)

    weighted_tokens = state.expert_scores.unsqueeze(-1) * expert_outputs
    expected = torch.zeros(tokens.size(0), 2, device=device)
    expected.index_add_(0, state.token_indices.to(torch.int64), weighted_tokens)
    assert combined.shape == (tokens.size(0), 2)
    torch.testing.assert_close(combined, expected)


# ------------------------------------------------------------------------------
# extra_repr
# ------------------------------------------------------------------------------


def test_local_token_dispatcher_extra_repr_reports_configuration() -> None:
    """Test that extra_repr reports the dispatcher configuration."""
    dispatcher = LocalTokenDispatcher(4, stable_permutation=False)

    assert dispatcher.extra_repr() == "num_experts=4, stable_permutation=False"
