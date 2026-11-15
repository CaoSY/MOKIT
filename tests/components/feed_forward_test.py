# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.components.feed_forward`."""

import math
from collections.abc import Callable
from functools import partial
from typing import cast

import pytest
import torch
import torch.nn.functional as F

from mokit.components.feed_forward import (
    FeedForward,
    GatedFeedForward,
    GatedGroupedFeedForward,
    GroupedFeedForward,
)
from mokit.components.linear import GroupedLinear
from mokit.types import DType

# ==============================================================================
# FeedForward
# ==============================================================================


# ------------------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------------------


def test_feed_forward_init_stores_dimensions_and_projections() -> None:
    """Test that __init__ stores the dimensions and creates both projections."""
    module = FeedForward(3, 5, 4)

    assert module.in_features == 3
    assert module.hidden_features == 5
    assert module.out_features == 4
    assert isinstance(module.up_proj, torch.nn.Linear)
    assert module.up_proj.in_features == 3
    assert module.up_proj.out_features == 5
    assert isinstance(module.down_proj, torch.nn.Linear)
    assert module.down_proj.in_features == 5
    assert module.down_proj.out_features == 4


def test_feed_forward_init_registers_default_activation() -> None:
    """Test that the default activation is a registered SiLU module."""
    module = FeedForward(3, 5, 4)

    assert isinstance(module.activation, torch.nn.SiLU)
    assert dict(module.named_modules())["activation"] is module.activation


def test_feed_forward_init_creates_distinct_default_activations() -> None:
    """Test that each module owns its activation instance."""
    first = FeedForward(3, 5, 4)
    second = FeedForward(3, 5, 4)

    assert first.activation is not second.activation


def test_feed_forward_init_builds_activation_from_factory() -> None:
    """Test that activation accepts a module type."""
    module = FeedForward(3, 5, 4, activation=torch.nn.GELU)

    assert isinstance(module.activation, torch.nn.GELU)


def test_feed_forward_init_configures_activation_from_partial() -> None:
    """Test that a partial factory can configure the activation."""
    module = FeedForward(3, 5, 4, activation=partial(torch.nn.SiLU, inplace=True))

    assert isinstance(module.activation, torch.nn.SiLU)
    assert module.activation.inplace


def test_feed_forward_init_default_bias_applies_to_both_projections() -> None:
    """Test that the default bias appears on both projections."""
    module = FeedForward(3, 5, 4)

    assert module.up_proj.bias is not None
    assert module.down_proj.bias is not None


def test_feed_forward_init_no_bias_removes_both_projection_biases() -> None:
    """Test that bias=False removes the bias from both projections."""
    module = FeedForward(3, 5, 4, bias=False)

    assert module.up_proj.bias is None
    assert module.down_proj.bias is None
    assert sorted(module.state_dict()) == ["down_proj.weight", "up_proj.weight"]


@pytest.mark.parametrize(
    "bias",
    [(True, False), (False, True), (False, False)],
    ids=["up_only", "down_only", "neither"],
)
def test_feed_forward_init_applies_tuple_bias_per_projection(bias: tuple[bool, bool]) -> None:
    """Test that a two-element tuple sets each projection's bias separately."""
    module = FeedForward(3, 5, 4, bias=bias)

    assert (module.up_proj.bias is not None) is bias[0]
    assert (module.down_proj.bias is not None) is bias[1]
    assert ("down_proj.bias" in module.state_dict()) is bias[1]


@pytest.mark.parametrize("bias", [(), (True,), (True, False, True)], ids=["empty", "one", "three"])
def test_feed_forward_init_rejects_bias_tuple_of_wrong_length(bias: tuple[bool, ...]) -> None:
    """Test that a tuple bias must have exactly two elements."""
    with pytest.raises(ValueError, match="unpack"):
        FeedForward(3, 5, 4, bias=cast(tuple[bool, bool], bias))


def test_feed_forward_init_respects_dtype() -> None:
    """Test that __init__ uses the requested parameter dtype."""
    module = FeedForward(3, 5, 4, dtype=torch.float64)

    assert module.up_proj.weight.dtype is torch.float64
    assert module.down_proj.weight.dtype is torch.float64


def test_feed_forward_init_uses_linear_initialization_bounds() -> None:
    """Test that each projection uses its own nn.Linear initialization bound."""
    module = FeedForward(8, 16, 4)

    assert module.up_proj.weight.abs().max().item() <= 1 / math.sqrt(8)
    assert module.down_proj.weight.abs().max().item() <= 1 / math.sqrt(16)


# ------------------------------------------------------------------------------
# forward
# ------------------------------------------------------------------------------


def _reference_forward(
    module: FeedForward,
    input: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute the expected result with F.linear and a functional activation."""
    hidden = activation(F.linear(input, module.up_proj.weight, module.up_proj.bias))
    return F.linear(hidden, module.down_proj.weight, module.down_proj.bias)


def _assert_projection_gradients(projection: torch.nn.Linear, learn_bias: bool) -> None:
    """Assert that a projection received gradients matching its parameters."""
    assert projection.weight.grad is not None
    assert projection.weight.grad.shape == projection.weight.shape
    if learn_bias:
        assert projection.bias is not None
        assert projection.bias.grad is not None
        assert projection.bias.grad.shape == projection.bias.shape
    else:
        assert projection.bias is None


@pytest.mark.parametrize("input_shape", [(3,), (2, 3), (2, 5, 3)], ids=["1d", "2d", "3d"])
def test_feed_forward_forward_matches_reference(input_shape: tuple[int, ...]) -> None:
    """Test that forward matches linear projections and SiLU."""
    torch.manual_seed(0)
    module = FeedForward(3, 5, 4)
    input = torch.randn(input_shape)

    output = module(input)
    expected = _reference_forward(module, input, F.silu)

    assert output.shape == (*input_shape[:-1], 4)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "bias",
    [True, False, (True, False), (False, True)],
    ids=["both", "neither", "up_only", "down_only"],
)
def test_feed_forward_forward_matches_reference_for_bias_configuration(bias: bool | tuple[bool, bool]) -> None:
    """Test that forward matches the reference for every bias configuration."""
    torch.manual_seed(0)
    module = FeedForward(3, 5, 4, bias=bias)
    input = torch.randn(2, 3)

    torch.testing.assert_close(module(input), _reference_forward(module, input, F.silu))


@pytest.mark.parametrize(
    ("factory", "activation"),
    [
        pytest.param(torch.nn.GELU, F.gelu, id="gelu"),
        pytest.param(partial(torch.nn.SiLU, inplace=True), F.silu, id="inplace_silu"),
    ],
)
def test_feed_forward_forward_uses_configured_activation(
    factory: Callable[..., torch.nn.Module],
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Test that forward applies the configured activation."""
    torch.manual_seed(0)
    module = FeedForward(3, 5, 4, activation=factory)
    input = torch.randn(2, 3)

    torch.testing.assert_close(module(input), _reference_forward(module, input, activation))


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.bfloat16],
    ids=["fp32", "fp64", "bf16"],
)
def test_feed_forward_forward_preserves_dtype(dtype: torch.dtype) -> None:
    """Test that forward preserves the parameter dtype."""
    module = FeedForward(3, 5, 4, dtype=dtype)
    input = torch.randn(2, 3, dtype=dtype)

    assert module(input).dtype is dtype


def test_feed_forward_forward_handles_empty_batch() -> None:
    """Test that an empty input produces an empty output."""
    module = FeedForward(3, 5, 4)

    assert module(torch.empty(0, 3)).shape == (0, 4)


def test_feed_forward_forward_rejects_wrong_input_width() -> None:
    """Test that up_proj rejects an input whose last dimension is not in_features."""
    module = FeedForward(3, 5, 4)

    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        module(torch.randn(2, 7))


@pytest.mark.parametrize(
    "bias",
    [True, False, (True, False), (False, True)],
    ids=["both", "neither", "up_only", "down_only"],
)
def test_feed_forward_forward_backward_computes_gradients(bias: bool | tuple[bool, bool]) -> None:
    """Test that forward supports autograd for the input and both projections."""
    torch.manual_seed(0)
    module = FeedForward(3, 5, 4, bias=bias)
    input = torch.randn(2, 3, requires_grad=True)

    output = module(input)
    output.backward(torch.ones_like(output))

    up_bias, down_bias = (bias, bias) if isinstance(bias, bool) else bias
    assert input.grad is not None
    assert input.grad.shape == input.shape
    _assert_projection_gradients(module.up_proj, up_bias)
    _assert_projection_gradients(module.down_proj, down_bias)


def test_feed_forward_forward_backward_reaches_activation_parameters() -> None:
    """Test that autograd reaches parameters of a parameterized activation."""
    torch.manual_seed(0)
    module = FeedForward(3, 5, 4, activation=partial(torch.nn.PReLU, num_parameters=5))
    activation = module.activation
    assert isinstance(activation, torch.nn.PReLU)

    output = module(torch.randn(2, 3))
    output.backward(torch.ones_like(output))

    assert activation.weight.grad is not None


# ------------------------------------------------------------------------------
# extra_repr
# ------------------------------------------------------------------------------


def test_feed_forward_extra_repr_reports_configuration() -> None:
    """Test that extra_repr reports the module configuration."""
    module = FeedForward(3, 5, 4)

    assert module.extra_repr() == "in_features=3, hidden_features=5, out_features=4"


# ==============================================================================
# GatedFeedForward
# ==============================================================================


# ------------------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------------------


def test_gated_feed_forward_init_stores_dimensions_and_projections() -> None:
    """Test that __init__ stores the dimensions and creates the fused projection."""
    module = GatedFeedForward(3, 5, 4)

    assert module.in_features == 3
    assert module.hidden_features == 5
    assert module.out_features == 4
    assert isinstance(module.gate_up_proj, torch.nn.Linear)
    assert module.gate_up_proj.in_features == 3
    assert module.gate_up_proj.out_features == 10
    assert isinstance(module.down_proj, torch.nn.Linear)
    assert module.down_proj.in_features == 5
    assert module.down_proj.out_features == 4


def test_gated_feed_forward_init_registers_default_activation() -> None:
    """Test that the default activation is a registered SiLU module."""
    module = GatedFeedForward(3, 5, 4)

    assert isinstance(module.activation, torch.nn.SiLU)
    assert dict(module.named_modules())["activation"] is module.activation


def test_gated_feed_forward_init_creates_distinct_default_activations() -> None:
    """Test that each module owns its activation instance."""
    first = GatedFeedForward(3, 5, 4)
    second = GatedFeedForward(3, 5, 4)

    assert first.activation is not second.activation


def test_gated_feed_forward_init_builds_activation_from_factory() -> None:
    """Test that activation accepts a module type."""
    module = GatedFeedForward(3, 5, 4, activation=torch.nn.GELU)

    assert isinstance(module.activation, torch.nn.GELU)


def test_gated_feed_forward_init_configures_activation_from_partial() -> None:
    """Test that a partial factory can configure the activation."""
    module = GatedFeedForward(3, 5, 4, activation=partial(torch.nn.SiLU, inplace=True))

    assert isinstance(module.activation, torch.nn.SiLU)
    assert module.activation.inplace


def test_gated_feed_forward_init_default_bias_applies_to_both_projections() -> None:
    """Test that the default bias appears on both projections."""
    module = GatedFeedForward(3, 5, 4)

    assert module.gate_up_proj.bias is not None
    assert module.gate_up_proj.bias.shape == (10,)
    assert module.down_proj.bias is not None


def test_gated_feed_forward_init_no_bias_removes_both_projection_biases() -> None:
    """Test that bias=False removes the bias from both projections."""
    module = GatedFeedForward(3, 5, 4, bias=False)

    assert module.gate_up_proj.bias is None
    assert module.down_proj.bias is None
    assert sorted(module.state_dict()) == ["down_proj.weight", "gate_up_proj.weight"]


@pytest.mark.parametrize(
    "bias",
    [(True, False), (False, True), (False, False)],
    ids=["gate_up_only", "down_only", "neither"],
)
def test_gated_feed_forward_init_applies_tuple_bias_per_projection(bias: tuple[bool, bool]) -> None:
    """Test that a two-element tuple sets each projection's bias separately."""
    module = GatedFeedForward(3, 5, 4, bias=bias)

    assert (module.gate_up_proj.bias is not None) is bias[0]
    assert (module.down_proj.bias is not None) is bias[1]


@pytest.mark.parametrize("bias", [(), (True,), (True, False, True)], ids=["empty", "one", "three"])
def test_gated_feed_forward_init_rejects_bias_tuple_of_wrong_length(bias: tuple[bool, ...]) -> None:
    """Test that a tuple bias must have exactly two elements."""
    with pytest.raises(ValueError, match="unpack"):
        GatedFeedForward(3, 5, 4, bias=cast(tuple[bool, bool], bias))


def test_gated_feed_forward_init_respects_dtype() -> None:
    """Test that __init__ uses the requested parameter dtype."""
    module = GatedFeedForward(3, 5, 4, dtype=torch.float64)

    assert module.gate_up_proj.weight.dtype is torch.float64
    assert module.down_proj.weight.dtype is torch.float64


def test_gated_feed_forward_init_uses_linear_initialization_bounds() -> None:
    """Test that each projection uses its own nn.Linear initialization bound."""
    module = GatedFeedForward(8, 16, 4)

    assert module.gate_up_proj.weight.abs().max().item() <= 1 / math.sqrt(8)
    assert module.down_proj.weight.abs().max().item() <= 1 / math.sqrt(16)


# ------------------------------------------------------------------------------
# forward
# ------------------------------------------------------------------------------


def _reference_gated_forward(
    module: GatedFeedForward,
    input: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute the expected result by splitting the fused projection."""
    hidden = module.hidden_features
    weight = module.gate_up_proj.weight
    bias = module.gate_up_proj.bias
    gate = F.linear(input, weight[:hidden], None if bias is None else bias[:hidden])
    up = F.linear(input, weight[hidden:], None if bias is None else bias[hidden:])
    return F.linear(activation(gate) * up, module.down_proj.weight, module.down_proj.bias)


@pytest.mark.parametrize("input_shape", [(3,), (2, 3), (2, 5, 3)], ids=["1d", "2d", "3d"])
def test_gated_feed_forward_forward_matches_reference(input_shape: tuple[int, ...]) -> None:
    """Test that forward matches separate gate and up projections."""
    torch.manual_seed(0)
    module = GatedFeedForward(3, 5, 4)
    input = torch.randn(input_shape)

    output = module(input)
    expected = _reference_gated_forward(module, input, F.silu)

    assert output.shape == (*input_shape[:-1], 4)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "bias",
    [True, False, (True, False), (False, True)],
    ids=["both", "neither", "gate_up_only", "down_only"],
)
def test_gated_feed_forward_forward_matches_reference_for_bias_configuration(
    bias: bool | tuple[bool, bool],
) -> None:
    """Test that forward matches the reference for every bias configuration."""
    torch.manual_seed(0)
    module = GatedFeedForward(3, 5, 4, bias=bias)
    input = torch.randn(2, 3)

    torch.testing.assert_close(module(input), _reference_gated_forward(module, input, F.silu))


@pytest.mark.parametrize(
    ("factory", "activation"),
    [
        pytest.param(torch.nn.GELU, F.gelu, id="gelu"),
        pytest.param(partial(torch.nn.SiLU, inplace=True), F.silu, id="inplace_silu"),
    ],
)
def test_gated_feed_forward_forward_uses_configured_activation(
    factory: Callable[..., torch.nn.Module],
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Test that forward applies the configured activation to the gate half."""
    torch.manual_seed(0)
    module = GatedFeedForward(3, 5, 4, activation=factory)
    input = torch.randn(2, 3)

    torch.testing.assert_close(module(input), _reference_gated_forward(module, input, activation))


def test_gated_feed_forward_forward_activates_the_first_half() -> None:
    """Test that the activation applies to the first half of the fused output."""
    module = GatedFeedForward(1, 1, 1, bias=False)
    with torch.no_grad():
        module.gate_up_proj.weight.copy_(torch.tensor([[1.0], [2.0]]))
        module.down_proj.weight.fill_(1.0)

    output = module(torch.ones(1, 1))

    expected = F.silu(torch.ones(1, 1)) * 2.0
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.bfloat16],
    ids=["fp32", "fp64", "bf16"],
)
def test_gated_feed_forward_forward_preserves_dtype(dtype: torch.dtype) -> None:
    """Test that forward preserves the parameter dtype."""
    module = GatedFeedForward(3, 5, 4, dtype=dtype)
    input = torch.randn(2, 3, dtype=dtype)

    assert module(input).dtype is dtype


def test_gated_feed_forward_forward_handles_empty_batch() -> None:
    """Test that an empty input produces an empty output."""
    module = GatedFeedForward(3, 5, 4)

    assert module(torch.empty(0, 3)).shape == (0, 4)


def test_gated_feed_forward_forward_rejects_wrong_input_width() -> None:
    """Test that the fused projection rejects an input whose last dimension is not in_features."""
    module = GatedFeedForward(3, 5, 4)

    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        module(torch.randn(2, 7))


@pytest.mark.parametrize(
    "bias",
    [True, False, (True, False), (False, True)],
    ids=["both", "neither", "gate_up_only", "down_only"],
)
def test_gated_feed_forward_forward_backward_computes_gradients(bias: bool | tuple[bool, bool]) -> None:
    """Test that forward supports autograd for the input and both projections."""
    torch.manual_seed(0)
    module = GatedFeedForward(3, 5, 4, bias=bias)
    input = torch.randn(2, 3, requires_grad=True)

    output = module(input)
    output.backward(torch.ones_like(output))

    gate_up_bias, down_bias = (bias, bias) if isinstance(bias, bool) else bias
    assert input.grad is not None
    assert input.grad.shape == input.shape
    assert module.gate_up_proj.weight.grad is not None
    assert module.gate_up_proj.weight.grad.shape == module.gate_up_proj.weight.shape
    assert module.down_proj.weight.grad is not None
    assert module.down_proj.weight.grad.shape == module.down_proj.weight.shape
    if gate_up_bias:
        assert module.gate_up_proj.bias is not None
        assert module.gate_up_proj.bias.grad is not None
        assert module.gate_up_proj.bias.grad.shape == module.gate_up_proj.bias.shape
    else:
        assert module.gate_up_proj.bias is None
    if down_bias:
        assert module.down_proj.bias is not None
        assert module.down_proj.bias.grad is not None
        assert module.down_proj.bias.grad.shape == module.down_proj.bias.shape
    else:
        assert module.down_proj.bias is None


def test_gated_feed_forward_forward_backward_reaches_activation_parameters() -> None:
    """Test that autograd reaches parameters of a parameterized activation."""
    torch.manual_seed(0)
    module = GatedFeedForward(3, 5, 4, activation=partial(torch.nn.PReLU, num_parameters=5))
    activation = module.activation
    assert isinstance(activation, torch.nn.PReLU)

    output = module(torch.randn(2, 3))
    output.backward(torch.ones_like(output))

    assert activation.weight.grad is not None


# ------------------------------------------------------------------------------
# extra_repr
# ------------------------------------------------------------------------------


def test_gated_feed_forward_extra_repr_reports_configuration() -> None:
    """Test that extra_repr reports the module configuration."""
    module = GatedFeedForward(3, 5, 4)

    assert module.extra_repr() == "in_features=3, hidden_features=5, out_features=4"


# ==============================================================================
# GroupedFeedForward
# ==============================================================================


# ------------------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------------------


def test_grouped_feed_forward_init_stores_dimensions_and_projections() -> None:
    """Test that __init__ stores the dimensions and creates both projections."""
    module = GroupedFeedForward(3, 8, 16, 4)

    assert module.num_groups == 3
    assert module.in_features == 8
    assert module.hidden_features == 16
    assert module.out_features == 4
    assert isinstance(module.up_proj, GroupedLinear)
    assert module.up_proj.num_groups == 3
    assert module.up_proj.in_features == 8
    assert module.up_proj.out_features == 16
    assert isinstance(module.down_proj, GroupedLinear)
    assert module.down_proj.num_groups == 3
    assert module.down_proj.in_features == 16
    assert module.down_proj.out_features == 4


def test_grouped_feed_forward_init_registers_default_activation() -> None:
    """Test that the default activation is a registered SiLU module."""
    module = GroupedFeedForward(3, 8, 16, 4)

    assert isinstance(module.activation, torch.nn.SiLU)
    assert dict(module.named_modules())["activation"] is module.activation


def test_grouped_feed_forward_init_creates_distinct_default_activations() -> None:
    """Test that each module owns its activation instance."""
    first = GroupedFeedForward(3, 8, 16, 4)
    second = GroupedFeedForward(3, 8, 16, 4)

    assert first.activation is not second.activation


def test_grouped_feed_forward_init_builds_activation_from_factory() -> None:
    """Test that activation accepts a module type."""
    module = GroupedFeedForward(3, 8, 16, 4, activation=torch.nn.GELU)

    assert isinstance(module.activation, torch.nn.GELU)


def test_grouped_feed_forward_init_configures_activation_from_partial() -> None:
    """Test that a partial factory can configure the activation."""
    module = GroupedFeedForward(3, 8, 16, 4, activation=partial(torch.nn.SiLU, inplace=True))

    assert isinstance(module.activation, torch.nn.SiLU)
    assert module.activation.inplace


def test_grouped_feed_forward_init_default_bias_applies_to_both_projections() -> None:
    """Test that the default bias appears on both projections."""
    module = GroupedFeedForward(3, 8, 16, 4)

    assert module.up_proj.bias is not None
    assert module.up_proj.bias.shape == (48,)
    assert module.down_proj.bias is not None
    assert module.down_proj.bias.shape == (12,)


def test_grouped_feed_forward_init_no_bias_removes_both_projection_biases() -> None:
    """Test that bias=False removes the bias from both projections."""
    module = GroupedFeedForward(3, 8, 16, 4, bias=False)

    assert module.up_proj.bias is None
    assert module.down_proj.bias is None
    assert sorted(module.state_dict()) == ["down_proj.weight", "up_proj.weight"]


@pytest.mark.parametrize(
    "bias",
    [(True, False), (False, True), (False, False)],
    ids=["up_only", "down_only", "neither"],
)
def test_grouped_feed_forward_init_applies_tuple_bias_per_projection(bias: tuple[bool, bool]) -> None:
    """Test that a two-element tuple sets each projection's bias separately."""
    module = GroupedFeedForward(3, 8, 16, 4, bias=bias)

    assert (module.up_proj.bias is not None) is bias[0]
    assert (module.down_proj.bias is not None) is bias[1]
    assert ("down_proj.bias" in module.state_dict()) is bias[1]


@pytest.mark.parametrize("bias", [(), (True,), (True, False, True)], ids=["empty", "one", "three"])
def test_grouped_feed_forward_init_rejects_bias_tuple_of_wrong_length(bias: tuple[bool, ...]) -> None:
    """Test that a tuple bias must have exactly two elements."""
    with pytest.raises(ValueError, match="unpack"):
        GroupedFeedForward(3, 8, 16, 4, bias=cast(tuple[bool, bool], bias))


def test_grouped_feed_forward_init_respects_dtype() -> None:
    """Test that __init__ uses the requested parameter dtype."""
    module = GroupedFeedForward(3, 8, 16, 4, dtype=torch.float64)

    assert module.up_proj.weight.dtype is torch.float64
    assert module.down_proj.weight.dtype is torch.float64


def test_grouped_feed_forward_init_uses_linear_initialization_bounds() -> None:
    """Test that each projection uses its own initialization bound."""
    module = GroupedFeedForward(3, 8, 16, 4)

    assert module.up_proj.weight.abs().max().item() <= 1 / math.sqrt(8)
    assert module.down_proj.weight.abs().max().item() <= 1 / math.sqrt(16)


# ------------------------------------------------------------------------------
# forward
# ------------------------------------------------------------------------------


def _reference_grouped_feed_forward(
    module: GroupedFeedForward,
    input: torch.Tensor,
    offs: torch.Tensor | None,
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute the expected result with independent per-group F.linear calls."""
    num_groups = module.num_groups
    up_weight = module.up_proj.weight.reshape(num_groups, module.hidden_features, module.in_features)
    up_bias = None if module.up_proj.bias is None else module.up_proj.bias.reshape(num_groups, module.hidden_features)
    down_weight = module.down_proj.weight.reshape(num_groups, module.out_features, module.hidden_features)
    down_bias = (
        None if module.down_proj.bias is None else module.down_proj.bias.reshape(num_groups, module.out_features)
    )

    def run(x_group: torch.Tensor, group: int) -> torch.Tensor:
        hidden = activation(F.linear(x_group, up_weight[group], None if up_bias is None else up_bias[group]))
        return F.linear(hidden, down_weight[group], None if down_bias is None else down_bias[group])

    if input.ndim == 3:
        return torch.stack([run(input[group], group) for group in range(num_groups)])

    assert offs is not None
    starts = [0, *offs[:-1].tolist()]
    return torch.cat(
        [run(input[start:end], group) for group, (start, end) in enumerate(zip(starts, offs.tolist(), strict=True))]
    )


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_grouped_feed_forward_forward_2d_matches_reference(bias: bool) -> None:
    """Test that 2D forward matches independent per-group feed-forward blocks."""
    torch.manual_seed(0)
    module = GroupedFeedForward(3, 8, 16, 4, bias=bias)
    input = torch.randn(10, 8)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    output = module(input, offs=offs)
    expected = _reference_grouped_feed_forward(module, input, offs, F.silu)

    assert output.shape == (10, 4)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_grouped_feed_forward_forward_3d_matches_reference(bias: bool) -> None:
    """Test that 3D forward matches independent per-group feed-forward blocks."""
    torch.manual_seed(0)
    module = GroupedFeedForward(3, 8, 16, 4, bias=bias)
    input = torch.randn(3, 4, 8)

    output = module(input)
    expected = _reference_grouped_feed_forward(module, input, None, F.silu)

    assert output.shape == (3, 4, 4)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "bias",
    [(True, False), (False, True)],
    ids=["up_only", "down_only"],
)
def test_grouped_feed_forward_forward_matches_reference_for_bias_configuration(bias: tuple[bool, bool]) -> None:
    """Test that forward matches the reference for mixed bias configurations."""
    torch.manual_seed(0)
    module = GroupedFeedForward(3, 8, 16, 4, bias=bias)
    input = torch.randn(10, 8)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    torch.testing.assert_close(
        module(input, offs=offs),
        _reference_grouped_feed_forward(module, input, offs, F.silu),
    )


@pytest.mark.parametrize(
    ("factory", "activation"),
    [
        pytest.param(torch.nn.GELU, F.gelu, id="gelu"),
        pytest.param(partial(torch.nn.SiLU, inplace=True), F.silu, id="inplace_silu"),
    ],
)
def test_grouped_feed_forward_forward_uses_configured_activation(
    factory: Callable[..., torch.nn.Module],
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Test that forward applies the configured activation."""
    torch.manual_seed(0)
    module = GroupedFeedForward(3, 8, 16, 4, activation=factory)
    input = torch.randn(10, 8)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    torch.testing.assert_close(
        module(input, offs=offs),
        _reference_grouped_feed_forward(module, input, offs, activation),
    )


@pytest.mark.parametrize(
    "offs_values",
    [[0, 5, 10], [2, 2, 10], [2, 10, 10]],
    ids=["first_empty", "middle_empty", "last_empty"],
)
def test_grouped_feed_forward_forward_supports_empty_groups(offs_values: list[int]) -> None:
    """Test that groups with no rows are handled correctly."""
    torch.manual_seed(0)
    module = GroupedFeedForward(3, 8, 16, 4)
    input = torch.randn(10, 8)
    offs = torch.tensor(offs_values, dtype=torch.int32)

    torch.testing.assert_close(
        module(input, offs=offs),
        _reference_grouped_feed_forward(module, input, offs, F.silu),
    )


def test_grouped_feed_forward_forward_handles_zero_rows_per_group() -> None:
    """Test that 3D input with zero rows per group produces an empty output."""
    module = GroupedFeedForward(3, 8, 16, 4)

    output = module(torch.empty(3, 0, 8))

    assert output.shape == (3, 0, 4)


def test_grouped_feed_forward_forward_handles_empty_input() -> None:
    """Test that an empty 2D input produces an empty output."""
    module = GroupedFeedForward(3, 8, 16, 4)
    offs = torch.tensor([0, 0, 0], dtype=torch.int32)

    output = module(torch.empty(0, 8), offs=offs)

    assert output.shape == (0, 4)


@pytest.mark.parametrize(
    "offs_values",
    [[2, 5, 8], [2, 5, 5]],
    ids=["normal_tail", "empty_last_group"],
)
def test_grouped_feed_forward_forward_matches_reference_before_tail(offs_values: list[int]) -> None:
    """Test that rows before offs[-1] match the per-group reference."""
    torch.manual_seed(0)
    module = GroupedFeedForward(3, 8, 16, 4)
    input = torch.randn(10, 8)
    offs = torch.tensor(offs_values, dtype=torch.int32)

    output = module(input, offs=offs)
    expected = _reference_grouped_feed_forward(module, input, offs, F.silu)

    assert output.shape == (10, 4)
    torch.testing.assert_close(output[: offs[-1].item()], expected)


def test_grouped_feed_forward_forward_handles_all_rows_as_tail() -> None:
    """Test that offsets ending at zero leave every output row unspecified."""
    module = GroupedFeedForward(3, 8, 16, 4)
    input = torch.randn(10, 8)
    offs = torch.tensor([0, 0, 0], dtype=torch.int32)

    output = module(input, offs=offs)

    assert output.shape == (10, 4)


@pytest.mark.parametrize(
    ("dtype", "in_features", "hidden_features"),
    [
        pytest.param(torch.float32, 8, 16, id="fp32"),
        pytest.param(torch.bfloat16, 8, 16, id="bf16"),
    ],
)
def test_grouped_feed_forward_forward_preserves_dtype(
    dtype: torch.dtype,
    in_features: int,
    hidden_features: int,
) -> None:
    """Test that forward preserves the parameter dtype."""
    module = GroupedFeedForward(3, in_features, hidden_features, 4, dtype=dtype)
    input = torch.randn(10, in_features, dtype=dtype)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    assert module(input, offs=offs).dtype is dtype


@pytest.mark.parametrize(
    "out_dtype",
    [
        pytest.param(torch.bfloat16, id="scalar"),
        pytest.param((torch.bfloat16, torch.bfloat16), id="pair"),
        pytest.param((torch.bfloat16, None), id="down_none"),
        pytest.param((None, torch.bfloat16), id="up_none"),
    ],
)
def test_grouped_feed_forward_forward_accepts_valid_out_dtype(out_dtype: DType | tuple[DType, DType]) -> None:
    """Test that a valid out_dtype preserves the output dtype and values."""
    module = GroupedFeedForward(3, 8, 16, 4, dtype=torch.bfloat16)
    input = torch.randn(10, 8, dtype=torch.bfloat16)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    expected = module(input, offs=offs)
    output = module(input, offs=offs, out_dtype=out_dtype)

    assert output.dtype is torch.bfloat16
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "out_dtype",
    [
        pytest.param((torch.float32, torch.bfloat16), id="up_mismatch"),
        pytest.param((torch.bfloat16, torch.float32), id="down_mismatch"),
    ],
)
def test_grouped_feed_forward_forward_rejects_mismatched_out_dtype(
    out_dtype: DType | tuple[DType, DType],
) -> None:
    """
    Test that a mismatched out_dtype is rejected by the current GEMM kernels.

    This is a known constraint of the grouped GEMM kernels. If the constraint
    is relaxed in the future, this test will fail and the parameter should be
    moved to ``test_grouped_feed_forward_forward_accepts_valid_out_dtype``.
    """
    module = GroupedFeedForward(3, 8, 16, 4, dtype=torch.bfloat16)
    input = torch.randn(10, 8, dtype=torch.bfloat16)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="output dtype must match"):
        module(input, offs=offs, out_dtype=out_dtype)


def test_grouped_feed_forward_forward_rejects_wrong_input_width() -> None:
    """Test that the up projection rejects an input whose last dimension is not in_features."""
    module = GroupedFeedForward(3, 8, 16, 4)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="contraction dimension"):
        module(torch.randn(10, 12), offs=offs)


def test_grouped_feed_forward_forward_rejects_2d_input_without_offs() -> None:
    """Test that 2D input requires offsets."""
    module = GroupedFeedForward(3, 8, 16, 4)

    with pytest.raises(RuntimeError, match="provide offsets"):
        module(torch.randn(10, 8))


def test_grouped_feed_forward_forward_rejects_3d_input_with_offs() -> None:
    """Test that 3D input forbids offsets."""
    module = GroupedFeedForward(3, 8, 16, 4)

    with pytest.raises(RuntimeError, match="no offset"):
        module(torch.randn(3, 4, 8), offs=torch.tensor([4, 8, 12], dtype=torch.int32))


def test_grouped_feed_forward_forward_rejects_int64_offsets() -> None:
    """Test that int64 offsets are rejected."""
    module = GroupedFeedForward(3, 8, 16, 4)

    with pytest.raises(RuntimeError, match="int32"):
        module(torch.randn(10, 8), offs=torch.tensor([2, 5, 10], dtype=torch.int64))


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_grouped_feed_forward_forward_backward_computes_gradients_2d(bias: bool) -> None:
    """Test that 2D forward supports autograd for all parameters."""
    torch.manual_seed(0)
    module = GroupedFeedForward(3, 8, 16, 4, bias=bias)
    input = torch.randn(10, 8, requires_grad=True)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    output = module(input, offs=offs)
    output.backward(torch.ones_like(output))

    assert input.grad is not None
    assert input.grad.shape == input.shape
    assert module.up_proj.weight.grad is not None
    assert module.up_proj.weight.grad.shape == module.up_proj.weight.shape
    assert module.down_proj.weight.grad is not None
    assert module.down_proj.weight.grad.shape == module.down_proj.weight.shape
    if bias:
        assert module.up_proj.bias is not None
        assert module.up_proj.bias.grad is not None
        assert module.down_proj.bias is not None
        assert module.down_proj.bias.grad is not None
    else:
        assert module.up_proj.bias is None
        assert module.down_proj.bias is None


def test_grouped_feed_forward_forward_backward_computes_gradients_3d() -> None:
    """Test that 3D forward supports autograd for all parameters."""
    torch.manual_seed(0)
    module = GroupedFeedForward(3, 8, 16, 4)
    input = torch.randn(3, 4, 8, requires_grad=True)

    output = module(input)
    output.backward(torch.ones_like(output))

    assert input.grad is not None
    assert input.grad.shape == input.shape
    assert module.up_proj.weight.grad is not None
    assert module.down_proj.weight.grad is not None
    assert module.up_proj.bias is not None
    assert module.up_proj.bias.grad is not None
    assert module.down_proj.bias is not None
    assert module.down_proj.bias.grad is not None


def test_grouped_feed_forward_forward_backward_reaches_activation_parameters() -> None:
    """Test that autograd reaches parameters of a parameterized activation."""
    torch.manual_seed(0)
    module = GroupedFeedForward(3, 8, 16, 4, activation=partial(torch.nn.PReLU, num_parameters=16))
    activation = module.activation
    assert isinstance(activation, torch.nn.PReLU)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    output = module(torch.randn(10, 8), offs=offs)
    output.backward(torch.ones_like(output))

    assert activation.weight.grad is not None


# ------------------------------------------------------------------------------
# extra_repr
# ------------------------------------------------------------------------------


def test_grouped_feed_forward_extra_repr_reports_configuration() -> None:
    """Test that extra_repr reports the module configuration."""
    module = GroupedFeedForward(3, 8, 16, 4)

    assert module.extra_repr() == "num_groups=3, in_features=8, hidden_features=16, out_features=4"


# ==============================================================================
# GatedGroupedFeedForward
# ==============================================================================


# ------------------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------------------


def test_gated_grouped_feed_forward_init_stores_dimensions_and_projections() -> None:
    """Test that __init__ stores the dimensions and creates both projections."""
    module = GatedGroupedFeedForward(3, 8, 16, 4)

    assert module.num_groups == 3
    assert module.in_features == 8
    assert module.hidden_features == 16
    assert module.out_features == 4
    assert isinstance(module.gate_up_proj, GroupedLinear)
    assert module.gate_up_proj.num_groups == 3
    assert module.gate_up_proj.in_features == 8
    assert module.gate_up_proj.out_features == 32
    assert isinstance(module.down_proj, GroupedLinear)
    assert module.down_proj.num_groups == 3
    assert module.down_proj.in_features == 16
    assert module.down_proj.out_features == 4


def test_gated_grouped_feed_forward_init_registers_default_activation() -> None:
    """Test that the default activation is a registered SiLU module."""
    module = GatedGroupedFeedForward(3, 8, 16, 4)

    assert isinstance(module.activation, torch.nn.SiLU)
    assert dict(module.named_modules())["activation"] is module.activation


def test_gated_grouped_feed_forward_init_creates_distinct_default_activations() -> None:
    """Test that each module owns its activation instance."""
    first = GatedGroupedFeedForward(3, 8, 16, 4)
    second = GatedGroupedFeedForward(3, 8, 16, 4)

    assert first.activation is not second.activation


def test_gated_grouped_feed_forward_init_builds_activation_from_factory() -> None:
    """Test that activation accepts a module type."""
    module = GatedGroupedFeedForward(3, 8, 16, 4, activation=torch.nn.GELU)

    assert isinstance(module.activation, torch.nn.GELU)


def test_gated_grouped_feed_forward_init_configures_activation_from_partial() -> None:
    """Test that a partial factory can configure the activation."""
    module = GatedGroupedFeedForward(3, 8, 16, 4, activation=partial(torch.nn.SiLU, inplace=True))

    assert isinstance(module.activation, torch.nn.SiLU)
    assert module.activation.inplace


def test_gated_grouped_feed_forward_init_default_bias_applies_to_both_projections() -> None:
    """Test that the default bias appears on both projections."""
    module = GatedGroupedFeedForward(3, 8, 16, 4)

    assert module.gate_up_proj.bias is not None
    assert module.gate_up_proj.bias.shape == (96,)
    assert module.down_proj.bias is not None
    assert module.down_proj.bias.shape == (12,)


def test_gated_grouped_feed_forward_init_no_bias_removes_both_projection_biases() -> None:
    """Test that bias=False removes the bias from both projections."""
    module = GatedGroupedFeedForward(3, 8, 16, 4, bias=False)

    assert module.gate_up_proj.bias is None
    assert module.down_proj.bias is None
    assert sorted(module.state_dict()) == ["down_proj.weight", "gate_up_proj.weight"]


@pytest.mark.parametrize(
    "bias",
    [(True, False), (False, True), (False, False)],
    ids=["gate_up_only", "down_only", "neither"],
)
def test_gated_grouped_feed_forward_init_applies_tuple_bias_per_projection(bias: tuple[bool, bool]) -> None:
    """Test that a two-element tuple sets each projection's bias separately."""
    module = GatedGroupedFeedForward(3, 8, 16, 4, bias=bias)

    assert (module.gate_up_proj.bias is not None) is bias[0]
    assert (module.down_proj.bias is not None) is bias[1]
    assert ("down_proj.bias" in module.state_dict()) is bias[1]


@pytest.mark.parametrize("bias", [(), (True,), (True, False, True)], ids=["empty", "one", "three"])
def test_gated_grouped_feed_forward_init_rejects_bias_tuple_of_wrong_length(bias: tuple[bool, ...]) -> None:
    """Test that a tuple bias must have exactly two elements."""
    with pytest.raises(ValueError, match="unpack"):
        GatedGroupedFeedForward(3, 8, 16, 4, bias=cast(tuple[bool, bool], bias))


def test_gated_grouped_feed_forward_init_respects_dtype() -> None:
    """Test that __init__ uses the requested parameter dtype."""
    module = GatedGroupedFeedForward(3, 8, 16, 4, dtype=torch.float64)

    assert module.gate_up_proj.weight.dtype is torch.float64
    assert module.down_proj.weight.dtype is torch.float64


def test_gated_grouped_feed_forward_init_uses_linear_initialization_bounds() -> None:
    """Test that each projection uses its own initialization bound."""
    module = GatedGroupedFeedForward(3, 8, 16, 4)

    assert module.gate_up_proj.weight.abs().max().item() <= 1 / math.sqrt(8)
    assert module.down_proj.weight.abs().max().item() <= 1 / math.sqrt(16)


# ------------------------------------------------------------------------------
# forward
# ------------------------------------------------------------------------------


def _reference_gated_grouped_feed_forward(
    module: GatedGroupedFeedForward,
    input: torch.Tensor,
    offs: torch.Tensor | None,
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute the expected result with independent per-group projections."""
    num_groups = module.num_groups
    hidden = module.hidden_features
    gate_up_weight = module.gate_up_proj.weight.reshape(num_groups, 2 * hidden, module.in_features)
    gate_up_bias = (
        None if module.gate_up_proj.bias is None else module.gate_up_proj.bias.reshape(num_groups, 2 * hidden)
    )
    down_weight = module.down_proj.weight.reshape(num_groups, module.out_features, hidden)
    down_bias = (
        None if module.down_proj.bias is None else module.down_proj.bias.reshape(num_groups, module.out_features)
    )

    def run(x_group: torch.Tensor, group: int) -> torch.Tensor:
        fused = F.linear(x_group, gate_up_weight[group], None if gate_up_bias is None else gate_up_bias[group])
        gate, up = fused.tensor_split(2, dim=-1)
        return F.linear(
            activation(gate) * up,
            down_weight[group],
            None if down_bias is None else down_bias[group],
        )

    if input.ndim == 3:
        return torch.stack([run(input[group], group) for group in range(num_groups)])

    assert offs is not None
    starts = [0, *offs[:-1].tolist()]
    return torch.cat(
        [run(input[start:end], group) for group, (start, end) in enumerate(zip(starts, offs.tolist(), strict=True))]
    )


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_gated_grouped_feed_forward_forward_2d_matches_reference(bias: bool) -> None:
    """Test that 2D forward matches independent per-group gated blocks."""
    torch.manual_seed(0)
    module = GatedGroupedFeedForward(3, 8, 16, 4, bias=bias)
    input = torch.randn(10, 8)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    output = module(input, offs=offs)
    expected = _reference_gated_grouped_feed_forward(module, input, offs, F.silu)

    assert output.shape == (10, 4)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_gated_grouped_feed_forward_forward_3d_matches_reference(bias: bool) -> None:
    """Test that 3D forward matches independent per-group gated blocks."""
    torch.manual_seed(0)
    module = GatedGroupedFeedForward(3, 8, 16, 4, bias=bias)
    input = torch.randn(3, 4, 8)

    output = module(input)
    expected = _reference_gated_grouped_feed_forward(module, input, None, F.silu)

    assert output.shape == (3, 4, 4)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "bias",
    [(True, False), (False, True)],
    ids=["gate_up_only", "down_only"],
)
def test_gated_grouped_feed_forward_forward_matches_reference_for_bias_configuration(
    bias: tuple[bool, bool],
) -> None:
    """Test that forward matches the reference for mixed bias configurations."""
    torch.manual_seed(0)
    module = GatedGroupedFeedForward(3, 8, 16, 4, bias=bias)
    input = torch.randn(10, 8)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    torch.testing.assert_close(
        module(input, offs=offs),
        _reference_gated_grouped_feed_forward(module, input, offs, F.silu),
    )


@pytest.mark.parametrize(
    ("factory", "activation"),
    [
        pytest.param(torch.nn.GELU, F.gelu, id="gelu"),
        pytest.param(partial(torch.nn.SiLU, inplace=True), F.silu, id="inplace_silu"),
    ],
)
def test_gated_grouped_feed_forward_forward_uses_configured_activation(
    factory: Callable[..., torch.nn.Module],
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Test that forward applies the configured activation to the gate half."""
    torch.manual_seed(0)
    module = GatedGroupedFeedForward(3, 8, 16, 4, activation=factory)
    input = torch.randn(10, 8)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    torch.testing.assert_close(
        module(input, offs=offs),
        _reference_gated_grouped_feed_forward(module, input, offs, activation),
    )


def test_gated_grouped_feed_forward_forward_activates_the_first_half() -> None:
    """Test that the activation applies to the first half of the fused output."""
    module = GatedGroupedFeedForward(1, 4, 4, 1, bias=False)
    with torch.no_grad():
        module.gate_up_proj.weight[:4].fill_(0.25)
        module.gate_up_proj.weight[4:].fill_(0.5)
        module.down_proj.weight.zero_()
        module.down_proj.weight[:, 0].fill_(1.0)

    output = module(torch.ones(1, 4), offs=torch.tensor([1], dtype=torch.int32))

    expected = F.silu(torch.ones(1, 1)) * 2.0
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "offs_values",
    [[0, 5, 10], [2, 2, 10], [2, 10, 10]],
    ids=["first_empty", "middle_empty", "last_empty"],
)
def test_gated_grouped_feed_forward_forward_supports_empty_groups(offs_values: list[int]) -> None:
    """Test that groups with no rows are handled correctly."""
    torch.manual_seed(0)
    module = GatedGroupedFeedForward(3, 8, 16, 4)
    input = torch.randn(10, 8)
    offs = torch.tensor(offs_values, dtype=torch.int32)

    torch.testing.assert_close(
        module(input, offs=offs),
        _reference_gated_grouped_feed_forward(module, input, offs, F.silu),
    )


def test_gated_grouped_feed_forward_forward_handles_zero_rows_per_group() -> None:
    """Test that 3D input with zero rows per group produces an empty output."""
    module = GatedGroupedFeedForward(3, 8, 16, 4)

    output = module(torch.empty(3, 0, 8))

    assert output.shape == (3, 0, 4)


def test_gated_grouped_feed_forward_forward_handles_empty_input() -> None:
    """Test that an empty 2D input produces an empty output."""
    module = GatedGroupedFeedForward(3, 8, 16, 4)
    offs = torch.tensor([0, 0, 0], dtype=torch.int32)

    output = module(torch.empty(0, 8), offs=offs)

    assert output.shape == (0, 4)


@pytest.mark.parametrize(
    "offs_values",
    [[2, 5, 8], [2, 5, 5]],
    ids=["normal_tail", "empty_last_group"],
)
def test_gated_grouped_feed_forward_forward_matches_reference_before_tail(offs_values: list[int]) -> None:
    """Test that rows before offs[-1] match the per-group reference."""
    torch.manual_seed(0)
    module = GatedGroupedFeedForward(3, 8, 16, 4)
    input = torch.randn(10, 8)
    offs = torch.tensor(offs_values, dtype=torch.int32)

    output = module(input, offs=offs)
    expected = _reference_gated_grouped_feed_forward(module, input, offs, F.silu)

    assert output.shape == (10, 4)
    torch.testing.assert_close(output[: offs[-1].item()], expected)


def test_gated_grouped_feed_forward_forward_handles_all_rows_as_tail() -> None:
    """Test that offsets ending at zero leave every output row unspecified."""
    module = GatedGroupedFeedForward(3, 8, 16, 4)
    input = torch.randn(10, 8)
    offs = torch.tensor([0, 0, 0], dtype=torch.int32)

    output = module(input, offs=offs)

    assert output.shape == (10, 4)


@pytest.mark.parametrize(
    ("dtype", "in_features", "hidden_features"),
    [
        pytest.param(torch.float32, 8, 16, id="fp32"),
        pytest.param(torch.bfloat16, 8, 16, id="bf16"),
    ],
)
def test_gated_grouped_feed_forward_forward_preserves_dtype(
    dtype: torch.dtype,
    in_features: int,
    hidden_features: int,
) -> None:
    """Test that forward preserves the parameter dtype."""
    module = GatedGroupedFeedForward(3, in_features, hidden_features, 4, dtype=dtype)
    input = torch.randn(10, in_features, dtype=dtype)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    assert module(input, offs=offs).dtype is dtype


@pytest.mark.parametrize(
    "out_dtype",
    [
        pytest.param(torch.bfloat16, id="scalar"),
        pytest.param((torch.bfloat16, torch.bfloat16), id="pair"),
        pytest.param((torch.bfloat16, None), id="down_none"),
        pytest.param((None, torch.bfloat16), id="gate_up_none"),
    ],
)
def test_gated_grouped_feed_forward_forward_accepts_valid_out_dtype(out_dtype: DType | tuple[DType, DType]) -> None:
    """Test that a valid out_dtype preserves the output dtype and values."""
    module = GatedGroupedFeedForward(3, 8, 16, 4, dtype=torch.bfloat16)
    input = torch.randn(10, 8, dtype=torch.bfloat16)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    expected = module(input, offs=offs)
    output = module(input, offs=offs, out_dtype=out_dtype)

    assert output.dtype is torch.bfloat16
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "out_dtype",
    [
        pytest.param((torch.float32, torch.bfloat16), id="gate_up_mismatch"),
        pytest.param((torch.bfloat16, torch.float32), id="down_mismatch"),
    ],
)
def test_gated_grouped_feed_forward_forward_rejects_mismatched_out_dtype(
    out_dtype: DType | tuple[DType, DType],
) -> None:
    """
    Test that a mismatched out_dtype is rejected by the current GEMM kernels.

    This is a known constraint of the grouped GEMM kernels. If the constraint
    is relaxed in the future, this test will fail and the parameter should be
    moved to ``test_gated_grouped_feed_forward_forward_accepts_valid_out_dtype``.
    """
    module = GatedGroupedFeedForward(3, 8, 16, 4, dtype=torch.bfloat16)
    input = torch.randn(10, 8, dtype=torch.bfloat16)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="output dtype must match"):
        module(input, offs=offs, out_dtype=out_dtype)


def test_gated_grouped_feed_forward_forward_rejects_wrong_input_width() -> None:
    """Test that the fused projection rejects an input whose last dimension is not in_features."""
    module = GatedGroupedFeedForward(3, 8, 16, 4)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="contraction dimension"):
        module(torch.randn(10, 12), offs=offs)


def test_gated_grouped_feed_forward_forward_rejects_2d_input_without_offs() -> None:
    """Test that 2D input requires offsets."""
    module = GatedGroupedFeedForward(3, 8, 16, 4)

    with pytest.raises(RuntimeError, match="provide offsets"):
        module(torch.randn(10, 8))


def test_gated_grouped_feed_forward_forward_rejects_3d_input_with_offs() -> None:
    """Test that 3D input forbids offsets."""
    module = GatedGroupedFeedForward(3, 8, 16, 4)

    with pytest.raises(RuntimeError, match="no offset"):
        module(torch.randn(3, 4, 8), offs=torch.tensor([4, 8, 12], dtype=torch.int32))


def test_gated_grouped_feed_forward_forward_rejects_int64_offsets() -> None:
    """Test that int64 offsets are rejected."""
    module = GatedGroupedFeedForward(3, 8, 16, 4)

    with pytest.raises(RuntimeError, match="int32"):
        module(torch.randn(10, 8), offs=torch.tensor([2, 5, 10], dtype=torch.int64))


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_gated_grouped_feed_forward_forward_backward_computes_gradients_2d(bias: bool) -> None:
    """Test that 2D forward supports autograd for all parameters."""
    torch.manual_seed(0)
    module = GatedGroupedFeedForward(3, 8, 16, 4, bias=bias)
    input = torch.randn(10, 8, requires_grad=True)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    output = module(input, offs=offs)
    output.backward(torch.ones_like(output))

    assert input.grad is not None
    assert input.grad.shape == input.shape
    assert module.gate_up_proj.weight.grad is not None
    assert module.gate_up_proj.weight.grad.shape == module.gate_up_proj.weight.shape
    assert module.down_proj.weight.grad is not None
    assert module.down_proj.weight.grad.shape == module.down_proj.weight.shape
    if bias:
        assert module.gate_up_proj.bias is not None
        assert module.gate_up_proj.bias.grad is not None
        assert module.down_proj.bias is not None
        assert module.down_proj.bias.grad is not None
    else:
        assert module.gate_up_proj.bias is None
        assert module.down_proj.bias is None


def test_gated_grouped_feed_forward_forward_backward_computes_gradients_3d() -> None:
    """Test that 3D forward supports autograd for all parameters."""
    torch.manual_seed(0)
    module = GatedGroupedFeedForward(3, 8, 16, 4)
    input = torch.randn(3, 4, 8, requires_grad=True)

    output = module(input)
    output.backward(torch.ones_like(output))

    assert input.grad is not None
    assert input.grad.shape == input.shape
    assert module.gate_up_proj.weight.grad is not None
    assert module.down_proj.weight.grad is not None
    assert module.gate_up_proj.bias is not None
    assert module.gate_up_proj.bias.grad is not None
    assert module.down_proj.bias is not None
    assert module.down_proj.bias.grad is not None


def test_gated_grouped_feed_forward_forward_backward_reaches_activation_parameters() -> None:
    """Test that autograd reaches parameters of a parameterized activation."""
    torch.manual_seed(0)
    module = GatedGroupedFeedForward(3, 8, 16, 4, activation=partial(torch.nn.PReLU, num_parameters=16))
    activation = module.activation
    assert isinstance(activation, torch.nn.PReLU)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    output = module(torch.randn(10, 8), offs=offs)
    output.backward(torch.ones_like(output))

    assert activation.weight.grad is not None


# ------------------------------------------------------------------------------
# extra_repr
# ------------------------------------------------------------------------------


def test_gated_grouped_feed_forward_extra_repr_reports_configuration() -> None:
    """Test that extra_repr reports the module configuration."""
    module = GatedGroupedFeedForward(3, 8, 16, 4)

    assert module.extra_repr() == "num_groups=3, in_features=8, hidden_features=16, out_features=4"
