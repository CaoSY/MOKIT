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

from mokit.components.feed_forward import FeedForward

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
