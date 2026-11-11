# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.components.linear`."""

import math

import pytest
import torch
import torch.nn.functional as F

from mokit.components.linear import BatchedLinear

# ==============================================================================
# BatchedLinear
# ==============================================================================


# ------------------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------------------


def test_init_stores_flattened_parameters() -> None:
    """Test that __init__ stores weight and bias in flattened shapes."""
    module = BatchedLinear(2, 3, 4)

    assert module.batches == 2
    assert module.in_features == 3
    assert module.out_features == 4
    assert module.weight.shape == (8, 3)
    assert isinstance(module.weight, torch.nn.Parameter)
    assert module.bias is not None
    assert module.bias.shape == (8,)
    assert isinstance(module.bias, torch.nn.Parameter)


def test_init_without_bias_registers_no_bias_parameter() -> None:
    """Test that bias=False leaves the bias unset."""
    module = BatchedLinear(2, 3, 4, bias=False)

    assert module.bias is None
    assert list(dict(module.named_parameters())) == ["weight"]


def test_init_respects_dtype() -> None:
    """Test that __init__ uses the requested parameter dtype."""
    module = BatchedLinear(2, 3, 4, dtype=torch.float64)

    assert module.weight.dtype is torch.float64
    assert module.bias is not None
    assert module.bias.dtype is torch.float64


# ------------------------------------------------------------------------------
# reset_parameters
# ------------------------------------------------------------------------------


def test_reset_parameters_initializes_within_linear_bounds() -> None:
    """Test that reset_parameters initializes parameters within nn.Linear bounds."""
    module = BatchedLinear(2, 3, 4)
    bound = 1 / math.sqrt(module.in_features)

    assert module.weight.abs().max().item() <= bound
    assert module.bias is not None
    assert module.bias.abs().max().item() <= bound


def test_reset_parameters_rejects_zero_in_features() -> None:
    """Test that reset_parameters rejects an invalid in_features value."""
    with pytest.raises(ValueError, match="in_features must be > 0"):
        BatchedLinear(2, 0, 4)


# ------------------------------------------------------------------------------
# forward
# ------------------------------------------------------------------------------


def _reference_forward(module: BatchedLinear, input: torch.Tensor) -> torch.Tensor:
    """Compute the expected result with independent F.linear calls per batch."""
    weight = module.weight.reshape(module.batches, module.out_features, module.in_features)
    bias = None if module.bias is None else module.bias.reshape(module.batches, module.out_features)
    outputs: list[torch.Tensor] = []
    for batch in range(module.batches):
        batch_bias = None if bias is None else bias[batch]
        outputs.append(F.linear(input[batch], weight[batch], batch_bias))
    return torch.stack(outputs)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize(
    "input_shape",
    [(2, 3), (2, 5, 3), (2, 6, 7, 3)],
    ids=["2d", "3d", "4d"],
)
def test_forward_matches_per_batch_linear(
    input_shape: tuple[int, ...],
    bias: bool,
) -> None:
    """Test that forward matches independent linear layers per batch."""
    torch.manual_seed(0)
    module = BatchedLinear(2, 3, 4, bias=bias)
    input = torch.randn(input_shape)

    output = module(input)
    expected = _reference_forward(module, input)

    assert output.shape == expected.shape
    torch.testing.assert_close(output, expected)


def test_forward_supports_single_batch() -> None:
    """Test that a module with one batch matches a single linear layer."""
    torch.manual_seed(0)
    module = BatchedLinear(1, 3, 4)
    input = torch.randn(1, 5, 3)

    torch.testing.assert_close(module(input), _reference_forward(module, input))


def test_forward_uses_each_batch_weight() -> None:
    """Test that each batch uses its own weight rows."""
    module = BatchedLinear(2, 3, 2, bias=False)
    with torch.no_grad():
        module.weight.zero_()
        module.weight[0].fill_(1.0)
    input = torch.ones(2, 1, 3)

    output = module(input)

    expected = torch.zeros(2, 1, 2)
    expected[0, 0, 0] = 3.0
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("input_shape", [(), (3,)], ids=["scalar", "1d"])
def test_forward_rejects_fewer_than_two_dimensions(
    input_shape: tuple[int, ...],
) -> None:
    """Test that forward rejects inputs with fewer than two dimensions."""
    module = BatchedLinear(1, 3, 4)

    with pytest.raises(ValueError, match="expects input of shape"):
        module(torch.ones(input_shape))


def test_forward_rejects_wrong_batch_size() -> None:
    """Test that forward rejects an input with the wrong batch dimension."""
    module = BatchedLinear(2, 3, 4)

    with pytest.raises(ValueError, match="batches=2"):
        module(torch.randn(3, 5, 3))


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("last_dim", [4, 7], ids=["divisible", "not_divisible"])
def test_forward_delegates_last_dimension_check_to_bmm(
    last_dim: int,
    bias: bool,
) -> None:
    """Test that bmm rejects a last dimension that is not in_features."""
    module = BatchedLinear(2, 3, 4, bias=bias)

    with pytest.raises(RuntimeError, match="Expected size for first two dimensions"):
        module(torch.randn(2, 5, last_dim))


def test_forward_preserves_dtype() -> None:
    """Test that forward preserves the parameter dtype."""
    module = BatchedLinear(2, 3, 4, dtype=torch.float64)
    input = torch.randn(2, 5, 3, dtype=torch.float64)

    assert module(input).dtype is torch.float64


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_forward_backward_computes_gradients(bias: bool) -> None:
    """Test that forward supports autograd for input, weight, and bias."""
    torch.manual_seed(0)
    module = BatchedLinear(2, 3, 4, bias=bias)
    input = torch.randn(2, 5, 3, requires_grad=True)

    module(input).sum().backward()

    assert input.grad is not None
    assert input.grad.shape == input.shape
    assert module.weight.grad is not None
    assert module.weight.grad.shape == module.weight.shape
    if bias:
        assert module.bias is not None
        assert module.bias.grad is not None
        assert module.bias.grad.shape == module.bias.shape
    else:
        assert module.bias is None


# ------------------------------------------------------------------------------
# extra_repr
# ------------------------------------------------------------------------------


def test_extra_repr_reports_configuration() -> None:
    """Test that extra_repr reports the module configuration."""
    module = BatchedLinear(2, 3, 4)

    assert module.extra_repr() == "batches=2, in_features=3, out_features=4, bias=True"


def test_extra_repr_reports_missing_bias() -> None:
    """Test that extra_repr reports bias=False."""
    module = BatchedLinear(2, 3, 4, bias=False)

    assert module.extra_repr() == "batches=2, in_features=3, out_features=4, bias=False"
