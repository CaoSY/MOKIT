# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.components.linear`."""

import math

import pytest
import torch
import torch.nn.functional as F

from mokit.components.linear import BatchedLinear, GroupedLinear

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


# ==============================================================================
# GroupedLinear
# ==============================================================================


# ------------------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------------------


def test_grouped_init_stores_flattened_parameters() -> None:
    """Test that GroupedLinear.__init__ stores flattened parameters."""
    module = GroupedLinear(2, 3, 4)

    assert module.num_groups == 2
    assert module.in_features == 3
    assert module.out_features == 4
    assert module.weight.shape == (8, 3)
    assert isinstance(module.weight, torch.nn.Parameter)
    assert module.bias is not None
    assert module.bias.shape == (8,)
    assert isinstance(module.bias, torch.nn.Parameter)


def test_grouped_init_without_bias_registers_no_bias_parameter() -> None:
    """Test that GroupedLinear(bias=False) leaves the bias unset."""
    module = GroupedLinear(2, 3, 4, bias=False)

    assert module.bias is None
    assert list(dict(module.named_parameters())) == ["weight"]


def test_grouped_init_respects_dtype() -> None:
    """Test that GroupedLinear.__init__ uses the requested parameter dtype."""
    module = GroupedLinear(2, 3, 4, dtype=torch.float64)

    assert module.weight.dtype is torch.float64
    assert module.bias is not None
    assert module.bias.dtype is torch.float64


def test_grouped_init_allows_zero_in_features() -> None:
    """Test that zero in_features produces empty weights and zero bias."""
    module = GroupedLinear(2, 0, 4)

    assert module.weight.shape == (8, 0)
    assert module.weight.numel() == 0
    assert module.bias is not None
    assert module.bias.shape == (8,)
    assert torch.count_nonzero(module.bias) == 0


def test_grouped_init_rejects_negative_in_features() -> None:
    """Test that negative dimensions are rejected by torch.empty."""
    with pytest.raises(RuntimeError, match="negative dimension"):
        GroupedLinear(2, -1, 4)


# ------------------------------------------------------------------------------
# reset_parameters
# ------------------------------------------------------------------------------


def test_grouped_reset_parameters_matches_uniform_initialization() -> None:
    """Test that parameters match independent uniform initialization."""
    torch.manual_seed(0)
    module = GroupedLinear(2, 8, 4)
    bound = 1 / math.sqrt(module.in_features)

    torch.manual_seed(0)
    expected_weight = torch.empty(8, 8).uniform_(-bound, bound)
    expected_bias = torch.empty(8).uniform_(-bound, bound)

    torch.testing.assert_close(module.weight, expected_weight)
    assert module.bias is not None
    torch.testing.assert_close(module.bias, expected_bias)


def test_grouped_reset_parameters_uses_zero_bound_for_zero_in_features() -> None:
    """Test that zero in_features uses a zero initialization bound."""
    module = GroupedLinear(2, 0, 4)

    assert module.bias is not None
    assert torch.count_nonzero(module.bias) == 0

    module.reset_parameters()
    assert torch.count_nonzero(module.bias) == 0


# ------------------------------------------------------------------------------
# forward
# ------------------------------------------------------------------------------


def _reference_grouped_forward(
    module: GroupedLinear,
    input: torch.Tensor,
    offs: torch.Tensor | None,
) -> torch.Tensor:
    """Compute the expected result with one F.linear call per group."""
    weight = module.weight.reshape(module.num_groups, module.out_features, module.in_features)
    bias = None if module.bias is None else module.bias.reshape(module.num_groups, module.out_features)

    if input.ndim == 3:
        return torch.stack(
            [
                F.linear(input[group], weight[group], None if bias is None else bias[group])
                for group in range(module.num_groups)
            ]
        )

    assert offs is not None
    starts = [0, *offs[:-1].tolist()]
    return torch.cat(
        [
            F.linear(input[start:end], weight[group], None if bias is None else bias[group])
            for group, (start, end) in enumerate(zip(starts, offs.tolist(), strict=True))
        ]
    )


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        pytest.param(torch.float32, 1e-5, 1e-6, id="fp32"),
        pytest.param(torch.bfloat16, 2e-2, 2e-3, id="bf16"),
    ],
)
def test_grouped_forward_2d_matches_per_group_linear(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
    bias: bool,
) -> None:
    """Test that 2D forward matches independent linear layers per group."""
    torch.manual_seed(0)
    module = GroupedLinear(3, 16, 8, bias=bias, dtype=dtype)
    input = torch.randn(10, 16, dtype=dtype)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    output = module(input, offs=offs)
    expected = _reference_grouped_forward(module, input, offs)

    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        pytest.param(torch.float32, 1e-5, 1e-6, id="fp32"),
        pytest.param(torch.bfloat16, 2e-2, 2e-3, id="bf16"),
    ],
)
def test_grouped_forward_3d_matches_stacked_linear(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
    bias: bool,
) -> None:
    """Test that 3D forward matches independent linear layers per group."""
    torch.manual_seed(0)
    module = GroupedLinear(3, 16, 8, bias=bias, dtype=dtype)
    input = torch.randn(3, 4, 16, dtype=dtype)

    output = module(input)
    expected = _reference_grouped_forward(module, input, None)

    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


def test_grouped_forward_handles_zero_rows_per_group() -> None:
    """Test that 3D input with zero rows per group produces an empty output."""
    module = GroupedLinear(3, 16, 8)

    output = module(torch.empty(3, 0, 16))

    assert output.shape == (3, 0, 8)


def test_grouped_forward_uses_each_group_weight() -> None:
    """Test that each group uses its own weight rows."""
    module = GroupedLinear(3, 8, 2, bias=False)
    with torch.no_grad():
        weight = module.weight.reshape(3, 2, 8)
        for group in range(3):
            weight[group].fill_(group + 1)

    input = torch.ones(9, 8)
    offs = torch.tensor([2, 5, 9], dtype=torch.int32)
    output = module(input, offs=offs)

    expected = torch.tensor([[8.0, 8.0]] * 2 + [[16.0, 16.0]] * 3 + [[24.0, 24.0]] * 4)
    torch.testing.assert_close(output, expected)


def test_grouped_forward_uses_each_group_bias() -> None:
    """Test that each group uses its own bias rows."""
    module = GroupedLinear(3, 8, 2, bias=True)
    bias = module.bias
    assert bias is not None
    with torch.no_grad():
        module.weight.zero_()
        bias.copy_(torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))

    input = torch.zeros(9, 8)
    offs = torch.tensor([2, 5, 9], dtype=torch.int32)
    output = module(input, offs=offs)

    expected = torch.tensor([[1.0, 1.0]] * 2 + [[2.0, 2.0]] * 3 + [[3.0, 3.0]] * 4)
    torch.testing.assert_close(output, expected)


def test_grouped_forward_supports_single_group() -> None:
    """Test that a single group matches one linear layer."""
    torch.manual_seed(0)
    module = GroupedLinear(1, 16, 8)
    input = torch.randn(10, 16)
    offs = torch.tensor([10], dtype=torch.int32)

    output = module(input, offs=offs)

    torch.testing.assert_close(output, F.linear(input, module.weight, module.bias))


def test_grouped_forward_supports_empty_groups() -> None:
    """Test that groups with no rows are skipped correctly."""
    torch.manual_seed(0)
    module = GroupedLinear(3, 16, 8)
    input = torch.randn(10, 16)
    offs = torch.tensor([2, 2, 10], dtype=torch.int32)

    output = module(input, offs=offs)

    torch.testing.assert_close(output, _reference_grouped_forward(module, input, offs))


@pytest.mark.parametrize(
    "offs_values",
    [[0, 5, 10], [2, 10, 10]],
    ids=["first_empty", "last_empty"],
)
def test_grouped_forward_supports_empty_boundary_groups(offs_values: list[int]) -> None:
    """Test that an empty first or last group is handled correctly."""
    torch.manual_seed(0)
    module = GroupedLinear(3, 16, 8)
    input = torch.randn(10, 16)
    offs = torch.tensor(offs_values, dtype=torch.int32)

    output = module(input, offs=offs)

    torch.testing.assert_close(output, _reference_grouped_forward(module, input, offs))


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_grouped_forward_matches_reference_before_tail(bias: bool) -> None:
    """Test that rows before offs[-1] match the per-group reference."""
    torch.manual_seed(0)
    module = GroupedLinear(3, 16, 8, bias=bias)
    input = torch.randn(10, 16)
    offs = torch.tensor([2, 5, 8], dtype=torch.int32)

    output = module(input, offs=offs)
    expected = _reference_grouped_forward(module, input, offs)

    assert output.shape == (10, 8)
    torch.testing.assert_close(output[: offs[-1].item()], expected)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_grouped_forward_matches_reference_before_tail_with_empty_last_group(bias: bool) -> None:
    """Test tail rows after an empty last group leave valid rows unchanged."""
    torch.manual_seed(0)
    module = GroupedLinear(3, 16, 8, bias=bias)
    input = torch.randn(10, 16)
    offs = torch.tensor([2, 5, 5], dtype=torch.int32)

    output = module(input, offs=offs)
    expected = _reference_grouped_forward(module, input, offs)

    assert output.shape == (10, 8)
    torch.testing.assert_close(output[: offs[-1].item()], expected)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_grouped_forward_handles_all_rows_as_tail(bias: bool) -> None:
    """Test that offsets ending at zero leave every output row unspecified."""
    module = GroupedLinear(3, 16, 8, bias=bias)
    input = torch.randn(10, 16)
    offs = torch.tensor([0, 0, 0], dtype=torch.int32)

    output = module(input, offs=offs)

    assert output.shape == (10, 8)


def test_grouped_forward_matches_reference_before_tail_with_single_group() -> None:
    """Test tail rows with a single group."""
    torch.manual_seed(0)
    module = GroupedLinear(1, 16, 8)
    input = torch.randn(10, 16)
    offs = torch.tensor([5], dtype=torch.int32)

    output = module(input, offs=offs)

    assert output.shape == (10, 8)
    torch.testing.assert_close(output[:5], F.linear(input[:5], module.weight, module.bias))


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_grouped_forward_handles_empty_input(bias: bool) -> None:
    """Test that empty inputs produce empty outputs."""
    module = GroupedLinear(3, 16, 8, bias=bias)
    offs = torch.tensor([0, 0, 0], dtype=torch.int32)

    output = module(torch.empty(0, 16), offs=offs)

    assert output.shape == (0, 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_grouped_forward_preserves_dtype(dtype: torch.dtype) -> None:
    """Test that forward preserves the input dtype."""
    module = GroupedLinear(3, 16, 8, dtype=dtype)
    input = torch.randn(10, 16, dtype=dtype)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    assert module(input, offs=offs).dtype is dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_grouped_forward_accepts_matching_out_dtype(dtype: torch.dtype) -> None:
    """Test that out_dtype matching the input dtype is accepted."""
    module = GroupedLinear(3, 16, 8, dtype=dtype)
    input = torch.randn(10, 16, dtype=dtype)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    assert module(input, offs=offs, out_dtype=dtype).dtype is dtype


def test_grouped_forward_rejects_mismatched_out_dtype() -> None:
    """Test that a mismatched out_dtype is rejected."""
    module = GroupedLinear(3, 16, 8, dtype=torch.float32)
    input = torch.randn(10, 16)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="output dtype must match"):
        module(input, offs=offs, out_dtype=torch.bfloat16)


def test_grouped_forward_rejects_int64_offsets() -> None:
    """Test that int64 offsets are rejected."""
    module = GroupedLinear(3, 16, 8)

    with pytest.raises(RuntimeError, match="int32"):
        module(torch.randn(10, 16), offs=torch.tensor([2, 5, 10], dtype=torch.int64))


def test_grouped_forward_rejects_wrong_offs_length() -> None:
    """Test that offsets must have one entry per group."""
    module = GroupedLinear(3, 16, 8)

    with pytest.raises(RuntimeError, match="batch sizes"):
        module(torch.randn(10, 16), offs=torch.tensor([5, 10], dtype=torch.int32))


def test_grouped_forward_rejects_2d_input_without_offs() -> None:
    """Test that 2D input requires offsets."""
    module = GroupedLinear(3, 16, 8)

    with pytest.raises(RuntimeError, match="provide offsets"):
        module(torch.randn(10, 16))


def test_grouped_forward_rejects_3d_input_with_offs() -> None:
    """Test that 3D input forbids offsets."""
    module = GroupedLinear(3, 16, 8)

    with pytest.raises(RuntimeError, match="no offset"):
        module(torch.randn(3, 4, 16), offs=torch.tensor([4, 8, 12], dtype=torch.int32))


def test_grouped_forward_rejects_wrong_input_width() -> None:
    """Test that the input's last dimension must equal in_features."""
    module = GroupedLinear(3, 16, 8)

    with pytest.raises(RuntimeError, match="contraction dimension"):
        module(torch.randn(10, 12), offs=torch.tensor([2, 5, 10], dtype=torch.int32))


@pytest.mark.parametrize("shape", [(16,), (1, 1, 10, 16)], ids=["1d", "4d"])
def test_grouped_forward_rejects_invalid_rank(shape: tuple[int, ...]) -> None:
    """Test that input must be 2D or 3D."""
    module = GroupedLinear(3, 16, 8)

    with pytest.raises(RuntimeError, match="2 or 3d"):
        module(torch.randn(shape), offs=torch.tensor([2, 5, 10], dtype=torch.int32))


def test_grouped_forward_backward_computes_gradients_2d() -> None:
    """Test that 2D forward supports autograd for all parameters."""
    module = GroupedLinear(3, 16, 8)
    input = torch.randn(10, 16, requires_grad=True)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    output = module(input, offs=offs)
    output.backward(torch.ones_like(output))

    assert input.grad is not None
    assert input.grad.shape == input.shape
    assert module.weight.grad is not None
    assert module.weight.grad.shape == module.weight.shape
    assert module.bias is not None
    assert module.bias.grad is not None
    assert module.bias.grad.shape == module.bias.shape


def test_grouped_forward_backward_computes_gradients_3d() -> None:
    """Test that 3D forward supports autograd for all parameters."""
    module = GroupedLinear(3, 16, 8)
    input = torch.randn(3, 4, 16, requires_grad=True)

    output = module(input)
    output.backward(torch.ones_like(output))

    assert input.grad is not None
    assert input.grad.shape == input.shape
    assert module.weight.grad is not None
    assert module.weight.grad.shape == module.weight.shape
    assert module.bias is not None
    assert module.bias.grad is not None
    assert module.bias.grad.shape == module.bias.shape


def test_grouped_forward_backward_without_bias() -> None:
    """Test that autograd works when bias is disabled."""
    module = GroupedLinear(3, 16, 8, bias=False)
    input = torch.randn(10, 16, requires_grad=True)
    offs = torch.tensor([2, 5, 10], dtype=torch.int32)

    output = module(input, offs=offs)
    output.backward(torch.ones_like(output))

    assert module.bias is None
    assert input.grad is not None
    assert module.weight.grad is not None


# ------------------------------------------------------------------------------
# extra_repr
# ------------------------------------------------------------------------------


def test_grouped_extra_repr_reports_configuration() -> None:
    """Test that extra_repr reports the module configuration."""
    module = GroupedLinear(2, 3, 4)

    assert module.extra_repr() == "num_groups=2, in_features=3, out_features=4, bias=True"


def test_grouped_extra_repr_reports_missing_bias() -> None:
    """Test that extra_repr reports bias=False."""
    module = GroupedLinear(2, 3, 4, bias=False)

    assert module.extra_repr() == "num_groups=2, in_features=3, out_features=4, bias=False"
