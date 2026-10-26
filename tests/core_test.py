# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.core`."""

import dataclasses
from collections.abc import Sequence
from typing import Any

import pytest
import torch

from mokit.core import MokitConfigBase, MokitWrapperTensorBase


def test_is_dataclass() -> None:
    """Test that MokitConfigBase is a dataclass."""
    assert dataclasses.is_dataclass(MokitConfigBase)


def test_is_frozen() -> None:
    """Test that instances of MokitConfigBase are immutable."""
    config = MokitConfigBase()
    attribute = "value"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, attribute, 1)


def test_is_slotted() -> None:
    """Test that instances of MokitConfigBase have no instance dictionary."""
    assert not hasattr(MokitConfigBase(), "__dict__")


def test_equality() -> None:
    """Test that instances of MokitConfigBase compare equal."""
    assert MokitConfigBase() == MokitConfigBase()


def test_is_hashable() -> None:
    """Test that instances of MokitConfigBase are hashable."""
    assert hash(MokitConfigBase()) == hash(MokitConfigBase())


class _Wrapper(MokitWrapperTensorBase):
    """A wrapper whose four flatten attributes cover every field kind, used for repr tests."""

    inner_tensor_names = ("payload",)
    metadata_names = ("tag",)
    keyword_inner_tensor_names = ("bias",)
    keyword_metadata_names = ("note",)

    def __new__(
        cls,
        inner: torch.Tensor,
        tag: object = "hi",
        outer_size: Sequence[int | torch.SymInt] | None = None,
        outer_stride: Sequence[int | torch.SymInt] | None = None,
        *,
        bias: torch.Tensor | None = None,
        note: object = "note",
        requires_grad: bool = False,
        layout: torch.layout = torch.strided,
    ) -> torch.Tensor:
        """Create the wrapper tensor."""
        if outer_size is None:
            outer_size = inner.shape
        if outer_stride is None:
            outer_stride = inner.stride()
        return torch.Tensor._make_wrapper_subclass(
            cls,
            outer_size,
            outer_stride,
            dtype=inner.dtype,
            device=inner.device,
            layout=layout,
            requires_grad=requires_grad,
        )

    def __init__(
        self,
        inner: torch.Tensor,
        tag: object = "hi",
        outer_size: Sequence[int | torch.SymInt] | None = None,
        outer_stride: Sequence[int | torch.SymInt] | None = None,
        *,
        bias: torch.Tensor | None = None,
        note: object = "note",
        requires_grad: bool = False,
        layout: torch.layout = torch.strided,
    ) -> None:
        """Store the wrapper fields."""
        self.payload = inner
        self.tag = tag
        self.bias = torch.zeros_like(inner) if bias is None else bias
        self.note = note


class _EmptyWrapper(MokitWrapperTensorBase):
    """A wrapper with no flattened fields, used for repr tests."""

    def __new__(
        cls,
        outer_size: Sequence[int | torch.SymInt] | None = None,
        outer_stride: Sequence[int | torch.SymInt] | None = None,
    ) -> torch.Tensor:
        """Create the wrapper tensor."""
        if outer_size is None:
            outer_size = (0,)
        return torch.Tensor._make_wrapper_subclass(cls, outer_size, outer_stride, dtype=torch.float32)

    def __init__(
        self,
        outer_size: Sequence[int | torch.SymInt] | None = None,
        outer_stride: Sequence[int | torch.SymInt] | None = None,
    ) -> None:
        """Accept the geometry arguments."""


class _RequiresGradFieldWrapper(_Wrapper):
    """A wrapper whose declared requires_grad field collides with the tensor attribute."""

    metadata_names = ("requires_grad",)
    keyword_inner_tensor_names = ()
    keyword_metadata_names = ()


class _GradFnFunction(torch.autograd.Function):
    """A custom autograd function returning a wrapper, used for repr tests."""

    @staticmethod
    def forward(ctx: Any, inner: torch.Tensor) -> torch.Tensor:
        """Return a wrapper around the input."""
        return _Wrapper(inner)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """Return the incoming gradient."""
        return grad_outputs[0]


def test_repr_defaults() -> None:
    """Test default repr output and the __str__ fallback."""
    wrapper = _Wrapper(torch.tensor([1.0, 2.0]))
    expected = "_Wrapper(payload=tensor([1., 2.]), tag='hi', bias=tensor([0., 0.]), note='note')"
    assert repr(wrapper) == expected
    assert str(wrapper) == expected


@pytest.mark.parametrize(
    ("tag", "rendered"),
    [
        ("a, b=c", "tag='a, b=c'"),
        ("", "tag=''"),
        (None, "tag=None"),
        ("line1\nline2", r"tag='line1\nline2'"),
    ],
)
def test_repr_formats_field_values_with_repr(tag: object, rendered: str) -> None:
    """Test that field values are formatted with repr."""
    assert rendered in repr(_Wrapper(torch.tensor([1.0, 2.0]), tag))


def test_repr_tensor_contents_first() -> None:
    """Test that tensor_contents is placed before the declared fields."""
    wrapper = _Wrapper(torch.tensor([1.0, 2.0]))
    assert wrapper.__repr__(tensor_contents="custom") == (
        "_Wrapper(custom, payload=tensor([1., 2.]), tag='hi', bias=tensor([0., 0.]), note='note')"
    )
    assert wrapper.__repr__(tensor_contents=None) == repr(wrapper)


def test_repr_empty_tensor_contents() -> None:
    """Test that an empty tensor_contents is honored like Tensor."""
    wrapper = _Wrapper(torch.tensor([1.0, 2.0]))
    assert wrapper.__repr__(tensor_contents="") == (
        "_Wrapper(, payload=tensor([1., 2.]), tag='hi', bias=tensor([0., 0.]), note='note')"
    )
    wrapper64 = _Wrapper(torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert wrapper64.__repr__(tensor_contents="").startswith("_Wrapper(, dtype=torch.float64, payload=")
    assert _EmptyWrapper().__repr__(tensor_contents="") == "_EmptyWrapper()"
    assert _EmptyWrapper().__repr__(tensor_contents="X") == "_EmptyWrapper(X)"


def test_repr_tensor_contents_before_attributes() -> None:
    """Test that tensor_contents precedes tensor attributes and declared fields."""
    wrapper = _Wrapper(torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert wrapper.__repr__(tensor_contents="custom").startswith("_Wrapper(custom, dtype=torch.float64, payload=")


def test_repr_dtype() -> None:
    """Test that a non-default dtype is printed before the declared fields."""
    wrapper = _Wrapper(torch.tensor([1.0, 2.0], dtype=torch.float64))
    expected = (
        "_Wrapper(dtype=torch.float64, "
        "payload=tensor([1., 2.], dtype=torch.float64), tag='hi', "
        "bias=tensor([0., 0.], dtype=torch.float64), note='note')"
    )
    assert repr(wrapper) == expected


def test_repr_device() -> None:
    """Test that a non-default device is printed before the declared fields."""
    wrapper = _Wrapper(torch.tensor([1.0, 2.0], device="meta"))
    expected = (
        "_Wrapper(device='meta', "
        "payload=tensor(..., device='meta', size=(2,)), tag='hi', "
        "bias=tensor(..., device='meta', size=(2,)), note='note')"
    )
    assert repr(wrapper) == expected


def test_repr_requires_grad() -> None:
    """Test that requires_grad is printed before the declared fields."""
    wrapper = _Wrapper(torch.tensor([1.0, 2.0]), requires_grad=True)
    expected = "_Wrapper(requires_grad=True, payload=tensor([1., 2.]), tag='hi', bias=tensor([0., 0.]), note='note')"
    assert repr(wrapper) == expected


def test_repr_grad_fn() -> None:
    """Test that grad_fn is printed before the declared fields."""
    wrapper = _GradFnFunction.apply(torch.tensor([1.0, 2.0], requires_grad=True))
    expected = (
        "_Wrapper(grad_fn=<_GradFnFunctionBackward>, "
        "payload=tensor([1., 2.], requires_grad=True), tag='hi', "
        "bias=tensor([0., 0.]), note='note')"
    )
    assert repr(wrapper) == expected


def test_repr_layout() -> None:
    """Test that a non-strided layout is printed."""
    wrapper = _Wrapper(torch.tensor([1.0, 2.0]), layout=torch.sparse_coo)
    assert wrapper.layout == torch.sparse_coo
    assert repr(wrapper) == (
        "_Wrapper(layout=torch.sparse_coo, payload=tensor([1., 2.]), tag='hi', bias=tensor([0., 0.]), note='note')"
    )


def test_repr_filters_previously_printed_field_names() -> None:
    """Test that declared fields skip names already printed as tensor attributes."""
    rendered = repr(_RequiresGradFieldWrapper(torch.tensor([1.0, 2.0]), requires_grad=True))
    assert rendered == ("_RequiresGradFieldWrapper(requires_grad=True, payload=tensor([1., 2.]))")
    assert rendered.count("requires_grad=") == 1


def test_repr_empty_wrapper() -> None:
    """Test that a wrapper with no fields prints empty parentheses."""
    assert repr(_EmptyWrapper()) == "_EmptyWrapper()"
