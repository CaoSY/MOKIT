# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.core.wrapper_tensor_base`."""

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Self, cast

import pytest
import torch
from torch._ops import OpOverload

from mokit.core import MokitWrapperTensorBase
from mokit.core.wrapper_tensor_base import unpack_args

# ==============================================================================
# unpack_args
# ==============================================================================


def test_unpack_args_forwards_metadata_and_positional_arguments() -> None:
    """Test that unpack_args forwards metadata and unpacks positional arguments."""
    calls: list[tuple[Any, ...]] = []

    def handler(
        cls: type[Any],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        first: int,
        second: int,
    ) -> int:
        """Record the call and return the sum of its arguments."""
        calls.append((cls, func, types, first, second))
        return first + second

    packed = unpack_args(handler)
    func = torch.ops.aten.add.default
    result = packed(int, func, (int,), (2, 3), {})

    assert result == 5
    assert calls == [(int, func, (int,), 2, 3)]


def test_unpack_args_expands_keyword_arguments() -> None:
    """Test that unpack_args expands keyword arguments for the handler."""
    calls: list[tuple[int, int]] = []

    def handler(
        cls: type[Any],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        first: int,
        *,
        second: int = 0,
    ) -> int:
        """Record the integer arguments and return their sum."""
        calls.append((first, second))
        return first + second

    packed = unpack_args(handler)
    result = packed(type(handler), str, (), (2,), {"second": 3})

    assert result == 5
    assert calls == [(2, 3)]


def test_unpack_args_treats_none_kwargs_as_empty() -> None:
    """Test that unpack_args treats None kwargs as no keyword arguments."""

    def handler(
        cls: type[Any],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        value: int,
        *,
        scale: int = 1,
    ) -> int:
        """Return the scaled value."""
        return value * scale

    packed = unpack_args(handler)

    assert packed(int, str, (), (4,), None) == 4


def test_unpack_args_preserves_handler_metadata() -> None:
    """Test that unpack_args preserves the wrapped handler's metadata."""

    def handler(
        cls: type[Any],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
    ) -> None:
        """Accept dispatch metadata."""

    packed = cast(Any, unpack_args(handler))

    assert packed is not handler
    assert packed.__name__ == handler.__name__
    assert packed.__doc__ == handler.__doc__
    assert packed.__module__ == handler.__module__
    assert packed.__wrapped__ is handler


def test_unpack_args_propagates_handler_exception() -> None:
    """Test that unpack_args propagates exceptions raised by the handler."""

    class HandlerError(Exception):
        """An exception raised by the test handler."""

    def handler(
        cls: type[Any],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
    ) -> None:
        """Raise the test exception."""
        raise HandlerError

    packed = unpack_args(handler)

    with pytest.raises(HandlerError):
        packed(int, str, (), (), None)


# ==============================================================================
# MokitWrapperTensorBase
# ==============================================================================

# ------------------------------------------------------------------------------
# __init_subclass__
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("attr_name", MokitWrapperTensorBase._FLATTEN_ATTR_NAMES)
def test_init_subclass_rejects_non_tuple_attribute(attr_name: str) -> None:
    """Test that a non-tuple flatten attribute is rejected."""
    with pytest.raises(TypeError, match=f"The class attribute '{attr_name}' must be a tuple."):
        type("Bad", (MokitWrapperTensorBase,), {attr_name: ["payload"]})


@pytest.mark.parametrize("attr_name", MokitWrapperTensorBase._FLATTEN_ATTR_NAMES)
def test_init_subclass_rejects_non_string_field(attr_name: str) -> None:
    """Test that a non-string field name is rejected."""
    with pytest.raises(TypeError, match=f"The class attribute '{attr_name}' must be a tuple of strings."):
        type("Bad", (MokitWrapperTensorBase,), {attr_name: (1,)})


@pytest.mark.parametrize(
    "attributes",
    [
        {"positional_inner_tensor_names": ("payload", "payload")},
        {"positional_inner_tensor_names": ("payload",), "positional_metadata_names": ("payload",)},
        {"keyword_inner_tensor_names": ("bias",), "keyword_metadata_names": ("bias",)},
    ],
)
def test_init_subclass_rejects_duplicate_field(attributes: dict[str, tuple[str, ...]]) -> None:
    """Test that a field name declared more than once is rejected."""
    with pytest.raises(ValueError, match="declared more than once"):
        type("Bad", (MokitWrapperTensorBase,), attributes)


@pytest.mark.parametrize("field", ["outer_size", "outer_stride"])
def test_init_subclass_rejects_reserved_field(field: str) -> None:
    """Test that a reserved field name is rejected."""
    with pytest.raises(ValueError, match=f"Reserved field name.*{field}"):
        type("Bad", (MokitWrapperTensorBase,), {"positional_inner_tensor_names": (field,)})


# ------------------------------------------------------------------------------
# __repr__
# ------------------------------------------------------------------------------


class _Wrapper(MokitWrapperTensorBase):
    """A wrapper whose four flatten attributes cover every field kind."""

    positional_inner_tensor_names = ("payload",)
    positional_metadata_names = ("tag",)
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
    ) -> Self:
        """Create the wrapper tensor."""
        if outer_size is None:
            outer_size = inner.shape
        if outer_stride is None:
            outer_stride = inner.stride()
        return cast(
            Self,
            torch.Tensor._make_wrapper_subclass(
                cls,
                outer_size,
                outer_stride,
                dtype=inner.dtype,
                device=inner.device,
                layout=layout,
                requires_grad=requires_grad,
            ),
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
    """A wrapper with no flattened fields."""

    def __new__(
        cls,
        outer_size: Sequence[int | torch.SymInt] | None = None,
        outer_stride: Sequence[int | torch.SymInt] | None = None,
    ) -> Self:
        """Create the wrapper tensor."""
        if outer_size is None:
            outer_size = (0,)
        return cast(
            Self,
            torch.Tensor._make_wrapper_subclass(cls, outer_size, outer_stride, dtype=torch.float32),
        )

    def __init__(
        self,
        outer_size: Sequence[int | torch.SymInt] | None = None,
        outer_stride: Sequence[int | torch.SymInt] | None = None,
    ) -> None:
        """Accept the geometry arguments."""


class _RequiresGradFieldWrapper(_Wrapper):
    """A wrapper whose declared requires_grad field collides with the tensor attribute."""

    positional_metadata_names = ("requires_grad",)
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


class _InvalidGradFnWrapper(_Wrapper):
    """A wrapper whose grad_fn access raises a RuntimeError."""

    @property
    def grad_fn(self) -> Any:  # pyrefly: ignore[bad-override]
        """Raise a RuntimeError."""
        raise RuntimeError("grad_fn unavailable")


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


@pytest.mark.parametrize(
    ("default_dtype", "dtype", "dtype_printed"),
    [
        (torch.float32, torch.complex64, False),
        (torch.float32, torch.complex128, True),
        (torch.float32, torch.int64, False),
        (torch.float32, torch.bool, False),
        (torch.double, torch.float64, False),
        (torch.double, torch.complex128, False),
        (torch.double, torch.complex64, True),
        (torch.double, torch.float32, True),
    ],
)
def test_repr_default_dtype_suppression(
    default_dtype: torch.dtype,
    dtype: torch.dtype,
    dtype_printed: bool,
) -> None:
    """Test that inferred default dtypes are omitted from the repr."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(default_dtype)
    try:
        rendered = repr(_Wrapper(torch.ones(2, dtype=dtype)))
    finally:
        torch.set_default_dtype(original_dtype)
    assert rendered.startswith(f"_Wrapper(dtype={dtype},") is dtype_printed


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


def test_repr_invalid_grad_fn() -> None:
    """Test that an invalid grad_fn is rendered as Invalid."""
    wrapper = _InvalidGradFnWrapper(torch.tensor([1.0]))
    assert repr(wrapper).startswith("_InvalidGradFnWrapper(grad_fn=<Invalid>, payload=")


def test_repr_cpp_function_grad_fn() -> None:
    """Test that a CppFunction grad_fn is rendered by its unqualified name."""

    class CppFunction:
        """A stub matching the type name inspected by __repr__."""

        def name(self) -> str:
            """Return a namespace-qualified function name."""
            return "namespace::op"

    class CppFunctionWrapper(_Wrapper):
        """A wrapper whose grad_fn is a C++ function stub."""

        @property
        def grad_fn(self) -> Any:  # pyrefly: ignore[bad-override]
            """Return the C++ function stub."""
            return CppFunction()

    wrapper = CppFunctionWrapper(torch.tensor([1.0]))
    assert repr(wrapper).startswith("CppFunctionWrapper(grad_fn=<op>, payload=")


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


# ------------------------------------------------------------------------------
# __tensor_flatten__
# ------------------------------------------------------------------------------


def test_tensor_flatten_returns_inner_names_and_metadata() -> None:
    """Test that flatten returns inner tensor names and metadata."""
    wrapper = _Wrapper(torch.tensor([1.0, 2.0]), "hello", bias=torch.tensor([3.0, 4.0]), note="world")
    tensor_names, metadata = wrapper.__tensor_flatten__()
    assert tuple(tensor_names) == ("payload", "bias")
    assert metadata == {"tag": "hello", "note": "world"}


def test_tensor_flatten_empty_wrapper() -> None:
    """Test that flatten returns empty names and metadata for a fieldless wrapper."""
    wrapper = _EmptyWrapper()
    tensor_names, metadata = wrapper.__tensor_flatten__()
    assert tuple(tensor_names) == ()
    assert metadata == {}


# ------------------------------------------------------------------------------
# __tensor_unflatten__
# ------------------------------------------------------------------------------


def test_tensor_unflatten_reconstructs_all_field_kinds() -> None:
    """Test that unflatten rebuilds positional and keyword inner tensors and metadata."""
    payload = torch.tensor([1.0, 2.0])
    bias = torch.tensor([3.0, 4.0])
    rebuilt = _Wrapper.__tensor_unflatten__(
        {"bias": bias, "payload": payload},
        {"note": "world", "tag": "hello"},
        (2,),
        (1,),
    )
    assert isinstance(rebuilt, _Wrapper)
    assert rebuilt.payload is payload
    assert rebuilt.bias is bias
    assert rebuilt.tag == "hello"
    assert rebuilt.note == "world"
    assert rebuilt.size() == torch.Size([2])
    assert rebuilt.stride() == (1,)


def test_tensor_unflatten_forwards_outer_geometry() -> None:
    """Test that unflatten passes outer_size and outer_stride to the constructor."""
    rebuilt = _EmptyWrapper.__tensor_unflatten__({}, {}, (4,), (2,))
    assert isinstance(rebuilt, _EmptyWrapper)
    assert rebuilt.size() == torch.Size([4])
    assert rebuilt.stride() == (2,)


def test_tensor_unflatten_empty_wrapper() -> None:
    """Test that unflatten rebuilds a fieldless wrapper."""
    rebuilt = _EmptyWrapper.__tensor_unflatten__({}, {}, (0,), (1,))
    assert isinstance(rebuilt, _EmptyWrapper)
    assert rebuilt.size() == torch.Size([0])
    assert rebuilt.stride() == (1,)


def test_tensor_flatten_unflatten_round_trip() -> None:
    """Test that flatten followed by unflatten preserves fields and geometry."""
    payload = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).t()
    bias = torch.zeros(2, 2)
    tag = "hello"
    note = "world"
    wrapper = _Wrapper(payload, tag, bias=bias, note=note)
    tensor_names, metadata = wrapper.__tensor_flatten__()
    inner_tensors = {name: getattr(wrapper, name) for name in tensor_names}
    rebuilt = type(wrapper).__tensor_unflatten__(inner_tensors, metadata, wrapper.size(), wrapper.stride())
    assert type(rebuilt) is type(wrapper)
    assert rebuilt.payload is wrapper.payload
    assert rebuilt.bias is wrapper.bias
    assert rebuilt.tag is tag
    assert rebuilt.note is note
    assert rebuilt.size() == wrapper.size()
    assert rebuilt.stride() == wrapper.stride()


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


class _DispatchWrapper(MokitWrapperTensorBase):
    """A minimal wrapper used to exercise the dispatch methods."""

    positional_inner_tensor_names = ("payload",)

    def __new__(cls, inner: torch.Tensor) -> torch.Tensor:
        """Create the wrapper tensor."""
        return torch.Tensor._make_wrapper_subclass(
            cls, inner.shape, inner.stride(), dtype=inner.dtype, device=inner.device
        )

    def __init__(self, inner: torch.Tensor) -> None:
        """Store the inner tensor."""
        self.payload = inner


def _new_wrapper_subclass(
    name: str,
    base: type[MokitWrapperTensorBase] = _DispatchWrapper,
) -> type[MokitWrapperTensorBase]:
    """Create a fresh wrapper subclass so tests do not share dispatch state."""
    return cast(type[MokitWrapperTensorBase], type(name, (base,), {}))


def _assert_no_dispatch_state(cls: type[MokitWrapperTensorBase]) -> None:
    """Assert that a class has no own dispatch method or handler table."""
    assert "__torch_dispatch__" not in cls.__dict__
    assert "_ATEN_OP_TABLE" not in cls.__dict__


class _RejectMethodMeta(type(MokitWrapperTensorBase)):
    """A metaclass that rejects __torch_dispatch__ assignment."""

    def __setattr__(cls, name: str, value: Any) -> None:
        """Reject the generated dispatch method."""
        if name == "__torch_dispatch__":
            raise RuntimeError("method installation rejected")
        super().__setattr__(name, value)


class _RejectTableMeta(type(MokitWrapperTensorBase)):
    """A metaclass that rejects _ATEN_OP_TABLE assignment."""

    def __setattr__(cls, name: str, value: Any) -> None:
        """Reject the generated table."""
        if name == "_ATEN_OP_TABLE":
            raise RuntimeError("table installation rejected")
        super().__setattr__(name, value)


class _RecordingHandler:
    """A dispatch handler that records calls and returns a filled tensor."""

    def __init__(self, value: float) -> None:
        """Store the fill value and initialize the call log."""
        self.value = value
        self.calls: list[tuple[Any, ...]] = []

    def __call__(
        self,
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Record the call and return a tensor filled with value."""
        self.calls.append((cls, func, types, args, kwargs))
        first = args[0]
        payload = first.payload if isinstance(first, _DispatchWrapper) else first
        return torch.full_like(payload, self.value)


def _assert_no_torch_function_state(cls: type[MokitWrapperTensorBase]) -> None:
    """Assert that a class has no own torch function method or handler table."""
    assert "__torch_function__" not in cls.__dict__
    assert "_TORCH_FN_TABLE" not in cls.__dict__


class _RejectTorchFunctionMethodMeta(type(MokitWrapperTensorBase)):
    """A metaclass that rejects __torch_function__ assignment."""

    def __setattr__(cls, name: str, value: Any) -> None:
        """Reject the generated torch function method."""
        if name == "__torch_function__":
            raise RuntimeError("method installation rejected")
        super().__setattr__(name, value)


class _RejectTorchFunctionTableMeta(type(MokitWrapperTensorBase)):
    """A metaclass that rejects _TORCH_FN_TABLE assignment."""

    def __setattr__(cls, name: str, value: Any) -> None:
        """Reject the generated torch function table."""
        if name == "_TORCH_FN_TABLE":
            raise RuntimeError("table installation rejected")
        super().__setattr__(name, value)


# ------------------------------------------------------------------------------
# implements
# ------------------------------------------------------------------------------


def test_implements_returns_a_side_effect_free_decorator() -> None:
    """Test that the factory validates without mutating the class until applied."""
    sub = _new_wrapper_subclass("DecoratorTarget")
    handler = _RecordingHandler(1.0)
    decorator = sub.implements(torch.ops.aten.neg.default)
    assert "__torch_dispatch__" not in sub.__dict__
    assert "_ATEN_OP_TABLE" not in sub.__dict__
    assert decorator(handler) is handler
    assert isinstance(sub.__dict__["__torch_dispatch__"], classmethod)
    assert {torch.ops.aten.neg.default: handler} == sub._ATEN_OP_TABLE


def test_implements_rejects_non_bool_override() -> None:
    """Test that override must be a bool and validation leaves the class unchanged."""
    sub = _new_wrapper_subclass("InvalidOverride")
    with pytest.raises(TypeError, match="override"):
        sub.implements(torch.ops.aten.neg.default, override=cast(Any, 1))
    _assert_no_dispatch_state(sub)


def test_implements_rejects_op_overload_packet() -> None:
    """Test that an OpOverloadPacket is rejected without mutating the class."""
    sub = _new_wrapper_subclass("PacketKey")
    with pytest.raises(TypeError, match="OpOverload"):
        sub.implements(cast(Any, torch.ops.aten.neg))
    _assert_no_dispatch_state(sub)


def test_implements_rejects_non_op_overload() -> None:
    """Test that a plain callable is rejected without mutating the class."""
    sub = _new_wrapper_subclass("NonOpKey")
    with pytest.raises(TypeError, match="OpOverload"):
        sub.implements(cast(Any, torch.neg))
    _assert_no_dispatch_state(sub)


def test_implements_rejects_duplicate_keys() -> None:
    """Test that duplicate operators are rejected without mutating the class."""
    sub = _new_wrapper_subclass("DuplicateKeys")
    op = torch.ops.aten.neg.default
    with pytest.raises(KeyError, match="unique"):
        sub.implements(op, op)
    _assert_no_dispatch_state(sub)


def test_implements_rejects_non_callable_handler() -> None:
    """Test that a non-callable handler leaves the class unchanged."""
    sub = _new_wrapper_subclass("NonCallable")
    with pytest.raises(TypeError, match="callable"):
        sub.implements(torch.ops.aten.neg.default)(cast(Any, 42))
    _assert_no_dispatch_state(sub)


def test_implements_installs_dispatch_method_and_table() -> None:
    """Test that registration installs a classmethod and an own table."""
    sub = _new_wrapper_subclass("Installs")
    handler = _RecordingHandler(3.0)
    op = torch.ops.aten.neg.default
    sub.implements(op)(handler)
    assert isinstance(sub.__dict__["__torch_dispatch__"], classmethod)
    assert sub.__dict__["_ATEN_OP_TABLE"] == {op: handler}
    inner = torch.tensor([1.0, 2.0])
    assert torch.equal(op(sub(inner)), torch.full_like(inner, 3.0))


def test_implements_registers_multiple_keys() -> None:
    """Test that one decorator registers every provided operator."""
    sub = _new_wrapper_subclass("MultipleKeys")
    handler = _RecordingHandler(4.0)
    neg = torch.ops.aten.neg.default
    absolute = torch.ops.aten.abs.default
    sub.implements(neg, absolute)(handler)
    assert {neg: handler, absolute: handler} == sub._ATEN_OP_TABLE
    inner = torch.tensor([-1.0, 2.0])
    wrapper = sub(inner)
    assert torch.equal(neg(wrapper), torch.full_like(inner, 4.0))
    assert torch.equal(absolute(wrapper), torch.full_like(inner, 4.0))


def test_implements_registers_wildcard() -> None:
    """Test that an empty operator list registers the None wildcard."""
    sub = _new_wrapper_subclass("Wildcard")
    handler = _RecordingHandler(5.0)
    sub.implements()(handler)
    assert {None: handler} == sub._ATEN_OP_TABLE
    inner = torch.tensor([1.0, 2.0])
    assert torch.equal(torch.ops.aten.neg.default(sub(inner)), torch.full_like(inner, 5.0))


def test_implements_rejects_conflicting_key_without_override() -> None:
    """Test that a conflict raises and leaves the original handler in place."""
    sub = _new_wrapper_subclass("Conflict")
    first = _RecordingHandler(1.0)
    second = _RecordingHandler(2.0)
    op = torch.ops.aten.neg.default
    sub.implements(op)(first)
    table = sub._ATEN_OP_TABLE
    snapshot = dict(table)
    with pytest.raises(KeyError, match="already registered"):
        sub.implements(op)(second)
    assert sub._ATEN_OP_TABLE is table
    assert dict(sub._ATEN_OP_TABLE) == snapshot
    assert {op: first} == sub._ATEN_OP_TABLE
    inner = torch.tensor([1.0])
    assert torch.equal(op(sub(inner)), torch.full_like(inner, 1.0))


def test_implements_override_replaces_handler() -> None:
    """Test that override=True replaces an existing handler in place."""
    sub = _new_wrapper_subclass("Override")
    first = _RecordingHandler(1.0)
    second = _RecordingHandler(2.0)
    op = torch.ops.aten.neg.default
    sub.implements(op)(first)
    table = sub._ATEN_OP_TABLE
    sub.implements(op, override=True)(second)
    assert sub._ATEN_OP_TABLE is table
    assert {op: second} == sub._ATEN_OP_TABLE
    inner = torch.tensor([1.0])
    assert torch.equal(op(sub(inner)), torch.full_like(inner, 2.0))


def test_implements_requires_own_table_for_custom_dispatch_method() -> None:
    """Test that a custom dispatch method without an own table is rejected."""

    class Custom(MokitWrapperTensorBase):
        """A wrapper with a custom dispatch method and no table."""

        @classmethod
        def __torch_dispatch__(
            cls,
            func: OpOverload,
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            """Return a sentinel."""
            return None

    original = Custom.__dict__["__torch_dispatch__"]
    with pytest.raises(KeyError, match="_ATEN_OP_TABLE"):
        Custom.implements(torch.ops.aten.neg.default)(_RecordingHandler(1.0))
    assert Custom.__dict__["__torch_dispatch__"] is original
    assert "_ATEN_OP_TABLE" not in Custom.__dict__


def test_implements_rejects_non_dict_table() -> None:
    """Test that a non-dict table is rejected without installing a method."""

    class BadTable(MokitWrapperTensorBase):
        """A wrapper with a non-dict dispatch table."""

        _ATEN_OP_TABLE = cast(Any, [])

    table = BadTable._ATEN_OP_TABLE
    snapshot = list(table)
    with pytest.raises(TypeError, match="must be a dict"):
        BadTable.implements(torch.ops.aten.neg.default)(_RecordingHandler(1.0))
    assert BadTable._ATEN_OP_TABLE is table
    assert list(BadTable._ATEN_OP_TABLE) == snapshot
    assert "__torch_dispatch__" not in BadTable.__dict__


def test_implements_rejects_dict_subclass_table() -> None:
    """Test that a dict subclass is rejected as a handler table."""

    class DictTable(dict[Any, Any]):
        """A dict subclass used as an invalid dispatch table."""

    class Sub(MokitWrapperTensorBase):
        """A wrapper with a dict-subclass dispatch table."""

        _ATEN_OP_TABLE: ClassVar[dict[Any, Any]] = DictTable()

    table = Sub._ATEN_OP_TABLE
    snapshot = dict(table)
    with pytest.raises(TypeError, match="must be a dict"):
        Sub.implements(torch.ops.aten.neg.default)(_RecordingHandler(1.0))
    assert Sub._ATEN_OP_TABLE is table
    assert dict(Sub._ATEN_OP_TABLE) == snapshot
    assert "__torch_dispatch__" not in Sub.__dict__


def test_implements_keeps_custom_dispatch_method() -> None:
    """Test that registration updates an own table without replacing a custom method."""

    class CustomTable(MokitWrapperTensorBase):
        """A wrapper with both a custom dispatch method and an own table."""

        _ATEN_OP_TABLE: ClassVar[dict[Any, Any]] = {}

        @classmethod
        def __torch_dispatch__(
            cls,
            func: OpOverload,
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            """Return a sentinel."""
            return None

    original = CustomTable.__dict__["__torch_dispatch__"]
    handler = _RecordingHandler(1.0)
    CustomTable.implements(torch.ops.aten.neg.default)(handler)
    assert CustomTable.__dict__["__torch_dispatch__"] is original
    assert {torch.ops.aten.neg.default: handler} == CustomTable._ATEN_OP_TABLE


def test_implements_installs_method_for_table_only_class() -> None:
    """Test that a class with an own table and no method gets the generated method."""

    class TableOnly(MokitWrapperTensorBase):
        """A wrapper with an own dispatch table and no dispatch method."""

        _ATEN_OP_TABLE: ClassVar[dict[Any, Any]] = {}

    handler = _RecordingHandler(1.0)
    TableOnly.implements(torch.ops.aten.neg.default)(handler)
    assert isinstance(TableOnly.__dict__["__torch_dispatch__"], classmethod)
    assert {torch.ops.aten.neg.default: handler} == TableOnly._ATEN_OP_TABLE


def test_implements_table_only_conflict_keeps_method_uninstalled() -> None:
    """Test that a table-only conflict raises before installing a method."""

    class TableConflict(MokitWrapperTensorBase):
        """A table-only wrapper with an existing handler."""

        _ATEN_OP_TABLE: ClassVar[dict[Any, Any]] = {}

    op = torch.ops.aten.neg.default
    existing = _RecordingHandler(1.0)
    candidate = _RecordingHandler(2.0)
    TableConflict._ATEN_OP_TABLE[op] = existing
    table = TableConflict._ATEN_OP_TABLE
    snapshot = dict(table)
    with pytest.raises(KeyError, match="already registered"):
        TableConflict.implements(op)(candidate)
    assert TableConflict._ATEN_OP_TABLE is table
    assert dict(TableConflict._ATEN_OP_TABLE) == snapshot
    assert {op: existing} == TableConflict._ATEN_OP_TABLE
    assert "__torch_dispatch__" not in TableConflict.__dict__


def test_implements_table_only_override_installs_method() -> None:
    """Test that a table-only override installs a method and updates the table."""

    class TableOverride(_DispatchWrapper):
        """A table-only wrapper with an existing handler."""

        _ATEN_OP_TABLE: ClassVar[dict[Any, Any]] = {}

    op = torch.ops.aten.neg.default
    first = _RecordingHandler(1.0)
    second = _RecordingHandler(2.0)
    TableOverride._ATEN_OP_TABLE[op] = first
    table = TableOverride._ATEN_OP_TABLE
    TableOverride.implements(op, override=True)(second)
    assert isinstance(TableOverride.__dict__["__torch_dispatch__"], classmethod)
    assert TableOverride._ATEN_OP_TABLE is table
    assert {op: second} == TableOverride._ATEN_OP_TABLE
    inner = torch.tensor([1.0])
    assert torch.equal(op(TableOverride(inner)), torch.full_like(inner, 2.0))


def test_implements_rejects_explicit_none() -> None:
    """Test that an explicit None key is rejected without mutating the class."""
    sub = _new_wrapper_subclass("ExplicitNone")
    with pytest.raises(TypeError, match="OpOverload"):
        sub.implements(cast(Any, None))
    _assert_no_dispatch_state(sub)


def test_implements_wildcard_conflict_and_override() -> None:
    """Test wildcard conflict and override replacement."""
    sub = _new_wrapper_subclass("WildcardOverride")
    first = _RecordingHandler(1.0)
    second = _RecordingHandler(2.0)
    sub.implements()(first)
    table = sub._ATEN_OP_TABLE
    snapshot = dict(table)
    with pytest.raises(KeyError, match="already registered"):
        sub.implements()(second)
    assert sub._ATEN_OP_TABLE is table
    assert dict(sub._ATEN_OP_TABLE) == snapshot
    assert {None: first} == sub._ATEN_OP_TABLE
    sub.implements(override=True)(second)
    assert sub._ATEN_OP_TABLE is table
    assert {None: second} == sub._ATEN_OP_TABLE
    inner = torch.tensor([1.0])
    assert torch.equal(torch.ops.aten.neg.default(sub(inner)), torch.full_like(inner, 2.0))


def test_implements_multi_key_conflict_leaves_class_unchanged() -> None:
    """Test that a later key conflict prevents every key in the call from registering."""
    sub = _new_wrapper_subclass("MultiKeyConflict")
    first = _RecordingHandler(1.0)
    second = _RecordingHandler(2.0)
    neg = torch.ops.aten.neg.default
    absolute = torch.ops.aten.abs.default
    sub.implements(neg)(first)
    with pytest.raises(KeyError, match="already registered"):
        sub.implements(neg, absolute)(second)
    assert {neg: first} == sub._ATEN_OP_TABLE
    assert absolute not in sub._ATEN_OP_TABLE
    inner = torch.tensor([1.0])
    assert torch.equal(neg(sub(inner)), torch.full_like(inner, 1.0))


def test_implements_installs_method_over_inherited_custom_dispatch() -> None:
    """Test that registration installs a generated method over an inherited custom one."""

    class CustomParent(_DispatchWrapper):
        """A wrapper with a custom dispatch method."""

        @classmethod
        def __torch_dispatch__(
            cls,
            func: OpOverload,
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            """Return a sentinel."""
            return None

    class Child(CustomParent):
        """A subclass that registers its first handler."""

    parent_method = CustomParent.__dict__["__torch_dispatch__"]
    handler = _RecordingHandler(3.0)
    op = torch.ops.aten.neg.default
    Child.implements(op)(handler)
    assert isinstance(Child.__dict__["__torch_dispatch__"], classmethod)
    assert Child.__dict__["__torch_dispatch__"] is not parent_method
    assert {op: handler} == Child._ATEN_OP_TABLE
    assert CustomParent.__dict__["__torch_dispatch__"] is parent_method
    inner = torch.tensor([1.0])
    assert torch.equal(op(Child(inner)), torch.full_like(inner, 3.0))


def test_implements_child_owns_method_and_table() -> None:
    """Test that a child registering handlers owns its method and table."""
    parent = _new_wrapper_subclass("OwnParent")
    child = _new_wrapper_subclass("OwnChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_handler = _RecordingHandler(2.0)
    op = torch.ops.aten.neg.default
    parent.implements(op)(parent_handler)
    parent_method = parent.__dict__["__torch_dispatch__"]
    child.implements(op)(child_handler)
    assert "__torch_dispatch__" in child.__dict__
    assert child.__dict__["__torch_dispatch__"] is not parent_method
    assert {op: child_handler} == child._ATEN_OP_TABLE
    assert {op: parent_handler} == parent._ATEN_OP_TABLE
    assert parent.__dict__["__torch_dispatch__"] is parent_method
    inner = torch.tensor([1.0])
    assert torch.equal(op(parent(inner)), torch.full_like(inner, 1.0))
    assert len(parent_handler.calls) == 1
    assert child_handler.calls == []
    assert torch.equal(op(child(inner)), torch.full_like(inner, 2.0))
    assert len(parent_handler.calls) == 1
    assert len(child_handler.calls) == 1


def test_implements_method_install_failure_is_raised() -> None:
    """Test that a fatal method-install failure propagates without being swallowed."""

    class FatalMethod(MokitWrapperTensorBase, metaclass=_RejectMethodMeta):
        """A wrapper whose metaclass rejects method installation."""

    with pytest.raises(RuntimeError, match="method installation rejected"):
        FatalMethod.implements(torch.ops.aten.neg.default)(_RecordingHandler(1.0))
    _assert_no_dispatch_state(FatalMethod)


def test_implements_method_install_failure_preserves_existing_table() -> None:
    """Test that a method-install failure leaves an existing table unchanged."""

    class FatalMethod(MokitWrapperTensorBase, metaclass=_RejectMethodMeta):
        """A wrapper whose metaclass rejects method installation."""

        _ATEN_OP_TABLE: ClassVar[dict[Any, Any]] = {}

    table = FatalMethod._ATEN_OP_TABLE
    snapshot = dict(table)
    with pytest.raises(RuntimeError, match="method installation rejected"):
        FatalMethod.implements(torch.ops.aten.neg.default)(_RecordingHandler(1.0))
    assert FatalMethod.__dict__["_ATEN_OP_TABLE"] is table
    assert dict(FatalMethod._ATEN_OP_TABLE) == snapshot
    assert "__torch_dispatch__" not in FatalMethod.__dict__


def test_implements_table_install_failure_is_raised() -> None:
    """Test that a fatal table-install failure propagates and leaves the method installed."""

    class FatalTable(MokitWrapperTensorBase, metaclass=_RejectTableMeta):
        """A wrapper whose metaclass rejects table installation."""

    with pytest.raises(RuntimeError, match="table installation rejected"):
        FatalTable.implements(torch.ops.aten.neg.default)(_RecordingHandler(1.0))
    assert "__torch_dispatch__" in FatalTable.__dict__
    assert "_ATEN_OP_TABLE" not in FatalTable.__dict__


# ------------------------------------------------------------------------------
# implements_torch_function
# ------------------------------------------------------------------------------


def test_implements_torch_function_returns_a_side_effect_free_decorator() -> None:
    """Test that the factory validates without mutating the class until applied."""
    sub = _new_wrapper_subclass("TorchFnDecoratorTarget")
    handler = _RecordingHandler(1.0)
    decorator = sub.implements_torch_function(torch.neg)
    assert "__torch_function__" not in sub.__dict__
    assert "_TORCH_FN_TABLE" not in sub.__dict__
    assert decorator(handler) is handler
    assert isinstance(sub.__dict__["__torch_function__"], classmethod)
    assert {torch.neg: handler} == sub._TORCH_FN_TABLE


def test_implements_torch_function_rejects_non_bool_override() -> None:
    """Test that override must be a bool and validation leaves the class unchanged."""
    sub = _new_wrapper_subclass("TorchFnInvalidOverride")
    with pytest.raises(TypeError, match="override"):
        sub.implements_torch_function(torch.neg, override=cast(Any, 1))
    _assert_no_torch_function_state(sub)


def test_implements_torch_function_accepts_plain_callable_key() -> None:
    """Test that any callable, not only torch functions, can be registered."""
    sub = _new_wrapper_subclass("TorchFnPlainKey")
    handler = _RecordingHandler(2.0)

    def custom_function(tensor: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged."""
        return tensor

    sub.implements_torch_function(custom_function)(handler)
    inner = torch.tensor([1.0])
    wrapper = sub(inner)
    result = sub.__torch_function__(custom_function, (sub,), (wrapper,), None)
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert handler.calls[0][1] is custom_function


def test_implements_torch_function_rejects_non_callable_key() -> None:
    """Test that a non-callable key is rejected without mutating the class."""
    sub = _new_wrapper_subclass("TorchFnNonCallableKey")
    with pytest.raises(TypeError, match="callable"):
        sub.implements_torch_function(cast(Any, 42))
    _assert_no_torch_function_state(sub)


def test_implements_torch_function_rejects_unhashable_key() -> None:
    """Test that an unhashable callable key is rejected without mutating the class."""
    sub = _new_wrapper_subclass("TorchFnUnhashableKey")

    class Unhashable:
        """A callable whose instances are unhashable."""

        __hash__ = None  # type: ignore[assignment]

        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            """Return the input unchanged."""
            return tensor

    with pytest.raises(TypeError, match="hashable"):
        sub.implements_torch_function(cast(Any, Unhashable()))
    _assert_no_torch_function_state(sub)


def test_implements_torch_function_rejects_duplicate_keys() -> None:
    """Test that duplicate torch functions are rejected without mutating the class."""
    sub = _new_wrapper_subclass("TorchFnDuplicateKeys")
    with pytest.raises(KeyError, match="unique"):
        sub.implements_torch_function(torch.neg, torch.neg)
    _assert_no_torch_function_state(sub)


def test_implements_torch_function_rejects_non_callable_handler() -> None:
    """Test that a non-callable handler leaves the class unchanged."""
    sub = _new_wrapper_subclass("TorchFnNonCallableHandler")
    with pytest.raises(TypeError, match="callable"):
        sub.implements_torch_function(torch.neg)(cast(Any, 42))
    _assert_no_torch_function_state(sub)


def test_implements_torch_function_installs_dispatch_method_and_table() -> None:
    """Test that registration installs a classmethod and an own table."""
    sub = _new_wrapper_subclass("TorchFnInstalls")
    handler = _RecordingHandler(3.0)
    sub.implements_torch_function(torch.neg)(handler)
    assert isinstance(sub.__dict__["__torch_function__"], classmethod)
    assert sub.__dict__["_TORCH_FN_TABLE"] == {torch.neg: handler}
    inner = torch.tensor([1.0, 2.0])
    assert torch.equal(torch.neg(sub(inner)), torch.full_like(inner, 3.0))


def test_implements_torch_function_registers_multiple_keys() -> None:
    """Test that one decorator registers every provided torch function."""
    sub = _new_wrapper_subclass("TorchFnMultipleKeys")
    handler = _RecordingHandler(4.0)
    sub.implements_torch_function(torch.neg, torch.abs)(handler)
    assert {torch.neg: handler, torch.abs: handler} == sub._TORCH_FN_TABLE
    inner = torch.tensor([-1.0, 2.0])
    wrapper = sub(inner)
    assert torch.equal(torch.neg(wrapper), torch.full_like(inner, 4.0))
    assert torch.equal(torch.abs(wrapper), torch.full_like(inner, 4.0))


def test_implements_torch_function_registers_wildcard() -> None:
    """Test that an empty torch function list registers the None wildcard."""
    sub = _new_wrapper_subclass("TorchFnWildcard")
    handler = _RecordingHandler(5.0)
    sub.implements_torch_function()(handler)
    assert {None: handler} == sub._TORCH_FN_TABLE
    inner = torch.tensor([1.0, 2.0])
    assert torch.equal(torch.neg(sub(inner)), torch.full_like(inner, 5.0))


def test_implements_torch_function_rejects_conflicting_key_without_override() -> None:
    """Test that a conflict raises and leaves the original handler in place."""
    sub = _new_wrapper_subclass("TorchFnConflict")
    first = _RecordingHandler(1.0)
    second = _RecordingHandler(2.0)
    sub.implements_torch_function(torch.neg)(first)
    table = sub._TORCH_FN_TABLE
    snapshot = dict(table)
    with pytest.raises(KeyError, match="already registered"):
        sub.implements_torch_function(torch.neg)(second)
    assert sub._TORCH_FN_TABLE is table
    assert dict(sub._TORCH_FN_TABLE) == snapshot
    assert {torch.neg: first} == sub._TORCH_FN_TABLE
    inner = torch.tensor([1.0])
    assert torch.equal(torch.neg(sub(inner)), torch.full_like(inner, 1.0))


def test_implements_torch_function_override_replaces_handler() -> None:
    """Test that override=True replaces an existing handler in place."""
    sub = _new_wrapper_subclass("TorchFnOverride")
    first = _RecordingHandler(1.0)
    second = _RecordingHandler(2.0)
    sub.implements_torch_function(torch.neg)(first)
    table = sub._TORCH_FN_TABLE
    sub.implements_torch_function(torch.neg, override=True)(second)
    assert sub._TORCH_FN_TABLE is table
    assert {torch.neg: second} == sub._TORCH_FN_TABLE
    inner = torch.tensor([1.0])
    assert torch.equal(torch.neg(sub(inner)), torch.full_like(inner, 2.0))


def test_implements_torch_function_requires_own_table_for_custom_method() -> None:
    """Test that a custom torch function method without an own table is rejected."""

    class Custom(MokitWrapperTensorBase):
        """A wrapper with a custom torch function method and no table."""

        @classmethod
        def __torch_function__(
            cls,
            func: Callable[..., Any],
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            """Return a sentinel."""
            return None

    original = Custom.__dict__["__torch_function__"]
    with pytest.raises(KeyError, match="_TORCH_FN_TABLE"):
        Custom.implements_torch_function(torch.neg)(_RecordingHandler(1.0))
    assert Custom.__dict__["__torch_function__"] is original
    assert "_TORCH_FN_TABLE" not in Custom.__dict__


def test_implements_torch_function_rejects_non_dict_table() -> None:
    """Test that a non-dict table is rejected without installing a method."""

    class BadTable(MokitWrapperTensorBase):
        """A wrapper with a non-dict torch function table."""

        _TORCH_FN_TABLE = cast(Any, [])

    table = BadTable._TORCH_FN_TABLE
    snapshot = list(table)
    with pytest.raises(TypeError, match="must be a dict"):
        BadTable.implements_torch_function(torch.neg)(_RecordingHandler(1.0))
    assert BadTable._TORCH_FN_TABLE is table
    assert list(BadTable._TORCH_FN_TABLE) == snapshot
    assert "__torch_function__" not in BadTable.__dict__


def test_implements_torch_function_rejects_dict_subclass_table() -> None:
    """Test that a dict subclass is rejected as a torch function table."""

    class DictTable(dict[Any, Any]):
        """A dict subclass used as an invalid torch function table."""

    class Sub(MokitWrapperTensorBase):
        """A wrapper with a dict-subclass torch function table."""

        _TORCH_FN_TABLE: ClassVar[dict[Any, Any]] = DictTable()

    table = Sub._TORCH_FN_TABLE
    snapshot = dict(table)
    with pytest.raises(TypeError, match="must be a dict"):
        Sub.implements_torch_function(torch.neg)(_RecordingHandler(1.0))
    assert Sub._TORCH_FN_TABLE is table
    assert dict(Sub._TORCH_FN_TABLE) == snapshot
    assert "__torch_function__" not in Sub.__dict__


def test_implements_torch_function_keeps_custom_method() -> None:
    """Test that registration updates an own table without replacing a custom method."""

    class CustomTable(MokitWrapperTensorBase):
        """A wrapper with both a custom torch function method and an own table."""

        _TORCH_FN_TABLE: ClassVar[dict[Any, Any]] = {}

        @classmethod
        def __torch_function__(
            cls,
            func: Callable[..., Any],
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            """Return a sentinel."""
            return None

    original = CustomTable.__dict__["__torch_function__"]
    handler = _RecordingHandler(1.0)
    CustomTable.implements_torch_function(torch.neg)(handler)
    assert CustomTable.__dict__["__torch_function__"] is original
    assert {torch.neg: handler} == CustomTable._TORCH_FN_TABLE


def test_custom_torch_function_method_shadows_registered_handlers() -> None:
    """Test that a custom torch function method makes registered handlers unreachable."""

    class Custom(_DispatchWrapper):
        """A wrapper with a custom torch function method and an own table."""

        _TORCH_FN_TABLE: ClassVar[dict[Any, Any]] = {}

        @classmethod
        def __torch_function__(
            cls,
            func: Callable[..., Any],
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> str:
            """Return a sentinel."""
            return "custom"

    handler = _RecordingHandler(1.0)
    Custom.implements_torch_function(torch.neg)(handler)
    result = torch.neg(Custom(torch.tensor([1.0])))
    assert result == "custom"
    assert handler.calls == []


def test_implements_torch_function_installs_method_for_table_only_class() -> None:
    """Test that a class with an own table and no method gets the generated method."""

    class TableOnly(MokitWrapperTensorBase):
        """A wrapper with an own torch function table and no method."""

        _TORCH_FN_TABLE: ClassVar[dict[Any, Any]] = {}

    handler = _RecordingHandler(1.0)
    TableOnly.implements_torch_function(torch.neg)(handler)
    assert isinstance(TableOnly.__dict__["__torch_function__"], classmethod)
    assert {torch.neg: handler} == TableOnly._TORCH_FN_TABLE


def test_implements_torch_function_table_only_conflict_keeps_method_uninstalled() -> None:
    """Test that a table-only conflict raises before installing a method."""

    class TableConflict(MokitWrapperTensorBase):
        """A table-only wrapper with an existing handler."""

        _TORCH_FN_TABLE: ClassVar[dict[Any, Any]] = {}

    existing = _RecordingHandler(1.0)
    candidate = _RecordingHandler(2.0)
    TableConflict._TORCH_FN_TABLE[torch.neg] = existing
    table = TableConflict._TORCH_FN_TABLE
    snapshot = dict(table)
    with pytest.raises(KeyError, match="already registered"):
        TableConflict.implements_torch_function(torch.neg)(candidate)
    assert TableConflict._TORCH_FN_TABLE is table
    assert dict(TableConflict._TORCH_FN_TABLE) == snapshot
    assert {torch.neg: existing} == TableConflict._TORCH_FN_TABLE
    assert "__torch_function__" not in TableConflict.__dict__


def test_implements_torch_function_table_only_override_installs_method() -> None:
    """Test that a table-only override installs a method and updates the table."""

    class TableOverride(_DispatchWrapper):
        """A table-only wrapper with an existing handler."""

        _TORCH_FN_TABLE: ClassVar[dict[Any, Any]] = {}

    first = _RecordingHandler(1.0)
    second = _RecordingHandler(2.0)
    TableOverride._TORCH_FN_TABLE[torch.neg] = first
    table = TableOverride._TORCH_FN_TABLE
    TableOverride.implements_torch_function(torch.neg, override=True)(second)
    assert isinstance(TableOverride.__dict__["__torch_function__"], classmethod)
    assert TableOverride._TORCH_FN_TABLE is table
    assert {torch.neg: second} == TableOverride._TORCH_FN_TABLE
    inner = torch.tensor([1.0])
    assert torch.equal(torch.neg(TableOverride(inner)), torch.full_like(inner, 2.0))


def test_implements_torch_function_rejects_explicit_none() -> None:
    """Test that an explicit None key is rejected without mutating the class."""
    sub = _new_wrapper_subclass("TorchFnExplicitNone")
    with pytest.raises(TypeError, match="callable"):
        sub.implements_torch_function(cast(Any, None))
    _assert_no_torch_function_state(sub)


def test_implements_torch_function_wildcard_conflict_and_override() -> None:
    """Test wildcard conflict and override replacement."""
    sub = _new_wrapper_subclass("TorchFnWildcardOverride")
    first = _RecordingHandler(1.0)
    second = _RecordingHandler(2.0)
    sub.implements_torch_function()(first)
    table = sub._TORCH_FN_TABLE
    snapshot = dict(table)
    with pytest.raises(KeyError, match="already registered"):
        sub.implements_torch_function()(second)
    assert sub._TORCH_FN_TABLE is table
    assert dict(sub._TORCH_FN_TABLE) == snapshot
    assert {None: first} == sub._TORCH_FN_TABLE
    sub.implements_torch_function(override=True)(second)
    assert sub._TORCH_FN_TABLE is table
    assert {None: second} == sub._TORCH_FN_TABLE
    inner = torch.tensor([1.0])
    assert torch.equal(torch.neg(sub(inner)), torch.full_like(inner, 2.0))


def test_implements_torch_function_multi_key_conflict_leaves_class_unchanged() -> None:
    """Test that a later key conflict prevents every key in the call from registering."""
    sub = _new_wrapper_subclass("TorchFnMultiKeyConflict")
    first = _RecordingHandler(1.0)
    second = _RecordingHandler(2.0)
    sub.implements_torch_function(torch.neg)(first)
    with pytest.raises(KeyError, match="already registered"):
        sub.implements_torch_function(torch.neg, torch.abs)(second)
    assert {torch.neg: first} == sub._TORCH_FN_TABLE
    assert torch.abs not in sub._TORCH_FN_TABLE
    inner = torch.tensor([1.0])
    assert torch.equal(torch.neg(sub(inner)), torch.full_like(inner, 1.0))


def test_implements_torch_function_installs_method_over_inherited_custom_method() -> None:
    """Test that registration installs a generated method over an inherited custom one."""

    class CustomParent(_DispatchWrapper):
        """A wrapper with a custom torch function method."""

        @classmethod
        def __torch_function__(
            cls,
            func: Callable[..., Any],
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            """Return a sentinel."""
            return None

    class Child(CustomParent):
        """A subclass that registers its first handler."""

    parent_method = CustomParent.__dict__["__torch_function__"]
    handler = _RecordingHandler(3.0)
    Child.implements_torch_function(torch.neg)(handler)
    assert isinstance(Child.__dict__["__torch_function__"], classmethod)
    assert Child.__dict__["__torch_function__"] is not parent_method
    assert {torch.neg: handler} == Child._TORCH_FN_TABLE
    assert CustomParent.__dict__["__torch_function__"] is parent_method
    inner = torch.tensor([1.0])
    assert torch.equal(torch.neg(Child(inner)), torch.full_like(inner, 3.0))


def test_implements_torch_function_child_owns_method_and_table() -> None:
    """Test that a child registering handlers owns its method and table."""
    parent = _new_wrapper_subclass("TorchFnOwnParent")
    child = _new_wrapper_subclass("TorchFnOwnChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_handler = _RecordingHandler(2.0)
    parent.implements_torch_function(torch.neg)(parent_handler)
    parent_method = parent.__dict__["__torch_function__"]
    child.implements_torch_function(torch.neg)(child_handler)
    assert "__torch_function__" in child.__dict__
    assert child.__dict__["__torch_function__"] is not parent_method
    assert {torch.neg: child_handler} == child._TORCH_FN_TABLE
    assert {torch.neg: parent_handler} == parent._TORCH_FN_TABLE
    assert parent.__dict__["__torch_function__"] is parent_method
    inner = torch.tensor([1.0])
    assert torch.equal(torch.neg(parent(inner)), torch.full_like(inner, 1.0))
    assert len(parent_handler.calls) == 1
    assert child_handler.calls == []
    assert torch.equal(torch.neg(child(inner)), torch.full_like(inner, 2.0))
    assert len(parent_handler.calls) == 1
    assert len(child_handler.calls) == 1


def test_implements_torch_function_method_install_failure_is_raised() -> None:
    """Test that a fatal method-install failure propagates without being swallowed."""

    class FatalMethod(MokitWrapperTensorBase, metaclass=_RejectTorchFunctionMethodMeta):
        """A wrapper whose metaclass rejects method installation."""

    with pytest.raises(RuntimeError, match="method installation rejected"):
        FatalMethod.implements_torch_function(torch.neg)(_RecordingHandler(1.0))
    _assert_no_torch_function_state(FatalMethod)


def test_implements_torch_function_method_install_failure_preserves_existing_table() -> None:
    """Test that a method-install failure leaves an existing table unchanged."""

    class FatalMethod(MokitWrapperTensorBase, metaclass=_RejectTorchFunctionMethodMeta):
        """A wrapper whose metaclass rejects method installation."""

        _TORCH_FN_TABLE: ClassVar[dict[Any, Any]] = {}

    table = FatalMethod._TORCH_FN_TABLE
    snapshot = dict(table)
    with pytest.raises(RuntimeError, match="method installation rejected"):
        FatalMethod.implements_torch_function(torch.neg)(_RecordingHandler(1.0))
    assert FatalMethod.__dict__["_TORCH_FN_TABLE"] is table
    assert dict(FatalMethod._TORCH_FN_TABLE) == snapshot
    assert "__torch_function__" not in FatalMethod.__dict__


def test_implements_torch_function_table_install_failure_is_raised() -> None:
    """Test that a fatal table-install failure propagates and leaves the method installed."""

    class FatalTable(MokitWrapperTensorBase, metaclass=_RejectTorchFunctionTableMeta):
        """A wrapper whose metaclass rejects table installation."""

    with pytest.raises(RuntimeError, match="table installation rejected"):
        FatalTable.implements_torch_function(torch.neg)(_RecordingHandler(1.0))
    assert "__torch_function__" in FatalTable.__dict__
    assert "_TORCH_FN_TABLE" not in FatalTable.__dict__


# ------------------------------------------------------------------------------
# __torch_dispatch__
# ------------------------------------------------------------------------------


def test_torch_dispatch_exact_handler_forwards_arguments() -> None:
    """Test that an exact handler receives cls, func, types, args, and kwargs."""
    sub = _new_wrapper_subclass("Exact")
    handler = _RecordingHandler(2.0)
    op = torch.ops.aten.neg.default
    sub.implements(op)(handler)
    inner = torch.tensor([1.0, 2.0])
    wrapper = sub(inner)
    result = op(wrapper)
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert len(handler.calls) == 1
    cls, func, types, args, kwargs = handler.calls[0]
    assert cls is sub
    assert func is op
    assert types == (sub,)
    assert len(args) == 1 and args[0] is wrapper
    assert kwargs == {}


def test_torch_dispatch_wildcard_handler() -> None:
    """Test that a wildcard handler receives operators with no exact handler."""
    sub = _new_wrapper_subclass("WildcardDispatch")
    handler = _RecordingHandler(3.0)
    sub.implements()(handler)
    inner = torch.tensor([1.0, 2.0])
    wrapper = sub(inner)
    result = torch.ops.aten.neg.default(wrapper)
    assert torch.equal(result, torch.full_like(inner, 3.0))
    cls, func, types, args, kwargs = handler.calls[0]
    assert cls is sub
    assert func is torch.ops.aten.neg.default
    assert types == (sub,)
    assert len(args) == 1 and args[0] is wrapper
    assert kwargs == {}


def test_torch_dispatch_exact_precedes_wildcard() -> None:
    """Test that an exact handler wins over a wildcard on the same class."""
    sub = _new_wrapper_subclass("ExactPrecedence")
    exact = _RecordingHandler(1.0)
    wildcard = _RecordingHandler(2.0)
    op = torch.ops.aten.neg.default
    sub.implements()(wildcard)
    sub.implements(op)(exact)
    inner = torch.tensor([1.0])
    wrapper = sub(inner)
    assert torch.equal(op(wrapper), torch.full_like(inner, 1.0))
    assert len(exact.calls) == 1
    assert wildcard.calls == []
    assert torch.equal(torch.ops.aten.abs.default(wrapper), torch.full_like(inner, 2.0))
    assert len(wildcard.calls) == 1


def test_torch_dispatch_forwards_kwargs() -> None:
    """Test that keyword arguments reach the handler."""
    sub = _new_wrapper_subclass("Kwargs")
    seen_kwargs: list[dict[str, Any] | None] = []

    def handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Add the inner payloads using alpha from kwargs."""
        seen_kwargs.append(kwargs)
        left, right = args[:2]
        alpha = (kwargs or {}).get("alpha", 1)
        return torch.add(left.payload, right.payload, alpha=alpha)

    sub.implements(torch.ops.aten.add.Tensor)(handler)
    left = sub(torch.tensor([1.0, 2.0]))
    right = sub(torch.tensor([3.0, 4.0]))
    result = torch.ops.aten.add.Tensor(left, right, alpha=2)
    assert torch.equal(result, torch.tensor([7.0, 10.0]))
    assert seen_kwargs == [{"alpha": 2}]


def test_torch_dispatch_inherits_parent_handler() -> None:
    """Test that a grandchild uses an inherited handler and passes runtime cls."""
    parent = _new_wrapper_subclass("Parent")
    child = _new_wrapper_subclass("Child", parent)
    grandchild = _new_wrapper_subclass("GrandChild", child)
    handler = _RecordingHandler(1.0)
    op = torch.ops.aten.neg.default
    parent.implements(op)(handler)
    inner = torch.tensor([1.0])
    result = op(grandchild(inner))
    assert torch.equal(result, torch.full_like(inner, 1.0))
    assert handler.calls[0][0] is grandchild


def test_torch_dispatch_child_handler_overrides_parent() -> None:
    """Test that a child handler wins over a parent handler for the same key."""
    parent = _new_wrapper_subclass("OverrideParent")
    child = _new_wrapper_subclass("OverrideChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_handler = _RecordingHandler(2.0)
    op = torch.ops.aten.neg.default
    parent.implements(op)(parent_handler)
    child.implements(op)(child_handler)
    inner = torch.tensor([1.0])
    result = op(child(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert parent_handler.calls == []
    assert len(child_handler.calls) == 1


def test_torch_dispatch_child_wildcard_shadows_parent_exact() -> None:
    """Test that a child wildcard is consulted before a parent exact handler."""
    parent = _new_wrapper_subclass("WildParent")
    child = _new_wrapper_subclass("WildChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_handler = _RecordingHandler(2.0)
    op = torch.ops.aten.neg.default
    parent.implements(op)(parent_handler)
    child.implements()(child_handler)
    inner = torch.tensor([1.0])
    result = op(child(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert parent_handler.calls == []
    assert len(child_handler.calls) == 1


def test_torch_dispatch_parent_exact_after_child_miss() -> None:
    """Test that a parent exact handler runs after a child table miss."""
    parent = _new_wrapper_subclass("ExactParent")
    child = _new_wrapper_subclass("ExactChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_handler = _RecordingHandler(2.0)
    neg = torch.ops.aten.neg.default
    absolute = torch.ops.aten.abs.default
    parent.implements(neg)(parent_handler)
    child.implements(absolute)(child_handler)
    inner = torch.tensor([1.0])
    result = neg(child(inner))
    assert torch.equal(result, torch.full_like(inner, 1.0))
    assert len(parent_handler.calls) == 1
    assert parent_handler.calls[0][0] is child
    assert child_handler.calls == []


def test_torch_dispatch_parent_wildcard_after_child_miss() -> None:
    """Test that a parent wildcard runs after a child table miss."""
    parent = _new_wrapper_subclass("WildcardParent")
    child = _new_wrapper_subclass("WildcardChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_handler = _RecordingHandler(2.0)
    neg = torch.ops.aten.neg.default
    absolute = torch.ops.aten.abs.default
    parent.implements()(parent_handler)
    child.implements(absolute)(child_handler)
    inner = torch.tensor([1.0])
    result = neg(child(inner))
    assert torch.equal(result, torch.full_like(inner, 1.0))
    assert len(parent_handler.calls) == 1
    assert parent_handler.calls[0][0] is child
    assert child_handler.calls == []


def test_torch_dispatch_same_key_super_composition() -> None:
    """Test that a child handler can continue the MRO for the same key."""
    parent = _new_wrapper_subclass("ComposeParent")
    child = _new_wrapper_subclass("ComposeChild", parent)
    order: list[str] = []
    seen_cls: list[type[MokitWrapperTensorBase]] = []

    def parent_handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Negate the payload."""
        order.append("parent")
        seen_cls.append(cls)
        return torch.neg(args[0].payload)

    def child_handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Delegate to the parent, then double the result."""
        order.append("child")
        result = super(child, cls).__torch_dispatch__(func, types, args, kwargs)  # pyrefly: ignore[invalid-argument]
        return result * 2

    op = torch.ops.aten.neg.default
    parent.implements(op)(parent_handler)
    child.implements(op)(child_handler)
    inner = torch.tensor([1.0, 2.0])
    result = op(child(inner))
    assert torch.equal(result, torch.tensor([-2.0, -4.0]))
    assert order == ["child", "parent"]
    assert seen_cls == [child]


def test_torch_dispatch_unhandled_operator_raises() -> None:
    """Test that the root dispatcher raises for an unhandled operator."""
    sub = _new_wrapper_subclass("Unhandled")
    wrapper = sub(torch.tensor([1.0]))
    with pytest.raises(NotImplementedError, match="Unhandled dispatch"):
        torch.ops.aten.neg.default(wrapper)


def test_torch_dispatch_root_table_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a handler registered on the root table is reached last."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_ATEN_OP_TABLE", {})
    handler = _RecordingHandler(6.0)
    op = torch.ops.aten.neg.default
    MokitWrapperTensorBase.implements(op)(handler)
    sub = _new_wrapper_subclass("RootTable")
    inner = torch.tensor([1.0])
    result = op(sub(inner))
    assert torch.equal(result, torch.full_like(inner, 6.0))
    assert handler.calls[0][0] is sub


def test_torch_dispatch_handler_receives_none_kwargs() -> None:
    """Test that the generated dispatcher forwards kwargs=None unchanged."""
    sub = _new_wrapper_subclass("NoneKwargs")
    handler = _RecordingHandler(4.0)
    op = torch.ops.aten.neg.default
    sub.implements(op)(handler)
    inner = torch.tensor([1.0])
    result = sub.__torch_dispatch__(op, (sub,), (sub(inner),), None)
    assert torch.equal(result, torch.full_like(inner, 4.0))
    assert handler.calls[0][4] is None


def test_torch_dispatch_root_preserves_none_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the root dispatcher preserves kwargs=None in the error message."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_ATEN_OP_TABLE", {})
    sub = _new_wrapper_subclass("RootNoneKwargs")
    wrapper = sub(torch.tensor([1.0]))
    with pytest.raises(NotImplementedError, match=r"kwarg_types=None"):
        sub.__torch_dispatch__(torch.ops.aten.neg.default, (sub,), (wrapper,), None)


def test_torch_dispatch_root_wildcard_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a root wildcard handler is used as the final lookup."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_ATEN_OP_TABLE", {})
    handler = _RecordingHandler(7.0)
    op = torch.ops.aten.neg.default
    MokitWrapperTensorBase.implements()(handler)
    sub = _new_wrapper_subclass("RootWildcard")
    inner = torch.tensor([1.0])
    result = op(sub(inner))
    assert torch.equal(result, torch.full_like(inner, 7.0))
    assert handler.calls[0][0] is sub
    assert handler.calls[0][1] is op


def test_torch_dispatch_parent_handler_wins_over_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a parent exact handler wins over a root exact handler."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_ATEN_OP_TABLE", {})
    root_handler = _RecordingHandler(1.0)
    op = torch.ops.aten.neg.default
    MokitWrapperTensorBase.implements(op)(root_handler)
    parent = _new_wrapper_subclass("RootVsParent")
    child = _new_wrapper_subclass("RootVsParentChild", parent)
    parent_handler = _RecordingHandler(2.0)
    parent.implements(op)(parent_handler)
    inner = torch.tensor([1.0])
    result = op(child(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert root_handler.calls == []
    assert parent_handler.calls[0][0] is child


def test_torch_dispatch_parent_wildcard_wins_over_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a parent wildcard wins over a root exact handler."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_ATEN_OP_TABLE", {})
    root_handler = _RecordingHandler(1.0)
    op = torch.ops.aten.neg.default
    MokitWrapperTensorBase.implements(op)(root_handler)
    parent = _new_wrapper_subclass("RootVsWildParent")
    child = _new_wrapper_subclass("RootVsWildChild", parent)
    parent_handler = _RecordingHandler(2.0)
    parent.implements()(parent_handler)
    inner = torch.tensor([1.0])
    result = op(child(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert root_handler.calls == []
    assert parent_handler.calls[0][0] is child


def test_torch_dispatch_root_exact_precedes_root_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the root table checks its exact key before its wildcard."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_ATEN_OP_TABLE", {})
    exact = _RecordingHandler(1.0)
    wildcard = _RecordingHandler(2.0)
    op = torch.ops.aten.neg.default
    MokitWrapperTensorBase.implements(op)(exact)
    MokitWrapperTensorBase.implements()(wildcard)
    sub = _new_wrapper_subclass("RootPrecedence")
    inner = torch.tensor([1.0])
    result = op(sub(inner))
    assert torch.equal(result, torch.full_like(inner, 1.0))
    assert len(exact.calls) == 1
    assert wildcard.calls == []


def test_torch_dispatch_unhandled_error_reports_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the unhandled error reports argument and keyword types."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_ATEN_OP_TABLE", {})
    sub = _new_wrapper_subclass("ErrorDetails")
    wrapper = sub(torch.tensor([1.0]))
    with pytest.raises(NotImplementedError) as exc_info:
        sub.__torch_dispatch__(torch.ops.aten.neg.default, (sub,), (wrapper,), {"alpha": 2})
    message = str(exc_info.value)
    assert "ErrorDetails dispatch" in message
    assert "func=" in message
    assert "types=" in message
    assert "arg_types=" in message
    assert "kwarg_types=" in message
    assert "alpha" in message
    assert "<class 'int'>" in message


@pytest.mark.parametrize(
    "result",
    [pytest.param(NotImplemented, id="not-implemented"), pytest.param(None, id="none")],
)
def test_torch_dispatch_handler_result_is_returned(result: Any) -> None:
    """Test that NotImplemented and None handler results are returned unchanged."""
    sub = _new_wrapper_subclass("HandlerResult")
    op = torch.ops.aten.neg.default

    def handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return the parametrized result."""
        return result

    sub.implements(op)(handler)
    wrapper = sub(torch.tensor([1.0]))
    assert sub.__torch_dispatch__(op, (sub,), (wrapper,), None) is result


def test_torch_dispatch_handler_exception_propagates() -> None:
    """Test that a handler exception propagates out of the dispatcher."""
    sub = _new_wrapper_subclass("HandlerError")
    op = torch.ops.aten.neg.default

    def handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Raise an error."""
        raise ValueError("handler failure")

    sub.implements(op)(handler)
    wrapper = sub(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="handler failure"):
        torch.ops.aten.neg.default(wrapper)


def test_torch_dispatch_handler_not_implemented_raises_outer_error() -> None:
    """Test that PyTorch rejects a dispatcher that returns NotImplemented."""
    sub = _new_wrapper_subclass("NotImplementedOuter")
    op = torch.ops.aten.neg.default

    def handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return NotImplemented."""
        return NotImplemented

    sub.implements(op)(handler)
    wrapper = sub(torch.tensor([1.0]))
    with pytest.raises(TypeError, match="Multiple dispatch failed"):
        torch.ops.aten.neg.default(wrapper)


def test_torch_dispatch_not_implemented_skips_parent_handler() -> None:
    """Test that a child handler returning NotImplemented does not reach the parent handler."""
    parent = _new_wrapper_subclass("NotImplementedParent")
    child = _new_wrapper_subclass("NotImplementedChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_calls: list[type[MokitWrapperTensorBase]] = []
    op = torch.ops.aten.neg.default
    parent.implements(op)(parent_handler)

    def child_handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return NotImplemented without delegating to the parent."""
        child_calls.append(cls)
        return NotImplemented

    child.implements(op)(child_handler)
    wrapper = child(torch.tensor([1.0]))
    with pytest.raises(TypeError, match="Multiple dispatch failed"):
        op(wrapper)
    assert child_calls == [child]
    assert parent_handler.calls == []


def test_torch_dispatch_deep_super_composition() -> None:
    """Test three-level same-key composition with a leaf runtime class."""
    top = _new_wrapper_subclass("DeepTop")
    middle = _new_wrapper_subclass("DeepMiddle", top)
    bottom = _new_wrapper_subclass("DeepBottom", middle)
    leaf = _new_wrapper_subclass("DeepLeaf", bottom)
    order: list[str] = []
    seen_cls: list[type[MokitWrapperTensorBase]] = []
    seen_types: list[tuple[type[Any], ...]] = []

    def top_handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Negate the payload."""
        order.append("top")
        seen_cls.append(cls)
        seen_types.append(types)
        return torch.neg(args[0].payload)

    def middle_handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Delegate to the top handler and add ten."""
        order.append("middle")
        seen_cls.append(cls)
        seen_types.append(types)
        parent = super(middle, cls)  # pyrefly: ignore[invalid-argument]
        return parent.__torch_dispatch__(func, types, args, kwargs) + 10

    def bottom_handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Delegate to the middle handler and double the result."""
        order.append("bottom")
        seen_cls.append(cls)
        seen_types.append(types)
        parent = super(bottom, cls)  # pyrefly: ignore[invalid-argument]
        return parent.__torch_dispatch__(func, types, args, kwargs) * 2

    op = torch.ops.aten.neg.default
    top.implements(op)(top_handler)
    middle.implements(op)(middle_handler)
    bottom.implements(op)(bottom_handler)
    inner = torch.tensor([1.0, 2.0])
    result = op(leaf(inner))
    assert torch.equal(result, torch.tensor([18.0, 16.0]))
    assert order == ["bottom", "middle", "top"]
    assert seen_cls == [leaf, leaf, leaf]
    assert seen_types == [(leaf,), (leaf,), (leaf,)]


def test_torch_dispatch_child_wildcard_shadows_parent_wildcard() -> None:
    """Test that a child wildcard wins over a parent wildcard."""
    parent = _new_wrapper_subclass("BothWildParent")
    child = _new_wrapper_subclass("BothWildChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_handler = _RecordingHandler(2.0)
    parent.implements()(parent_handler)
    child.implements()(child_handler)
    inner = torch.tensor([1.0])
    result = torch.ops.aten.neg.default(child(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert parent_handler.calls == []
    assert len(child_handler.calls) == 1


def test_torch_dispatch_super_relaunch_forwards_kwargs() -> None:
    """Test that a child handler re-launches the MRO lookup and forwards kwargs."""
    parent = _new_wrapper_subclass("KwargsComposeParent")
    child = _new_wrapper_subclass("KwargsComposeChild", parent)
    seen: list[dict[str, Any] | None] = []

    def parent_handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Add the payloads using alpha from kwargs."""
        seen.append(kwargs)
        left, right = args[:2]
        alpha = (kwargs or {}).get("alpha", 1)
        return torch.add(left.payload, right.payload, alpha=alpha)

    def child_handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Delegate to the parent handler."""
        # Re-enter the generated dispatcher after Child in the runtime MRO.
        parent_dispatch = super(child, cls)  # pyrefly: ignore[invalid-argument]
        return parent_dispatch.__torch_dispatch__(func, types, args, kwargs)

    op = torch.ops.aten.add.Tensor
    parent.implements(op)(parent_handler)
    child.implements(op)(child_handler)
    left = child(torch.tensor([1.0, 2.0]))
    right = child(torch.tensor([3.0, 4.0]))
    result = op(left, right, alpha=2)
    assert torch.equal(result, torch.tensor([7.0, 10.0]))
    assert seen == [{"alpha": 2}]


def test_torch_dispatch_super_reaches_custom_parent() -> None:
    """Test that super composition can reach an inherited custom dispatcher."""
    calls: list[str] = []

    class CustomParent(_DispatchWrapper):
        """A wrapper with a custom dispatch method."""

        @classmethod
        def __torch_dispatch__(
            cls,
            func: OpOverload,
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> torch.Tensor:
            """Negate the payload."""
            calls.append("parent")
            return torch.neg(args[0].payload)

    class CustomChild(CustomParent):
        """A child that registers its own handler."""

    def child_handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Delegate to the custom parent and double the result."""
        parent = super(CustomChild, cls)  # pyrefly: ignore[invalid-super-call]
        return parent.__torch_dispatch__(func, types, args, kwargs) * 2

    op = torch.ops.aten.neg.default
    CustomChild.implements(op)(child_handler)
    inner = torch.tensor([1.0, 2.0])
    result = op(CustomChild(inner))
    assert torch.equal(result, torch.tensor([-2.0, -4.0]))
    assert calls == ["parent"]


def test_torch_dispatch_parent_handler_registered_after_child() -> None:
    """Test that a parent handler added later is visible to child dispatch."""
    parent = _new_wrapper_subclass("LateParent")
    child = _new_wrapper_subclass("LateChild", parent)
    child_handler = _RecordingHandler(1.0)
    parent_handler = _RecordingHandler(2.0)
    neg = torch.ops.aten.neg.default
    absolute = torch.ops.aten.abs.default
    child.implements(neg)(child_handler)
    parent.implements(absolute)(parent_handler)
    inner = torch.tensor([1.0])
    result = absolute(child(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert parent_handler.calls[0][0] is child
    assert child_handler.calls == []


def test_torch_dispatch_super_reaches_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that super composition can reach the root table."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_ATEN_OP_TABLE", {})
    root_handler = _RecordingHandler(1.0)
    op = torch.ops.aten.neg.default
    MokitWrapperTensorBase.implements(op)(root_handler)
    sub = _new_wrapper_subclass("SuperToRoot")

    def handler(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Delegate to the next dispatcher in the MRO."""
        parent = super(sub, cls)  # pyrefly: ignore[invalid-argument]
        return parent.__torch_dispatch__(func, types, args, kwargs)

    sub.implements(op)(handler)
    inner = torch.tensor([1.0])
    result = op(sub(inner))
    assert torch.equal(result, torch.full_like(inner, 1.0))
    assert root_handler.calls[0][0] is sub


def test_torch_dispatch_multiple_inheritance_uses_mro_order() -> None:
    """Test dispatch order across two independent base classes."""
    left = _new_wrapper_subclass("MroLeft")
    right = _new_wrapper_subclass("MroRight")
    combined = cast(type[MokitWrapperTensorBase], type("MroCombined", (left, right), {}))
    left_handler = _RecordingHandler(1.0)
    right_handler = _RecordingHandler(2.0)
    neg = torch.ops.aten.neg.default
    absolute = torch.ops.aten.abs.default
    left.implements(neg)(left_handler)
    right.implements(absolute)(right_handler)
    inner = torch.tensor([-1.0])
    wrapper = combined(inner)
    assert torch.equal(neg(wrapper), torch.full_like(inner, 1.0))
    assert left_handler.calls[0][0] is combined
    assert right_handler.calls == []
    assert torch.equal(absolute(wrapper), torch.full_like(inner, 2.0))
    assert right_handler.calls[0][0] is combined
    assert len(left_handler.calls) == 1


def test_torch_dispatch_multiple_inheritance_wildcard_shadows_exact() -> None:
    """Test that an earlier base wildcard shadows a later base exact handler."""
    left = _new_wrapper_subclass("MroWildLeft")
    right = _new_wrapper_subclass("MroWildRight")
    combined = cast(type[MokitWrapperTensorBase], type("MroWildCombined", (left, right), {}))
    left_handler = _RecordingHandler(1.0)
    right_handler = _RecordingHandler(2.0)
    op = torch.ops.aten.neg.default
    left.implements()(left_handler)
    right.implements(op)(right_handler)
    inner = torch.tensor([1.0])
    result = op(combined(inner))
    assert torch.equal(result, torch.full_like(inner, 1.0))
    assert left_handler.calls[0][0] is combined
    assert right_handler.calls == []


def test_torch_dispatch_multiple_inheritance_falls_back_to_second_base_wildcard() -> None:
    """Test that dispatch reaches a wildcard on the second base after the first misses."""
    left = _new_wrapper_subclass("MroExactLeft")
    right = _new_wrapper_subclass("MroWildcardRight")
    combined = cast(type[MokitWrapperTensorBase], type("MroExactWildCombined", (left, right), {}))
    left_handler = _RecordingHandler(1.0)
    right_handler = _RecordingHandler(2.0)
    neg = torch.ops.aten.neg.default
    absolute = torch.ops.aten.abs.default
    left.implements(neg)(left_handler)
    right.implements()(right_handler)
    inner = torch.tensor([-1.0])
    result = absolute(combined(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert left_handler.calls == []
    assert right_handler.calls[0][0] is combined


def test_torch_dispatch_forwards_mixed_types() -> None:
    """Test that dispatch forwards types from multiple wrapper classes."""
    left = _new_wrapper_subclass("MixedLeft")
    right = _new_wrapper_subclass("MixedRight")
    handler = _RecordingHandler(1.0)
    op = torch.ops.aten.add.Tensor
    left.implements(op)(handler)
    inner_left = torch.tensor([1.0])
    inner_right = torch.tensor([2.0])
    wrapper_left = left(inner_left)
    wrapper_right = right(inner_right)
    result = left.__torch_dispatch__(op, (left, right), (wrapper_left, wrapper_right), {})
    assert torch.equal(result, torch.full_like(inner_left, 1.0))
    assert handler.calls[0][2] == (left, right)


def test_torch_dispatch_forwards_related_types() -> None:
    """Test that normal dispatch forwards types from related wrapper classes."""
    parent = _new_wrapper_subclass("RelatedParent")
    child = _new_wrapper_subclass("RelatedChild", parent)
    handler = _RecordingHandler(1.0)
    op = torch.ops.aten.add.Tensor
    parent.implements(op)(handler)
    inner_left = torch.tensor([1.0])
    inner_right = torch.tensor([2.0])
    result = op(child(inner_left), parent(inner_right))
    assert torch.equal(result, torch.full_like(inner_left, 1.0))
    assert handler.calls[0][2] == (child, parent)


# ------------------------------------------------------------------------------
# __torch_function__
# ------------------------------------------------------------------------------


def test_torch_function_exact_handler_forwards_arguments() -> None:
    """Test that an exact handler receives cls, func, types, args, and kwargs."""
    sub = _new_wrapper_subclass("TorchFnExact")
    handler = _RecordingHandler(2.0)
    sub.implements_torch_function(torch.neg)(handler)
    inner = torch.tensor([1.0, 2.0])
    wrapper = sub(inner)
    result = torch.neg(wrapper)
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert len(handler.calls) == 1
    cls, func, types, args, kwargs = handler.calls[0]
    assert cls is sub
    assert func is torch.neg
    assert types == (sub,)
    assert len(args) == 1 and args[0] is wrapper
    assert kwargs is None


def test_torch_function_wildcard_handler() -> None:
    """Test that a wildcard handler receives torch functions with no exact handler."""
    sub = _new_wrapper_subclass("TorchFnWildcardDispatch")
    handler = _RecordingHandler(3.0)
    sub.implements_torch_function()(handler)
    inner = torch.tensor([1.0, 2.0])
    wrapper = sub(inner)
    result = torch.neg(wrapper)
    assert torch.equal(result, torch.full_like(inner, 3.0))
    cls, func, types, args, kwargs = handler.calls[0]
    assert cls is sub
    assert func is torch.neg
    assert types == (sub,)
    assert len(args) == 1 and args[0] is wrapper
    assert kwargs is None


def test_torch_function_exact_precedes_wildcard() -> None:
    """Test that an exact handler wins over a wildcard on the same class."""
    sub = _new_wrapper_subclass("TorchFnExactPrecedence")
    exact = _RecordingHandler(1.0)
    wildcard = _RecordingHandler(2.0)
    sub.implements_torch_function()(wildcard)
    sub.implements_torch_function(torch.neg)(exact)
    inner = torch.tensor([1.0])
    wrapper = sub(inner)
    assert torch.equal(torch.neg(wrapper), torch.full_like(inner, 1.0))
    assert len(exact.calls) == 1
    assert wildcard.calls == []
    assert torch.equal(torch.abs(wrapper), torch.full_like(inner, 2.0))
    assert len(wildcard.calls) == 1


def test_torch_function_forwards_kwargs() -> None:
    """Test that keyword arguments reach the handler."""
    sub = _new_wrapper_subclass("TorchFnKwargs")
    seen_kwargs: list[dict[str, Any] | None] = []

    def handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Add the inner payloads using alpha from kwargs."""
        seen_kwargs.append(kwargs)
        left, right = args[:2]
        alpha = (kwargs or {}).get("alpha", 1)
        return torch.add(left.payload, right.payload, alpha=alpha)

    sub.implements_torch_function(torch.add)(handler)
    left = sub(torch.tensor([1.0, 2.0]))
    right = sub(torch.tensor([3.0, 4.0]))
    result = torch.add(left, right, alpha=2)
    assert torch.equal(result, torch.tensor([7.0, 10.0]))
    assert seen_kwargs == [{"alpha": 2}]


def test_torch_function_function_key_does_not_handle_method_call() -> None:
    """Test that a functional key does not match the method-call descriptor."""
    sub = _new_wrapper_subclass("TorchFnFunctionalMethodMiss")
    handler = _RecordingHandler(1.0)
    sub.implements_torch_function(torch.neg)(handler)
    wrapper = sub(torch.tensor([1.0]))
    with pytest.raises(NotImplementedError, match="dispatch"):
        wrapper.neg()
    assert handler.calls == []


def test_torch_function_method_key_handles_method_call() -> None:
    """Test that a method descriptor key handles the corresponding method call."""
    sub = _new_wrapper_subclass("TorchFnMethodKey")
    handler = _RecordingHandler(2.0)
    sub.implements_torch_function(torch.Tensor.neg)(handler)
    inner = torch.tensor([1.0])
    wrapper = sub(inner)
    result = wrapper.neg()
    assert torch.equal(result, torch.full_like(inner, 2.0))
    cls, func, types, args, kwargs = handler.calls[0]
    assert cls is sub
    assert func is torch.Tensor.neg
    assert types == (sub,)
    assert len(args) == 1 and args[0] is wrapper
    assert kwargs is None


def test_torch_function_function_key_does_not_handle_operator_call() -> None:
    """Test that a functional key does not match the operator-call descriptor."""
    sub = _new_wrapper_subclass("TorchFnFunctionalOperatorMiss")
    handler = _RecordingHandler(1.0)
    sub.implements_torch_function(torch.add)(handler)
    left = sub(torch.tensor([1.0]))
    right = sub(torch.tensor([2.0]))
    with pytest.raises(NotImplementedError, match="dispatch"):
        _ = left + right
    assert handler.calls == []


def test_torch_function_method_key_handles_operator_call() -> None:
    """Test that the add method descriptor key handles the + operator."""
    sub = _new_wrapper_subclass("TorchFnOperatorKey")
    handler = _RecordingHandler(3.0)
    sub.implements_torch_function(torch.Tensor.add)(handler)
    inner_left = torch.tensor([1.0])
    inner_right = torch.tensor([2.0])
    left = sub(inner_left)
    right = sub(inner_right)
    result = left + right
    assert torch.equal(result, torch.full_like(inner_left, 3.0))
    cls, func, types, args, kwargs = handler.calls[0]
    assert cls is sub
    assert func is torch.Tensor.add
    assert types == (sub,)
    assert len(args) == 2 and args[0] is left and args[1] is right
    assert kwargs is None


def test_torch_function_inherits_parent_handler() -> None:
    """Test that a grandchild uses an inherited handler and passes runtime cls."""
    parent = _new_wrapper_subclass("TorchFnParent")
    child = _new_wrapper_subclass("TorchFnChild", parent)
    grandchild = _new_wrapper_subclass("TorchFnGrandChild", child)
    handler = _RecordingHandler(1.0)
    parent.implements_torch_function(torch.neg)(handler)
    inner = torch.tensor([1.0])
    result = torch.neg(grandchild(inner))
    assert torch.equal(result, torch.full_like(inner, 1.0))
    assert handler.calls[0][0] is grandchild


def test_torch_function_child_handler_overrides_parent() -> None:
    """Test that a child handler wins over a parent handler for the same key."""
    parent = _new_wrapper_subclass("TorchFnOverrideParent")
    child = _new_wrapper_subclass("TorchFnOverrideChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_handler = _RecordingHandler(2.0)
    parent.implements_torch_function(torch.neg)(parent_handler)
    child.implements_torch_function(torch.neg)(child_handler)
    inner = torch.tensor([1.0])
    result = torch.neg(child(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert parent_handler.calls == []
    assert len(child_handler.calls) == 1


def test_torch_function_child_wildcard_shadows_parent_exact() -> None:
    """Test that a child wildcard is consulted before a parent exact handler."""
    parent = _new_wrapper_subclass("TorchFnWildParent")
    child = _new_wrapper_subclass("TorchFnWildChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_handler = _RecordingHandler(2.0)
    parent.implements_torch_function(torch.neg)(parent_handler)
    child.implements_torch_function()(child_handler)
    inner = torch.tensor([1.0])
    result = torch.neg(child(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert parent_handler.calls == []
    assert len(child_handler.calls) == 1


def test_torch_function_parent_exact_after_child_miss() -> None:
    """Test that a parent exact handler runs after a child table miss."""
    parent = _new_wrapper_subclass("TorchFnExactParent")
    child = _new_wrapper_subclass("TorchFnExactChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_handler = _RecordingHandler(2.0)
    parent.implements_torch_function(torch.neg)(parent_handler)
    child.implements_torch_function(torch.abs)(child_handler)
    inner = torch.tensor([1.0])
    result = torch.neg(child(inner))
    assert torch.equal(result, torch.full_like(inner, 1.0))
    assert len(parent_handler.calls) == 1
    assert parent_handler.calls[0][0] is child
    assert child_handler.calls == []


def test_torch_function_parent_wildcard_after_child_miss() -> None:
    """Test that a parent wildcard runs after a child table miss."""
    parent = _new_wrapper_subclass("TorchFnWildcardParent")
    child = _new_wrapper_subclass("TorchFnWildcardChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_handler = _RecordingHandler(2.0)
    parent.implements_torch_function()(parent_handler)
    child.implements_torch_function(torch.abs)(child_handler)
    inner = torch.tensor([1.0])
    result = torch.neg(child(inner))
    assert torch.equal(result, torch.full_like(inner, 1.0))
    assert len(parent_handler.calls) == 1
    assert parent_handler.calls[0][0] is child
    assert child_handler.calls == []


def test_torch_function_same_key_super_composition() -> None:
    """Test that a child handler can continue the MRO for the same key."""
    parent = _new_wrapper_subclass("TorchFnComposeParent")
    child = _new_wrapper_subclass("TorchFnComposeChild", parent)
    order: list[str] = []
    seen_cls: list[type[MokitWrapperTensorBase]] = []

    def parent_handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Negate the payload."""
        order.append("parent")
        seen_cls.append(cls)
        return torch.neg(args[0].payload)

    def child_handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Delegate to the parent, then double the result."""
        order.append("child")
        parent_dispatch = super(child, cls)  # pyrefly: ignore[invalid-argument]
        result = parent_dispatch.__torch_function__(func, types, args, kwargs)
        return result * 2

    parent.implements_torch_function(torch.neg)(parent_handler)
    child.implements_torch_function(torch.neg)(child_handler)
    inner = torch.tensor([1.0, 2.0])
    result = torch.neg(child(inner))
    assert torch.equal(result, torch.tensor([-2.0, -4.0]))
    assert order == ["child", "parent"]
    assert seen_cls == [child]


def test_torch_function_fallback_raises_without_torch_dispatch_handler() -> None:
    """Test that the root fallback raises when no ATen handler is registered."""
    sub = _new_wrapper_subclass("TorchFnUnhandled")
    wrapper = sub(torch.tensor([1.0]))
    with pytest.raises(NotImplementedError, match="dispatch"):
        torch.neg(wrapper)


def test_torch_function_fallback_reaches_registered_torch_dispatch_handler() -> None:
    """Test that the root fallback reaches a registered ATen handler for an unhandled torch function."""
    sub = _new_wrapper_subclass("TorchFnWithDispatch")
    torch_function_handler = _RecordingHandler(2.0)
    dispatch_handler = _RecordingHandler(4.0)
    sub.implements_torch_function(torch.neg)(torch_function_handler)
    sub.implements(torch.ops.aten.abs.default)(dispatch_handler)
    inner = torch.tensor([-1.0, 2.0])
    wrapper = sub(inner)
    assert torch.equal(torch.neg(wrapper), torch.full_like(inner, 2.0))
    assert len(torch_function_handler.calls) == 1
    assert dispatch_handler.calls == []
    assert torch.equal(torch.abs(wrapper), torch.full_like(inner, 4.0))
    assert len(torch_function_handler.calls) == 1
    assert len(dispatch_handler.calls) == 1
    assert dispatch_handler.calls[0][0] is sub
    assert dispatch_handler.calls[0][1] is torch.ops.aten.abs.default


def test_torch_function_root_table_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a handler registered on the root table is reached last."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_TORCH_FN_TABLE", {})
    handler = _RecordingHandler(6.0)
    MokitWrapperTensorBase.implements_torch_function(torch.neg)(handler)
    sub = _new_wrapper_subclass("TorchFnRootTable")
    inner = torch.tensor([1.0])
    result = torch.neg(sub(inner))
    assert torch.equal(result, torch.full_like(inner, 6.0))
    assert handler.calls[0][0] is sub


def test_torch_function_handler_receives_none_kwargs() -> None:
    """Test that the generated dispatcher forwards kwargs=None unchanged."""
    sub = _new_wrapper_subclass("TorchFnNoneKwargs")
    handler = _RecordingHandler(4.0)
    sub.implements_torch_function(torch.neg)(handler)
    inner = torch.tensor([1.0])
    result = sub.__torch_function__(torch.neg, (sub,), (sub(inner),), None)
    assert torch.equal(result, torch.full_like(inner, 4.0))
    assert handler.calls[0][4] is None


def test_torch_function_root_fallback_accepts_none_kwargs() -> None:
    """Test that the root fallback accepts kwargs=None."""
    sub = _new_wrapper_subclass("TorchFnRootNoneKwargs")
    wrapper = sub(torch.tensor([1.0]))

    def func(tensor: torch.Tensor, *, scale: int = 1) -> int:
        """Return the scale."""
        return scale

    assert sub.__torch_function__(func, (sub,), (wrapper,), None) == 1


def test_torch_function_root_wildcard_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a root wildcard handler is used as the final lookup."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_TORCH_FN_TABLE", {})
    handler = _RecordingHandler(7.0)
    MokitWrapperTensorBase.implements_torch_function()(handler)
    sub = _new_wrapper_subclass("TorchFnRootWildcard")
    inner = torch.tensor([1.0])
    result = torch.neg(sub(inner))
    assert torch.equal(result, torch.full_like(inner, 7.0))
    assert handler.calls[0][0] is sub
    assert handler.calls[0][1] is torch.neg


def test_torch_function_parent_handler_wins_over_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a parent exact handler wins over a root exact handler."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_TORCH_FN_TABLE", {})
    root_handler = _RecordingHandler(1.0)
    MokitWrapperTensorBase.implements_torch_function(torch.neg)(root_handler)
    parent = _new_wrapper_subclass("TorchFnRootVsParent")
    child = _new_wrapper_subclass("TorchFnRootVsParentChild", parent)
    parent_handler = _RecordingHandler(2.0)
    parent.implements_torch_function(torch.neg)(parent_handler)
    inner = torch.tensor([1.0])
    result = torch.neg(child(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert root_handler.calls == []
    assert parent_handler.calls[0][0] is child


def test_torch_function_parent_wildcard_wins_over_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a parent wildcard wins over a root exact handler."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_TORCH_FN_TABLE", {})
    root_handler = _RecordingHandler(1.0)
    MokitWrapperTensorBase.implements_torch_function(torch.neg)(root_handler)
    parent = _new_wrapper_subclass("TorchFnRootVsWildParent")
    child = _new_wrapper_subclass("TorchFnRootVsWildChild", parent)
    parent_handler = _RecordingHandler(2.0)
    parent.implements_torch_function()(parent_handler)
    inner = torch.tensor([1.0])
    result = torch.neg(child(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert root_handler.calls == []
    assert parent_handler.calls[0][0] is child


def test_torch_function_root_exact_precedes_root_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the root table checks its exact key before its wildcard."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_TORCH_FN_TABLE", {})
    exact = _RecordingHandler(1.0)
    wildcard = _RecordingHandler(2.0)
    MokitWrapperTensorBase.implements_torch_function(torch.neg)(exact)
    MokitWrapperTensorBase.implements_torch_function()(wildcard)
    sub = _new_wrapper_subclass("TorchFnRootPrecedence")
    inner = torch.tensor([1.0])
    result = torch.neg(sub(inner))
    assert torch.equal(result, torch.full_like(inner, 1.0))
    assert len(exact.calls) == 1
    assert wildcard.calls == []


def test_torch_function_handshake_returns_not_implemented() -> None:
    """Test that the root returns NotImplemented for a non-participating class."""
    sub = _new_wrapper_subclass("TorchFnHandshakeSub")
    other = _new_wrapper_subclass("TorchFnHandshakeOther")
    wrapper = sub(torch.tensor([1.0]))

    def func(tensor: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged."""
        return tensor

    assert sub.__torch_function__(func, (other,), (wrapper,), None) is NotImplemented


def test_torch_function_handshake_allows_subclass_types() -> None:
    """Test that the root dispatches when cls subclasses every participating type."""
    parent = _new_wrapper_subclass("TorchFnHandshakeParent")
    child = _new_wrapper_subclass("TorchFnHandshakeChild", parent)
    wrapper = child(torch.tensor([1.0]))
    seen: list[torch.Tensor] = []

    def func(tensor: torch.Tensor) -> str:
        """Record the argument and return a sentinel."""
        seen.append(tensor)
        return "fallback"

    assert child.__torch_function__(func, (parent, child), (wrapper,), None) == "fallback"
    assert len(seen) == 1
    assert seen[0] is wrapper


def test_torch_function_handshake_requires_every_type() -> None:
    """Test that the root handshake fails when any participating type is unrelated."""
    parent = _new_wrapper_subclass("TorchFnHandshakeMixedParent")
    child = _new_wrapper_subclass("TorchFnHandshakeMixedChild", parent)
    other = _new_wrapper_subclass("TorchFnHandshakeMixedOther")
    wrapper = child(torch.tensor([1.0]))

    def func(tensor: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged."""
        return tensor

    assert child.__torch_function__(func, (parent, other), (wrapper,), None) is NotImplemented


def test_torch_function_handshake_not_implemented_falls_through_to_next_type() -> None:
    """Test that a handshake NotImplemented lets PyTorch try the next type's handler."""
    first = _new_wrapper_subclass("TorchFnHandshakeFirst")
    second = _new_wrapper_subclass("TorchFnHandshakeSecond")
    handler = _RecordingHandler(9.0)
    second.implements_torch_function(torch.add)(handler)
    inner_first = torch.tensor([1.0])
    inner_second = torch.tensor([2.0])
    result = torch.add(first(inner_first), second(inner_second))
    assert torch.equal(result, torch.full_like(inner_first, 9.0))
    assert handler.calls[0][0] is second
    assert handler.calls[0][2] == (first, second)


def test_torch_function_handler_precedes_handshake() -> None:
    """Test that a registered handler claims a call before the root handshake."""
    sub = _new_wrapper_subclass("TorchFnHandshakeHandler")
    other = _new_wrapper_subclass("TorchFnHandshakeUnrelated")
    handler = _RecordingHandler(4.0)
    sub.implements_torch_function(torch.neg)(handler)
    inner = torch.tensor([1.0])
    wrapper = sub(inner)
    result = sub.__torch_function__(torch.neg, (other,), (wrapper,), None)
    assert torch.equal(result, torch.full_like(inner, 4.0))
    assert handler.calls[0][0] is sub
    assert handler.calls[0][2] == (other,)


def test_torch_function_root_fallback_calls_function() -> None:
    """Test that the root forwards args and kwargs to the original function."""
    sub = _new_wrapper_subclass("TorchFnFallback")
    wrapper = sub(torch.tensor([1.0]))
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def func(*args: Any, **kwargs: Any) -> str:
        """Record the call and return a sentinel."""
        seen.append((args, kwargs))
        return "result"

    assert sub.__torch_function__(func, (sub,), (wrapper, 2), {"scale": 3}) == "result"
    assert len(seen) == 1
    assert seen[0][0][0] is wrapper
    assert seen[0][0][1] == 2
    assert seen[0][1] == {"scale": 3}


def test_torch_function_root_fallback_exception_propagates() -> None:
    """Test that an exception from the fallback function propagates."""
    sub = _new_wrapper_subclass("TorchFnFallbackError")
    wrapper = sub(torch.tensor([1.0]))

    def func(tensor: torch.Tensor) -> None:
        """Raise an error."""
        raise ValueError("fallback failure")

    with pytest.raises(ValueError, match="fallback failure"):
        sub.__torch_function__(func, (sub,), (wrapper,), None)


def test_torch_function_root_fallback_disables_subclass_dispatch() -> None:
    """Test that the root fallback disables nested torch function dispatch."""
    sub = _new_wrapper_subclass("TorchFnFallbackDisable")
    wrapper = sub(torch.tensor([1.0]))

    def func(tensor: torch.Tensor) -> torch.Tensor:
        """Call a torch function without registering a torch function handler."""
        return torch.neg(tensor)

    with pytest.raises(NotImplementedError, match="dispatch"):
        sub.__torch_function__(func, (sub,), (wrapper,), None)


@pytest.mark.parametrize(
    "result",
    [pytest.param(NotImplemented, id="not-implemented"), pytest.param(None, id="none")],
)
def test_torch_function_handler_result_is_returned(result: Any) -> None:
    """Test that NotImplemented and None handler results are returned unchanged."""
    sub = _new_wrapper_subclass("TorchFnHandlerResult")

    def handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return the parametrized result."""
        return result

    sub.implements_torch_function(torch.neg)(handler)
    wrapper = sub(torch.tensor([1.0]))
    assert sub.__torch_function__(torch.neg, (sub,), (wrapper,), None) is result


def test_torch_function_handler_exception_propagates() -> None:
    """Test that a handler exception propagates out of the dispatcher."""
    sub = _new_wrapper_subclass("TorchFnHandlerError")

    def handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Raise an error."""
        raise ValueError("handler failure")

    sub.implements_torch_function(torch.neg)(handler)
    wrapper = sub(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="handler failure"):
        torch.neg(wrapper)


def test_torch_function_handler_not_implemented_raises_outer_error() -> None:
    """Test that PyTorch rejects a dispatcher that returns NotImplemented."""
    sub = _new_wrapper_subclass("TorchFnNotImplementedOuter")

    def handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return NotImplemented."""
        return NotImplemented

    sub.implements_torch_function(torch.neg)(handler)
    wrapper = sub(torch.tensor([1.0]))
    with pytest.raises(TypeError, match="Multiple dispatch failed"):
        torch.neg(wrapper)


def test_torch_function_not_implemented_skips_parent_handler() -> None:
    """Test that a child handler returning NotImplemented does not reach the parent handler."""
    parent = _new_wrapper_subclass("TorchFnNotImplementedParent")
    child = _new_wrapper_subclass("TorchFnNotImplementedChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_calls: list[type[MokitWrapperTensorBase]] = []
    parent.implements_torch_function(torch.neg)(parent_handler)

    def child_handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return NotImplemented without delegating to the parent."""
        child_calls.append(cls)
        return NotImplemented

    child.implements_torch_function(torch.neg)(child_handler)
    wrapper = child(torch.tensor([1.0]))
    with pytest.raises(TypeError, match="Multiple dispatch failed"):
        torch.neg(wrapper)
    assert child_calls == [child]
    assert parent_handler.calls == []


def test_torch_function_not_implemented_does_not_fall_back_to_torch_dispatch() -> None:
    """Test that NotImplemented does not fall back to a registered ATen handler."""
    sub = _new_wrapper_subclass("TorchFnNotImplementedNoFallback")
    dispatch_handler = _RecordingHandler(2.0)
    torch_function_calls: list[Callable[..., Any]] = []
    op = torch.ops.aten.neg.default

    def torch_function_handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return NotImplemented."""
        torch_function_calls.append(func)
        return NotImplemented

    sub.implements_torch_function(torch.neg)(torch_function_handler)
    sub.implements(op)(dispatch_handler)
    wrapper = sub(torch.tensor([1.0]))
    with pytest.raises(TypeError, match="Multiple dispatch failed"):
        torch.neg(wrapper)
    assert torch_function_calls == [torch.neg]
    assert dispatch_handler.calls == []


def test_torch_function_not_implemented_falls_through_to_next_type() -> None:
    """Test that PyTorch tries the next participating type after NotImplemented."""
    first = _new_wrapper_subclass("TorchFnFallthroughFirst")
    second = _new_wrapper_subclass("TorchFnFallthroughSecond")
    first_calls: list[torch.Tensor] = []

    def first_handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Return NotImplemented so PyTorch tries the next type."""
        first_calls.append(args[0])
        return NotImplemented

    second_handler = _RecordingHandler(9.0)
    first.implements_torch_function(torch.add)(first_handler)
    second.implements_torch_function(torch.add)(second_handler)
    inner_first = torch.tensor([1.0])
    inner_second = torch.tensor([2.0])
    wrapper_first = first(inner_first)
    wrapper_second = second(inner_second)
    result = torch.add(wrapper_first, wrapper_second)
    assert torch.equal(result, torch.full_like(inner_first, 9.0))
    assert len(first_calls) == 1
    assert first_calls[0] is wrapper_first
    assert second_handler.calls[0][0] is second
    assert second_handler.calls[0][2] == (first, second)


def test_torch_function_deep_super_composition() -> None:
    """Test three-level same-key composition with a leaf runtime class."""
    top = _new_wrapper_subclass("TorchFnDeepTop")
    middle = _new_wrapper_subclass("TorchFnDeepMiddle", top)
    bottom = _new_wrapper_subclass("TorchFnDeepBottom", middle)
    leaf = _new_wrapper_subclass("TorchFnDeepLeaf", bottom)
    order: list[str] = []
    seen_cls: list[type[MokitWrapperTensorBase]] = []
    seen_types: list[tuple[type[Any], ...]] = []

    def top_handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Negate the payload."""
        order.append("top")
        seen_cls.append(cls)
        seen_types.append(types)
        return torch.neg(args[0].payload)

    def middle_handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Delegate to the top handler and add ten."""
        order.append("middle")
        seen_cls.append(cls)
        seen_types.append(types)
        parent = super(middle, cls)  # pyrefly: ignore[invalid-argument]
        return parent.__torch_function__(func, types, args, kwargs) + 10

    def bottom_handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Delegate to the middle handler and double the result."""
        order.append("bottom")
        seen_cls.append(cls)
        seen_types.append(types)
        parent = super(bottom, cls)  # pyrefly: ignore[invalid-argument]
        return parent.__torch_function__(func, types, args, kwargs) * 2

    top.implements_torch_function(torch.neg)(top_handler)
    middle.implements_torch_function(torch.neg)(middle_handler)
    bottom.implements_torch_function(torch.neg)(bottom_handler)
    inner = torch.tensor([1.0, 2.0])
    result = torch.neg(leaf(inner))
    assert torch.equal(result, torch.tensor([18.0, 16.0]))
    assert order == ["bottom", "middle", "top"]
    assert seen_cls == [leaf, leaf, leaf]
    assert seen_types == [(leaf,), (leaf,), (leaf,)]


def test_torch_function_child_wildcard_shadows_parent_wildcard() -> None:
    """Test that a child wildcard wins over a parent wildcard."""
    parent = _new_wrapper_subclass("TorchFnBothWildParent")
    child = _new_wrapper_subclass("TorchFnBothWildChild", parent)
    parent_handler = _RecordingHandler(1.0)
    child_handler = _RecordingHandler(2.0)
    parent.implements_torch_function()(parent_handler)
    child.implements_torch_function()(child_handler)
    inner = torch.tensor([1.0])
    result = torch.neg(child(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert parent_handler.calls == []
    assert len(child_handler.calls) == 1


def test_torch_function_super_relaunch_forwards_kwargs() -> None:
    """Test that a child handler re-launches the MRO lookup and forwards kwargs."""
    parent = _new_wrapper_subclass("TorchFnKwargsComposeParent")
    child = _new_wrapper_subclass("TorchFnKwargsComposeChild", parent)
    seen: list[dict[str, Any] | None] = []

    def parent_handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Add the payloads using alpha from kwargs."""
        seen.append(kwargs)
        left, right = args[:2]
        alpha = (kwargs or {}).get("alpha", 1)
        return torch.add(left.payload, right.payload, alpha=alpha)

    def child_handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Delegate to the parent handler."""
        # Re-enter the generated dispatcher after Child in the runtime MRO.
        parent_dispatch = super(child, cls)  # pyrefly: ignore[invalid-argument]
        return parent_dispatch.__torch_function__(func, types, args, kwargs)

    parent.implements_torch_function(torch.add)(parent_handler)
    child.implements_torch_function(torch.add)(child_handler)
    left = child(torch.tensor([1.0, 2.0]))
    right = child(torch.tensor([3.0, 4.0]))
    result = torch.add(left, right, alpha=2)
    assert torch.equal(result, torch.tensor([7.0, 10.0]))
    assert seen == [{"alpha": 2}]


def test_torch_function_super_reaches_custom_parent() -> None:
    """Test that super composition can reach an inherited custom dispatcher."""
    calls: list[str] = []

    class CustomParent(_DispatchWrapper):
        """A wrapper with a custom torch function method."""

        @classmethod
        def __torch_function__(
            cls,
            func: Callable[..., Any],
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> torch.Tensor:
            """Negate the payload."""
            calls.append("parent")
            return torch.neg(args[0].payload)

    class CustomChild(CustomParent):
        """A child that registers its own handler."""

    def child_handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Delegate to the custom parent and double the result."""
        parent = super(CustomChild, cls)  # pyrefly: ignore[invalid-super-call]
        return parent.__torch_function__(func, types, args, kwargs) * 2

    CustomChild.implements_torch_function(torch.neg)(child_handler)
    inner = torch.tensor([1.0, 2.0])
    result = torch.neg(CustomChild(inner))
    assert torch.equal(result, torch.tensor([-2.0, -4.0]))
    assert calls == ["parent"]


def test_torch_function_parent_handler_registered_after_child() -> None:
    """Test that a parent handler added later is visible to child dispatch."""
    parent = _new_wrapper_subclass("TorchFnLateParent")
    child = _new_wrapper_subclass("TorchFnLateChild", parent)
    child_handler = _RecordingHandler(1.0)
    parent_handler = _RecordingHandler(2.0)
    child.implements_torch_function(torch.neg)(child_handler)
    parent.implements_torch_function(torch.abs)(parent_handler)
    inner = torch.tensor([1.0])
    result = torch.abs(child(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert parent_handler.calls[0][0] is child
    assert child_handler.calls == []


def test_torch_function_super_reaches_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that super composition can reach the root table."""
    monkeypatch.setattr(MokitWrapperTensorBase, "_TORCH_FN_TABLE", {})
    root_handler = _RecordingHandler(1.0)
    MokitWrapperTensorBase.implements_torch_function(torch.neg)(root_handler)
    sub = _new_wrapper_subclass("TorchFnSuperToRoot")

    def handler(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Delegate to the next dispatcher in the MRO."""
        parent = super(sub, cls)  # pyrefly: ignore[invalid-argument]
        return parent.__torch_function__(func, types, args, kwargs)

    sub.implements_torch_function(torch.neg)(handler)
    inner = torch.tensor([1.0])
    result = torch.neg(sub(inner))
    assert torch.equal(result, torch.full_like(inner, 1.0))
    assert root_handler.calls[0][0] is sub


def test_torch_function_multiple_inheritance_uses_mro_order() -> None:
    """Test dispatch order across two independent base classes."""
    left = _new_wrapper_subclass("TorchFnMroLeft")
    right = _new_wrapper_subclass("TorchFnMroRight")
    combined = cast(type[MokitWrapperTensorBase], type("TorchFnMroCombined", (left, right), {}))
    left_handler = _RecordingHandler(1.0)
    right_handler = _RecordingHandler(2.0)
    left.implements_torch_function(torch.neg)(left_handler)
    right.implements_torch_function(torch.abs)(right_handler)
    inner = torch.tensor([-1.0])
    wrapper = combined(inner)
    assert torch.equal(torch.neg(wrapper), torch.full_like(inner, 1.0))
    assert left_handler.calls[0][0] is combined
    assert right_handler.calls == []
    assert torch.equal(torch.abs(wrapper), torch.full_like(inner, 2.0))
    assert right_handler.calls[0][0] is combined
    assert len(left_handler.calls) == 1


def test_torch_function_multiple_inheritance_wildcard_shadows_exact() -> None:
    """Test that an earlier base wildcard shadows a later base exact handler."""
    left = _new_wrapper_subclass("TorchFnMroWildLeft")
    right = _new_wrapper_subclass("TorchFnMroWildRight")
    combined = cast(type[MokitWrapperTensorBase], type("TorchFnMroWildCombined", (left, right), {}))
    left_handler = _RecordingHandler(1.0)
    right_handler = _RecordingHandler(2.0)
    left.implements_torch_function()(left_handler)
    right.implements_torch_function(torch.neg)(right_handler)
    inner = torch.tensor([1.0])
    result = torch.neg(combined(inner))
    assert torch.equal(result, torch.full_like(inner, 1.0))
    assert left_handler.calls[0][0] is combined
    assert right_handler.calls == []


def test_torch_function_multiple_inheritance_falls_back_to_second_base_wildcard() -> None:
    """Test that dispatch reaches a wildcard on the second base after the first misses."""
    left = _new_wrapper_subclass("TorchFnMroExactLeft")
    right = _new_wrapper_subclass("TorchFnMroWildcardRight")
    combined = cast(type[MokitWrapperTensorBase], type("TorchFnMroExactWildCombined", (left, right), {}))
    left_handler = _RecordingHandler(1.0)
    right_handler = _RecordingHandler(2.0)
    left.implements_torch_function(torch.neg)(left_handler)
    right.implements_torch_function()(right_handler)
    inner = torch.tensor([-1.0])
    result = torch.abs(combined(inner))
    assert torch.equal(result, torch.full_like(inner, 2.0))
    assert left_handler.calls == []
    assert right_handler.calls[0][0] is combined


def test_torch_function_forwards_mixed_types() -> None:
    """Test that dispatch forwards types from multiple wrapper classes."""
    left = _new_wrapper_subclass("TorchFnMixedLeft")
    right = _new_wrapper_subclass("TorchFnMixedRight")
    handler = _RecordingHandler(1.0)
    left.implements_torch_function(torch.add)(handler)
    inner_left = torch.tensor([1.0])
    inner_right = torch.tensor([2.0])
    wrapper_left = left(inner_left)
    wrapper_right = right(inner_right)
    result = left.__torch_function__(torch.add, (left, right), (wrapper_left, wrapper_right), {})
    assert torch.equal(result, torch.full_like(inner_left, 1.0))
    assert handler.calls[0][2] == (left, right)
    assert handler.calls[0][4] == {}


def test_torch_function_forwards_related_types() -> None:
    """Test that normal dispatch forwards types from related wrapper classes."""
    parent = _new_wrapper_subclass("TorchFnRelatedParent")
    child = _new_wrapper_subclass("TorchFnRelatedChild", parent)
    handler = _RecordingHandler(1.0)
    parent.implements_torch_function(torch.add)(handler)
    inner_left = torch.tensor([1.0])
    inner_right = torch.tensor([2.0])
    result = torch.add(child(inner_left), parent(inner_right))
    assert torch.equal(result, torch.full_like(inner_left, 1.0))
    assert handler.calls[0][2] == (child, parent)


# ------------------------------------------------------------------------------
# _apply_fn_to_outer_tensor
# ------------------------------------------------------------------------------


_VIEW_OP_CASES = (
    pytest.param(torch.ops.aten.alias.default, (), (2, 3), (3, 1), id="alias"),
    pytest.param(torch.ops.aten.transpose.int, (0, 1), (3, 2), (1, 3), id="transpose"),
    pytest.param(torch.ops.aten.t.default, (), (3, 2), (1, 3), id="t"),
    pytest.param(torch.ops.aten.permute.default, ((1, 0),), (3, 2), (1, 3), id="permute"),
    pytest.param(torch.ops.aten.movedim.int, (0, 1), (3, 2), (1, 3), id="movedim"),
    pytest.param(torch.ops.aten.squeeze.default, (), (2, 3), (3, 1), id="squeeze"),
    pytest.param(torch.ops.aten.unsqueeze.default, (0,), (1, 2, 3), (6, 3, 1), id="unsqueeze"),
    pytest.param(torch.ops.aten.select.int, (0, 0), (3,), (1,), id="select"),
    pytest.param(torch.ops.aten.slice.Tensor, (0, 0, 1, 1), (1, 3), (3, 1), id="slice"),
    pytest.param(torch.ops.aten.narrow.default, (0, 0, 1), (1, 3), (3, 1), id="narrow"),
    pytest.param(torch.ops.aten.diagonal.default, (0, 0, 1), (2,), (4,), id="diagonal"),
    pytest.param(torch.ops.aten.expand.default, ((4, 2, 3),), (4, 2, 3), (0, 3, 1), id="expand"),
    pytest.param(torch.ops.aten.flatten.using_ints, (0, 1), (6,), (1,), id="flatten"),
    pytest.param(torch.ops.aten.unflatten.int, (0, (1, 2)), (1, 2, 3), (6, 3, 1), id="unflatten"),
    pytest.param(torch.ops.aten.as_strided.default, ((3, 2), (1, 3), 0), (3, 2), (1, 3), id="as_strided"),
    pytest.param(torch.ops.aten.reshape.default, ((6,),), (6,), (1,), id="reshape"),
    pytest.param(torch.ops.aten.view.default, ((3, 2),), (3, 2), (2, 1), id="view"),
)


@pytest.mark.parametrize(("func", "args", "expected_shape", "expected_stride"), _VIEW_OP_CASES)
def test_apply_fn_to_outer_tensor_applies_view_op_to_wrapper_shell(
    func: OpOverload,
    args: tuple[Any, ...],
    expected_shape: tuple[int, ...],
    expected_stride: tuple[int, ...],
) -> None:
    """Test that _apply_fn_to_outer_tensor applies metadata-only ops to the outer shell."""
    wrapper = _Wrapper(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    version = wrapper._version

    shell = wrapper._apply_fn_to_outer_tensor(func, *args)

    assert type(shell) is _Wrapper
    assert shell is not wrapper
    assert shell._is_view()
    assert shell._base is wrapper
    assert shell.shape == expected_shape
    assert shell.stride() == expected_stride
    assert shell._version == version
    assert vars(shell) == {}

    torch._C._increment_version([wrapper])
    assert shell._version == wrapper._version


def test_apply_fn_to_outer_tensor_forwards_arguments() -> None:
    """Test that _apply_fn_to_outer_tensor forwards positional and keyword arguments."""
    wrapper = _Wrapper(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    calls: list[tuple[torch.Tensor, int, int, str]] = []

    def view_fn(tensor: torch.Tensor, dim0: int, dim1: int, *, label: str) -> torch.Tensor:
        """Record the call and transpose the tensor."""
        calls.append((tensor, dim0, dim1, label))
        return torch.ops.aten.transpose.int(tensor, dim0, dim1)

    shell = wrapper._apply_fn_to_outer_tensor(view_fn, 0, 1, label="test")

    assert len(calls) == 1
    assert calls[0][0] is wrapper
    assert calls[0][1:] == (0, 1, "test")
    assert type(shell) is _Wrapper
    assert shell.shape == (3, 2)
    assert shell.stride() == (1, 3)


# ------------------------------------------------------------------------------
# _tensor_field_aliasing_group
# ------------------------------------------------------------------------------


def test_tensor_field_aliasing_group_groups_shared_tensors() -> None:
    """Test that shared inner tensors are grouped by object identity."""
    shared = torch.arange(4, dtype=torch.float32)
    wrapper = _Wrapper(shared, bias=shared)

    assert wrapper._tensor_field_aliasing_group() == (("payload", "bias"),)


def test_tensor_field_aliasing_group_orders_distinct_groups_by_first_occurrence() -> None:
    """Test that distinct inner tensors form groups in field order."""
    payload = torch.arange(4, dtype=torch.float32)
    bias = torch.ones(4, dtype=torch.float32)
    wrapper = _Wrapper(payload, bias=bias)

    assert wrapper._tensor_field_aliasing_group() == (("payload",), ("bias",))


def test_tensor_field_aliasing_group_handles_empty_wrapper() -> None:
    """Test that an empty wrapper has no aliasing groups."""
    assert _EmptyWrapper()._tensor_field_aliasing_group() == ()


# ------------------------------------------------------------------------------
# _has_same_tensor_field_aliasing
# ------------------------------------------------------------------------------


def test_has_same_tensor_field_aliasing_accepts_matching_groups() -> None:
    """Test that matching group structures are accepted."""
    source = _AliasWrapper(
        torch.arange(4, dtype=torch.float32),
        torch.arange(4, dtype=torch.float32),
    )
    target = _AliasWrapper(
        torch.ones(4, dtype=torch.float32),
        torch.ones(4, dtype=torch.float32),
    )

    assert target._has_same_tensor_field_aliasing(source._tensor_field_aliasing_group())


def test_has_same_tensor_field_aliasing_accepts_matching_shared_groups() -> None:
    """Test that matching shared-tensor groups are accepted."""
    source_shared = torch.arange(4, dtype=torch.float32)
    target_shared = torch.ones(4, dtype=torch.float32)
    source = _AliasWrapper(source_shared, source_shared)
    target = _AliasWrapper(target_shared, target_shared)

    assert target._has_same_tensor_field_aliasing(source._tensor_field_aliasing_group())


def test_has_same_tensor_field_aliasing_rejects_mismatch_without_raising() -> None:
    """Test that a mismatch returns False when errors are disabled."""
    shared = torch.arange(4, dtype=torch.float32)
    source = _AliasWrapper(shared, shared)
    target = _AliasWrapper(
        torch.arange(4, dtype=torch.float32),
        torch.ones(4, dtype=torch.float32),
    )

    assert not target._has_same_tensor_field_aliasing(source._tensor_field_aliasing_group())


def test_has_same_tensor_field_aliasing_raises_on_mismatch() -> None:
    """Test that a mismatch raises with both alias groups."""
    shared = torch.arange(4, dtype=torch.float32)
    source = _AliasWrapper(shared, shared)
    target = _AliasWrapper(
        torch.arange(4, dtype=torch.float32),
        torch.ones(4, dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="field aliasing differs"):
        target._has_same_tensor_field_aliasing(
            source._tensor_field_aliasing_group(),
            raise_on_mismatch=True,
        )


# ==============================================================================
# Clone handlers
# ==============================================================================

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


class _AliasWrapper(MokitWrapperTensorBase):
    """A wrapper with two inner tensor fields for alias-topology tests."""

    positional_inner_tensor_names = ("left", "right")

    def __new__(
        cls,
        left: torch.Tensor,
        right: torch.Tensor,
        outer_size: Sequence[int | torch.SymInt] | None = None,
        outer_stride: Sequence[int | torch.SymInt] | None = None,
        *,
        storage_offset: int | None = None,
    ) -> Self:
        """Create the wrapper tensor."""
        if outer_size is None:
            outer_size = left.shape
        if outer_stride is None:
            outer_stride = left.stride()
        return torch.Tensor._make_wrapper_subclass(
            cls,
            outer_size,
            outer_stride,
            storage_offset=storage_offset,
            dtype=left.dtype,
            device=left.device,
        )

    def __init__(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        outer_size: Sequence[int | torch.SymInt] | None = None,
        outer_stride: Sequence[int | torch.SymInt] | None = None,
        *,
        storage_offset: int | None = None,
    ) -> None:
        """Store the inner tensors."""
        self.left = left
        self.right = right


class _WrapperSubclass(_Wrapper):
    """A wrapper subclass used to test dispatch-class checks."""


class _Uncomparable:
    """A metadata value whose equality comparison raises."""

    def __eq__(self, _other: object) -> bool:
        """Reject equality comparison."""
        raise RuntimeError("comparison is unsupported")


# ------------------------------------------------------------------------------
# aten.clone.default
# ------------------------------------------------------------------------------


def test_clone_default_clones_inner_tensors_and_preserves_metadata() -> None:
    """Test that clone.default clones inner tensors and preserves metadata references."""
    inner = torch.arange(4, dtype=torch.float32)
    tag = ["source"]
    note = {"note": 1}
    wrapper = _Wrapper(inner, tag, bias=torch.ones(4), note=note)

    cloned = wrapper.clone()

    assert type(cloned) is _Wrapper
    assert cloned is not wrapper
    assert cloned.payload is not wrapper.payload
    assert torch.equal(cloned.payload, wrapper.payload)
    assert cloned.bias is not wrapper.bias
    assert torch.equal(cloned.bias, wrapper.bias)
    assert cloned.tag is tag
    assert cloned.note is note
    assert cloned.shape == wrapper.shape
    assert cloned.stride() == wrapper.stride()


def test_clone_default_deduplicates_shared_inner_tensors() -> None:
    """Test that clone.default clones an inner tensor once per object identity."""
    shared = torch.arange(4, dtype=torch.float32)
    wrapper = _AliasWrapper(shared, shared)

    cloned = torch.ops.aten.clone.default(wrapper)

    assert cloned.left is cloned.right
    assert cloned.left is not shared
    assert torch.equal(cloned.left, shared)


def test_clone_default_does_not_preserve_view_storage_aliasing() -> None:
    """Test that clone.default gives distinct views independent clones."""
    storage = torch.arange(6, dtype=torch.float32)
    wrapper = _AliasWrapper(storage[:3], storage[3:])

    cloned = torch.ops.aten.clone.default(wrapper)

    assert cloned.left is not cloned.right
    assert cloned.left.data_ptr() != cloned.right.data_ptr()
    assert torch.equal(cloned.left, storage[:3])
    assert torch.equal(cloned.right, storage[3:])


def test_clone_default_clones_nested_wrapper_inner_tensor() -> None:
    """Test that clone.default recursively clones an inner wrapper tensor."""
    leaf = torch.arange(4, dtype=torch.float32)
    nested_tag = object()
    outer_tag = object()
    nested = _Wrapper(leaf, tag=nested_tag)
    outer = _Wrapper(nested, tag=outer_tag, bias=torch.zeros(4, dtype=torch.float32))

    cloned = torch.ops.aten.clone.default(outer)

    assert type(cloned) is _Wrapper
    assert cloned is not outer
    assert cloned.tag is outer_tag
    assert cloned.payload is not nested
    assert type(cloned.payload) is _Wrapper
    assert cloned.payload.tag is nested_tag
    assert cloned.payload.payload is not leaf
    assert torch.equal(cloned.payload.payload, leaf)
    assert cloned.payload.shape == nested.shape
    assert cloned.payload.stride() == nested.stride()


def test_clone_default_deduplicates_shared_nested_wrappers() -> None:
    """Test that clone.default clones a shared nested wrapper once."""
    shared = _Wrapper(torch.arange(4, dtype=torch.float32))
    wrapper = _AliasWrapper(shared, shared)

    cloned = torch.ops.aten.clone.default(wrapper)

    assert cloned.left is cloned.right
    assert cloned.left is not shared
    assert cloned.left.payload is not shared.payload


@pytest.mark.parametrize("memory_format", [None, torch.preserve_format])
def test_clone_default_preserves_outer_stride_without_explicit_format(
    memory_format: torch.memory_format | None,
) -> None:
    """Test that clone.default preserves the outer stride without an explicit format."""
    inner = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    wrapper = _Wrapper(inner, outer_size=(2, 3), outer_stride=(1, 2))

    cloned = torch.ops.aten.clone.default(wrapper, memory_format=memory_format)

    assert cloned.stride() == (1, 2)


@pytest.mark.parametrize(
    ("inner", "memory_format", "expected_stride"),
    [
        (torch.arange(6, dtype=torch.float32).reshape(2, 3).t(), torch.contiguous_format, (2, 1)),
        (torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2), torch.channels_last, (12, 1, 6, 3)),
    ],
)
def test_clone_default_recomputes_outer_stride_for_explicit_memory_format(
    inner: torch.Tensor,
    memory_format: torch.memory_format,
    expected_stride: tuple[int, ...],
) -> None:
    """Test that clone.default applies an explicit memory format to outer and inner geometry."""
    wrapper = _Wrapper(inner)

    cloned = torch.ops.aten.clone.default(wrapper, memory_format=memory_format)

    assert cloned.stride() == expected_stride
    assert cloned.payload.stride() == expected_stride


def test_clone_default_handles_empty_wrapper() -> None:
    """Test that clone.default reconstructs a wrapper with no fields."""
    wrapper = _EmptyWrapper(outer_size=(2, 3), outer_stride=(3, 1))

    cloned = torch.ops.aten.clone.default(wrapper)

    assert type(cloned) is _EmptyWrapper
    assert cloned is not wrapper
    assert cloned.shape == wrapper.shape
    assert cloned.stride() == wrapper.stride()


def test_clone_default_rejects_dispatch_class_mismatch() -> None:
    """Test that clone.default rejects a mismatched dispatch class."""
    handler = MokitWrapperTensorBase._ATEN_OP_TABLE[torch.ops.aten.clone.default]
    wrapper = _Wrapper(torch.zeros(2, 3))

    with pytest.raises(AssertionError, match="does not match type"):
        handler(_WrapperSubclass, torch.ops.aten.clone.default, (type(wrapper),), (wrapper,), {})


# ------------------------------------------------------------------------------
# aten.clone.out
# ------------------------------------------------------------------------------


def test_clone_out_updates_existing_inner_tensors_and_returns_out() -> None:
    """Test that clone.out copies into out's inner tensors and returns out."""
    source = _Wrapper(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    out = _Wrapper(torch.full((2, 3), -1.0, dtype=torch.float32))
    payload = out.payload
    bias = out.bias
    version = out._version

    returned = torch.ops.aten.clone.out(source, out=out)

    assert returned is out
    assert out.payload is payload
    assert out.bias is bias
    assert torch.equal(out.payload, source.payload)
    assert torch.equal(out.bias, source.bias)
    assert out._version == version + 1


def test_clone_out_updates_nested_wrapper_in_place() -> None:
    """Test that clone.out recursively updates a nested wrapper's inner tensor."""
    source = _Wrapper(
        _Wrapper(torch.arange(4, dtype=torch.float32)),
        bias=torch.zeros(4, dtype=torch.float32),
    )
    out_nested = _Wrapper(torch.zeros(4, dtype=torch.float32))
    out = _Wrapper(out_nested, bias=torch.zeros(4, dtype=torch.float32))
    nested_ref = out.payload
    leaf_ref = out_nested.payload

    returned = torch.ops.aten.clone.out(source, out=out)

    assert returned is out
    assert out.payload is nested_ref
    assert out_nested.payload is leaf_ref
    assert torch.equal(out_nested.payload, cast(_Wrapper, source.payload).payload)


def test_clone_out_rejects_nested_wrapper_metadata_mismatch() -> None:
    """Test that clone.out propagates validation from a nested wrapper."""
    source = _Wrapper(
        _Wrapper(torch.arange(4, dtype=torch.float32), tag="source"),
        bias=torch.zeros(4, dtype=torch.float32),
    )
    out_nested = _Wrapper(torch.zeros(4, dtype=torch.float32), tag="out")
    out = _Wrapper(out_nested, bias=torch.zeros(4, dtype=torch.float32))

    with pytest.raises(ValueError, match="metadata 'tag' mismatch"):
        torch.ops.aten.clone.out(source, out=out)


def test_clone_out_accepts_identical_metadata_objects() -> None:
    """Test that clone.out accepts metadata fields that are the same object."""
    tag = object()
    source = _Wrapper(torch.zeros(2, 3), tag=tag)
    out = _Wrapper(torch.zeros(2, 3), tag=tag)

    returned = torch.ops.aten.clone.out(source, out=out)

    assert returned is out
    assert out.tag is tag


@pytest.mark.parametrize(
    ("make_out", "message"),
    [
        pytest.param(lambda: _Wrapper(torch.zeros(2, 3, dtype=torch.float64)), "dtype mismatch", id="dtype"),
        pytest.param(lambda: _Wrapper(torch.zeros(2, 3, device="meta")), "device mismatch", id="device"),
        pytest.param(
            lambda: _Wrapper(torch.zeros(2, 3), layout=torch.sparse_coo),
            "layout mismatch",
            id="layout",
        ),
        pytest.param(
            lambda: _Wrapper(torch.zeros(2, 3), outer_size=(3, 2), outer_stride=(2, 1)),
            "shape mismatch",
            id="shape",
        ),
        pytest.param(lambda: _Wrapper(torch.zeros(2, 3), outer_stride=(1, 2)), "stride mismatch", id="stride"),
    ],
)
def test_clone_out_rejects_outer_attribute_mismatch(make_out: Callable[[], _Wrapper], message: str) -> None:
    """Test that clone.out rejects mismatched outer attributes."""
    source = _Wrapper(torch.arange(6, dtype=torch.float32).reshape(2, 3))

    with pytest.raises(ValueError, match=message):
        torch.ops.aten.clone.out(source, out=make_out())


@pytest.mark.parametrize(
    ("source_cls", "out_cls"),
    [
        pytest.param(_Wrapper, _WrapperSubclass, id="source-base"),
        pytest.param(_WrapperSubclass, _Wrapper, id="out-base"),
    ],
)
def test_clone_out_rejects_out_of_different_class(source_cls: type[_Wrapper], out_cls: type[_Wrapper]) -> None:
    """Test that clone.out rejects a source and out of different wrapper classes."""
    source = source_cls(torch.zeros(2, 3))
    out = out_cls(torch.zeros(2, 3))

    with pytest.raises(AssertionError, match="does not match"):
        torch.ops.aten.clone.out(source, out=out)


def test_clone_out_rejects_metadata_mismatch() -> None:
    """Test that clone.out rejects unequal declared metadata."""
    source = _Wrapper(torch.zeros(2, 3), tag="source")
    out = _Wrapper(torch.zeros(2, 3), tag="out")

    with pytest.raises(ValueError, match="metadata 'tag' mismatch"):
        torch.ops.aten.clone.out(source, out=out)


def test_clone_out_rejects_non_comparable_metadata() -> None:
    """Test that clone.out rejects metadata that cannot be compared."""
    source = _Wrapper(torch.zeros(2, 3), tag=_Uncomparable())
    out = _Wrapper(torch.zeros(2, 3), tag=_Uncomparable())

    with pytest.raises(ValueError, match="not equality-comparable"):
        torch.ops.aten.clone.out(source, out=out)


def test_clone_out_keeps_out_metadata_when_values_are_equal() -> None:
    """Test that clone.out keeps out's metadata when it equals the source metadata."""
    source_tag = ["source"]
    out_tag = ["source"]
    source = _Wrapper(torch.zeros(2, 3), tag=source_tag)
    out = _Wrapper(torch.zeros(2, 3), tag=out_tag)

    returned = torch.ops.aten.clone.out(source, out=out)

    assert returned is out
    assert out.tag is out_tag
    assert out.tag is not source_tag


def test_clone_out_rejects_alias_topology_mismatch() -> None:
    """Test that clone.out rejects differing exact-object alias topologies."""
    shared = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = _AliasWrapper(shared, shared)
    out = _AliasWrapper(torch.zeros(2, 3), torch.zeros(2, 3))

    with pytest.raises(ValueError, match="field aliasing differs"):
        torch.ops.aten.clone.out(source, out=out)


def test_clone_out_rejects_nested_alias_topology_mismatch() -> None:
    """Test that clone.out checks exact-object topology for nested wrappers."""
    shared = _Wrapper(torch.arange(4, dtype=torch.float32))
    source = _AliasWrapper(shared, shared)
    out = _AliasWrapper(
        _Wrapper(torch.zeros(4, dtype=torch.float32)),
        _Wrapper(torch.zeros(4, dtype=torch.float32)),
    )

    with pytest.raises(ValueError, match="field aliasing differs"):
        torch.ops.aten.clone.out(source, out=out)


def test_clone_out_accepts_matching_nested_alias_topology() -> None:
    """Test that clone.out recursively copies a shared nested wrapper."""
    source_nested = _Wrapper(torch.arange(4, dtype=torch.float32))
    out_nested = _Wrapper(torch.zeros(4, dtype=torch.float32))
    source = _AliasWrapper(source_nested, source_nested)
    out = _AliasWrapper(out_nested, out_nested)

    returned = torch.ops.aten.clone.out(source, out=out)

    assert returned is out
    assert out.left is out.right
    assert torch.equal(out_nested.payload, source_nested.payload)


def test_clone_out_accepts_matching_alias_topology() -> None:
    """Test that clone.out accepts matching exact-object alias topologies."""
    shared = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out_left = torch.zeros(2, 3)
    source = _AliasWrapper(shared, shared)
    out = _AliasWrapper(out_left, out_left)

    returned = torch.ops.aten.clone.out(source, out=out)

    assert returned is out
    assert out.left is out.right
    assert torch.equal(out.left, shared)


def test_clone_out_allows_distinct_views_sharing_storage() -> None:
    """Test that clone.out does not inspect storage-level view aliasing."""
    storage = torch.arange(4, dtype=torch.float32)
    source = _AliasWrapper(storage[:2], storage[2:])
    out_left = torch.zeros(2)
    out_right = torch.zeros(2)
    out = _AliasWrapper(out_left, out_right)

    returned = torch.ops.aten.clone.out(source, out=out)

    assert returned is out
    assert out.left is out_left
    assert out.right is out_right
    assert torch.equal(out.left, storage[:2])
    assert torch.equal(out.right, storage[2:])


def test_clone_out_ignores_outer_storage_offset() -> None:
    """Test that clone.out ignores the synthetic outer storage offset."""
    left = torch.arange(4, dtype=torch.float32)
    right = torch.ones(4)
    source = _AliasWrapper(left, right, storage_offset=1)
    out = _AliasWrapper(torch.zeros(4), torch.zeros(4), storage_offset=7)

    returned = torch.ops.aten.clone.out(source, out=out)

    assert returned is out
    assert torch.equal(out.left, left)
    assert torch.equal(out.right, right)


def test_clone_out_forwards_memory_format_without_validating_outer_format() -> None:
    """Test that clone.out forwards memory_format without validating outer memory format."""
    inner = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    source = _Wrapper(inner)
    out = _Wrapper(torch.zeros(2, 3).t())

    returned = torch.ops.aten.clone.out(source, memory_format=torch.contiguous_format, out=out)

    assert returned is out
    assert torch.equal(out.payload, source.payload)


def test_clone_out_handles_empty_wrapper() -> None:
    """Test that clone.out handles a wrapper with no fields."""
    source = _EmptyWrapper(outer_size=(2, 3), outer_stride=(3, 1))
    out = _EmptyWrapper(outer_size=(2, 3), outer_stride=(3, 1))
    version = out._version

    returned = torch.ops.aten.clone.out(source, out=out)

    assert returned is out
    assert out._version == version + 1


# ==============================================================================
# aten.alias.default
# ==============================================================================


def test_alias_default_aliases_inner_tensors_and_shares_metadata() -> None:
    """Test that alias.default aliases inner tensors and shares metadata."""
    inner = torch.arange(4, dtype=torch.float32)
    tag = ["tag"]
    note = {"note": 1}
    wrapper = _Wrapper(inner, tag, bias=torch.ones(4), note=note)

    aliased = torch.ops.aten.alias.default(wrapper)

    assert type(aliased) is _Wrapper
    assert aliased is not wrapper
    assert aliased._is_view()
    assert aliased._base is wrapper
    assert aliased.payload is not wrapper.payload
    assert aliased.payload.data_ptr() == wrapper.payload.data_ptr()
    assert aliased.bias is not wrapper.bias
    assert aliased.bias.data_ptr() == wrapper.bias.data_ptr()
    assert aliased.tag is tag
    assert aliased.note is note
    assert aliased.shape == wrapper.shape
    assert aliased.stride() == wrapper.stride()


def test_alias_default_initializes_via_subclass_init() -> None:
    """Test that alias.default calls __init__ without invoking __new__."""

    class InitWrapper(_Wrapper):
        """A wrapper that derives state in its initializer."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.derived: object = self.tag

    tag = object()
    wrapper = InitWrapper(torch.arange(4, dtype=torch.float32), tag)

    aliased = torch.ops.aten.alias.default(wrapper)

    assert aliased.derived is tag


def test_alias_default_mutations_are_visible_through_both_wrappers() -> None:
    """Test that alias.default shares storage through the inner tensors."""
    inner = torch.arange(4, dtype=torch.float32)
    wrapper = _Wrapper(inner)
    aliased = torch.ops.aten.alias.default(wrapper)

    aliased.payload.add_(1)

    assert wrapper.payload is inner
    assert torch.equal(inner, torch.arange(1, 5, dtype=torch.float32))


def test_alias_default_aliases_shared_inner_tensor_once() -> None:
    """Test that alias.default aliases a shared inner tensor once."""
    shared = torch.arange(4, dtype=torch.float32)
    wrapper = _AliasWrapper(shared, shared)

    aliased = torch.ops.aten.alias.default(wrapper)

    assert aliased.left is aliased.right
    assert aliased.left is not shared
    assert aliased.left.data_ptr() == shared.data_ptr()


def test_alias_default_recursively_aliases_nested_wrapper() -> None:
    """Test that alias.default recursively aliases a nested wrapper inner tensor."""
    nested_tag = object()
    outer_tag = object()
    nested = _Wrapper(torch.arange(4, dtype=torch.float32), tag=nested_tag)
    wrapper = _Wrapper(nested, tag=outer_tag, bias=torch.zeros(4, dtype=torch.float32))

    aliased = torch.ops.aten.alias.default(wrapper)

    assert aliased.tag is outer_tag
    assert aliased.payload is not nested
    assert type(aliased.payload) is _Wrapper
    assert aliased.payload.tag is nested_tag
    assert aliased.payload.payload is not nested.payload
    assert aliased.payload.payload.data_ptr() == nested.payload.data_ptr()


def test_alias_default_shares_outer_version_counter() -> None:
    """Test that alias.default shares the outer version counter."""
    wrapper = _Wrapper(torch.arange(4, dtype=torch.float32))
    aliased = torch.ops.aten.alias.default(wrapper)
    version = wrapper._version

    assert aliased._version == version

    torch._C._increment_version([wrapper])
    assert wrapper._version == version + 1
    assert aliased._version == wrapper._version

    torch._C._increment_version([aliased])
    assert wrapper._version == version + 2
    assert aliased._version == wrapper._version


def test_alias_default_inner_version_counters_are_independent() -> None:
    """Test that inner aliases do not share version counters below dispatch."""
    inner = torch.arange(4, dtype=torch.float32)
    wrapper = _Wrapper(inner)
    aliased = torch.ops.aten.alias.default(wrapper)
    version = inner._version
    alias_version = aliased.payload._version

    torch._C._increment_version([inner])
    assert inner._version == version + 1
    assert aliased.payload._version == alias_version

    torch._C._increment_version([aliased.payload])
    assert inner._version == version + 1
    assert aliased.payload._version == alias_version + 1


def test_alias_default_preserves_outer_geometry_and_autograd() -> None:
    """Test that alias.default preserves geometry and creates an autograd view."""
    inner = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    wrapper = _Wrapper(inner, outer_size=(2, 3), outer_stride=(1, 2), requires_grad=True)

    aliased = torch.ops.aten.alias.default(wrapper)

    assert aliased.shape == wrapper.shape
    assert aliased.stride() == wrapper.stride()
    assert aliased.requires_grad
    assert aliased.grad_fn is not None
    assert aliased._base is wrapper


def test_alias_default_preserves_outer_layout() -> None:
    """Test that alias.default preserves the outer layout."""
    wrapper = _Wrapper(torch.arange(4, dtype=torch.float32), layout=torch.sparse_coo)

    aliased = torch.ops.aten.alias.default(wrapper)

    assert aliased.layout is torch.sparse_coo


def test_alias_default_preserves_outer_storage_offset() -> None:
    """Test that alias.default preserves the outer storage offset."""
    wrapper = _AliasWrapper(
        torch.arange(4, dtype=torch.float32),
        torch.ones(4, dtype=torch.float32),
        storage_offset=7,
    )

    aliased = torch.ops.aten.alias.default(wrapper)

    assert aliased.storage_offset() == 7
    assert aliased.left is not wrapper.left
    assert aliased.left.data_ptr() == wrapper.left.data_ptr()
    assert aliased.right is not wrapper.right
    assert aliased.right.data_ptr() == wrapper.right.data_ptr()


def test_alias_default_handles_empty_wrapper() -> None:
    """Test that alias.default handles a wrapper with no fields."""
    wrapper = _EmptyWrapper(outer_size=(2, 3), outer_stride=(3, 1))
    version = wrapper._version

    aliased = torch.ops.aten.alias.default(wrapper)

    assert type(aliased) is _EmptyWrapper
    assert aliased is not wrapper
    assert aliased.shape == wrapper.shape
    assert aliased.stride() == wrapper.stride()
    assert aliased._version == version


def test_alias_default_rejects_dispatch_class_mismatch() -> None:
    """Test that alias.default rejects a mismatched dispatch class."""
    handler = MokitWrapperTensorBase._ATEN_OP_TABLE[torch.ops.aten.alias.default]
    wrapper = _Wrapper(torch.zeros(2, 3))

    with pytest.raises(AssertionError, match="does not match type"):
        handler(_WrapperSubclass, torch.ops.aten.alias.default, (type(wrapper),), (wrapper,), {})


# ==============================================================================
# aten.detach.default
# ==============================================================================


def test_detach_default_detaches_outer_and_inner_tensors() -> None:
    """Test that detach.default detaches the wrapper and its inner tensors."""
    inner = torch.arange(4, dtype=torch.float32, requires_grad=True)
    bias = torch.ones(4, dtype=torch.float32, requires_grad=True)
    tag = ["tag"]
    note = {"note": 1}
    wrapper = _Wrapper(inner, tag, bias=bias, note=note, requires_grad=True)

    detached = torch.ops.aten.detach.default(wrapper)

    assert type(detached) is _Wrapper
    assert detached is not wrapper
    assert not detached._is_view()
    assert detached._base is None
    assert not detached.requires_grad
    assert detached.grad_fn is None
    assert detached.tag is tag
    assert detached.note is note
    assert detached.payload is not wrapper.payload
    assert detached.payload.data_ptr() == wrapper.payload.data_ptr()
    assert not detached.payload.requires_grad
    assert detached.payload.grad_fn is None
    assert not detached.payload._is_view()
    assert detached.payload._base is None
    assert detached.bias is not wrapper.bias
    assert detached.bias.data_ptr() == wrapper.bias.data_ptr()
    assert not detached.bias.requires_grad
    assert detached.shape == wrapper.shape
    assert detached.stride() == wrapper.stride()


def test_detach_default_mutations_are_visible_through_both_wrappers() -> None:
    """Test that detach.default shares storage through the inner tensors."""
    inner = torch.arange(4, dtype=torch.float32)
    wrapper = _Wrapper(inner)
    detached = torch.ops.aten.detach.default(wrapper)

    detached.payload.add_(1)

    assert wrapper.payload is inner
    assert torch.equal(inner, torch.arange(1, 5, dtype=torch.float32))


def test_detach_default_detaches_shared_inner_tensor_once() -> None:
    """Test that detach.default detaches a shared inner tensor once."""
    shared = torch.arange(4, dtype=torch.float32)
    wrapper = _AliasWrapper(shared, shared)

    detached = torch.ops.aten.detach.default(wrapper)

    assert detached.left is detached.right
    assert detached.left is not shared
    assert detached.left.data_ptr() == shared.data_ptr()


def test_detach_default_recursively_detaches_nested_wrapper() -> None:
    """Test that detach.default recursively detaches a nested wrapper."""
    nested_tag = object()
    outer_tag = object()
    nested = _Wrapper(torch.arange(4, dtype=torch.float32), tag=nested_tag)
    wrapper = _Wrapper(nested, tag=outer_tag, bias=torch.zeros(4, dtype=torch.float32))

    detached = torch.ops.aten.detach.default(wrapper)

    assert detached.tag is outer_tag
    assert detached.payload is not nested
    assert type(detached.payload) is _Wrapper
    assert not detached.payload._is_view()
    assert detached.payload._base is None
    assert detached.payload.tag is nested_tag
    assert detached.payload.payload is not nested.payload
    assert detached.payload.payload.data_ptr() == nested.payload.data_ptr()


def test_detach_default_shares_outer_version_counter() -> None:
    """Test that detach.default shares the outer version counter."""
    wrapper = _Wrapper(torch.arange(4, dtype=torch.float32))
    detached = torch.ops.aten.detach.default(wrapper)
    version = wrapper._version

    assert detached._version == version

    torch._C._increment_version([wrapper])
    assert wrapper._version == version + 1
    assert detached._version == wrapper._version

    torch._C._increment_version([detached])
    assert wrapper._version == version + 2
    assert detached._version == wrapper._version


def test_detach_default_inner_version_counters_are_independent() -> None:
    """Test that inner detach results do not share version counters below dispatch."""
    inner = torch.arange(4, dtype=torch.float32)
    wrapper = _Wrapper(inner)
    detached = torch.ops.aten.detach.default(wrapper)
    version = inner._version
    detached_version = detached.payload._version

    torch._C._increment_version([inner])
    assert inner._version == version + 1
    assert detached.payload._version == detached_version

    torch._C._increment_version([detached.payload])
    assert inner._version == version + 1
    assert detached.payload._version == detached_version + 1


def test_detach_default_initializes_via_subclass_init() -> None:
    """Test that detach.default calls __init__ without invoking __new__."""

    class InitWrapper(_Wrapper):
        """A wrapper that derives state in its initializer."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.derived: object = self.tag

    tag = object()
    wrapper = InitWrapper(torch.arange(4, dtype=torch.float32), tag)

    detached = torch.ops.aten.detach.default(wrapper)

    assert detached.derived is tag


def test_detach_default_handles_empty_wrapper() -> None:
    """Test that detach.default handles a wrapper with no fields."""
    wrapper = _EmptyWrapper(outer_size=(2, 3), outer_stride=(3, 1))
    version = wrapper._version

    detached = torch.ops.aten.detach.default(wrapper)

    assert type(detached) is _EmptyWrapper
    assert detached is not wrapper
    assert not detached._is_view()
    assert detached._base is None
    assert detached.shape == wrapper.shape
    assert detached.stride() == wrapper.stride()
    assert detached._version == version


def test_detach_default_rejects_dispatch_class_mismatch() -> None:
    """Test that detach.default rejects a mismatched dispatch class."""
    handler = MokitWrapperTensorBase._ATEN_OP_TABLE[torch.ops.aten.detach.default]
    wrapper = _Wrapper(torch.zeros(2, 3))

    with pytest.raises(AssertionError, match="does not match type"):
        handler(_WrapperSubclass, torch.ops.aten.detach.default, (type(wrapper),), (wrapper,), {})


# ==============================================================================
# aten.copy_
# ==============================================================================


def test_copy_copies_inner_tensors_in_place() -> None:
    """Test that copy_ overwrites inner tensors and returns the destination."""
    source = _Wrapper(
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        bias=torch.ones(2, 3),
    )
    destination = _Wrapper(torch.zeros(2, 3))
    payload = destination.payload
    bias = destination.bias
    version = destination._version

    returned = torch.ops.aten.copy_.default(destination, source)

    assert returned is destination
    assert destination.payload is payload
    assert destination.bias is bias
    assert torch.equal(destination.payload, source.payload)
    assert torch.equal(destination.bias, source.bias)
    assert destination._version == version + 1


def test_copy_preserves_destination_metadata() -> None:
    """Test that copy_ retains the destination's metadata references."""
    source_tag = ["source"]
    destination_tag = ["source"]
    source = _Wrapper(torch.arange(4, dtype=torch.float32), tag=source_tag)
    destination = _Wrapper(torch.zeros(4, dtype=torch.float32), tag=destination_tag)

    torch.ops.aten.copy_.default(destination, source)

    assert destination.tag is destination_tag
    assert destination.tag is not source_tag


def test_copy_accepts_matching_alias_groups() -> None:
    """Test that copy_ copies through matching shared-tensor groups."""
    shared = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = _AliasWrapper(shared, shared)
    out_left = torch.zeros(2, 3)
    destination = _AliasWrapper(out_left, out_left)

    torch.ops.aten.copy_.default(destination, source)

    assert destination.left is destination.right
    assert torch.equal(destination.left, shared)


def test_copy_recursively_updates_nested_wrapper_in_place() -> None:
    """Test that copy_ recursively updates a nested wrapper's inner tensors."""
    nested_source = _Wrapper(torch.arange(4, dtype=torch.float32))
    source = _Wrapper(nested_source, bias=torch.ones(4, dtype=torch.float32))
    nested_destination = _Wrapper(torch.full((4,), -1.0, dtype=torch.float32))
    destination = _Wrapper(nested_destination, bias=torch.zeros(4, dtype=torch.float32))
    nested_payload = nested_destination.payload

    returned = torch.ops.aten.copy_.default(destination, source)

    assert returned is destination
    assert destination.payload is nested_destination
    assert nested_destination.payload is nested_payload
    assert torch.equal(nested_destination.payload, nested_source.payload)
    assert torch.equal(destination.bias, source.bias)


def test_copy_rejects_alias_group_mismatch() -> None:
    """Test that copy_ rejects differing field alias groups."""
    shared = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = _AliasWrapper(shared, shared)
    destination = _AliasWrapper(torch.zeros(2, 3), torch.zeros(2, 3))

    with pytest.raises(ValueError, match="field aliasing differs"):
        torch.ops.aten.copy_.default(destination, source)


@pytest.mark.parametrize(
    ("make_destination", "message"),
    [
        pytest.param(
            lambda: _Wrapper(torch.zeros(2, 3, dtype=torch.float64)),
            "dtype mismatch",
            id="dtype",
        ),
        pytest.param(
            lambda: _Wrapper(torch.zeros(2, 3, device="meta")),
            "device mismatch",
            id="device",
        ),
        pytest.param(
            lambda: _Wrapper(torch.zeros(2, 3), layout=torch.sparse_coo),
            "layout mismatch",
            id="layout",
        ),
        pytest.param(
            lambda: _Wrapper(torch.zeros(2, 3), outer_size=(3, 2), outer_stride=(2, 1)),
            "shape mismatch",
            id="shape",
        ),
        pytest.param(
            lambda: _Wrapper(torch.zeros(2, 3), outer_stride=(1, 2)),
            "stride mismatch",
            id="stride",
        ),
    ],
)
def test_copy_rejects_outer_attribute_mismatch(
    make_destination: Callable[[], _Wrapper],
    message: str,
) -> None:
    """Test that copy_ rejects mismatched outer attributes."""
    source = _Wrapper(torch.arange(6, dtype=torch.float32).reshape(2, 3))

    with pytest.raises(ValueError, match=message):
        torch.ops.aten.copy_.default(make_destination(), source)


def test_copy_rejects_metadata_mismatch() -> None:
    """Test that copy_ rejects unequal declared metadata."""
    source = _Wrapper(torch.zeros(2, 3), tag="source")
    destination = _Wrapper(torch.zeros(2, 3), tag="destination")

    with pytest.raises(ValueError, match="metadata 'tag' mismatch"):
        torch.ops.aten.copy_.default(destination, source)


def test_copy_rejects_different_source_class() -> None:
    """Test that copy_ rejects a source of a different wrapper class."""
    source = _WrapperSubclass(torch.zeros(2, 3))
    destination = _Wrapper(torch.zeros(2, 3))

    with pytest.raises(AssertionError, match="does not match type"):
        torch.ops.aten.copy_.default(destination, source)


def test_copy_rejects_plain_tensor_source() -> None:
    """Test that copy_ rejects a plain tensor source."""
    destination = _Wrapper(torch.zeros(2, 3))

    with pytest.raises(ValueError, match="type mismatch"):
        torch.ops.aten.copy_.default(destination, torch.zeros(2, 3))
