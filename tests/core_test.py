# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.core`."""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Self, cast

import pytest
import torch
from torch._ops import OpOverload

from mokit.core import MokitConfigBase, MokitWrapperTensorBase

# ==============================================================================
# MokitConfigBase
# ==============================================================================


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


# ==============================================================================
# MokitWrapperTensorBase
# ==============================================================================

# ------------------------------------------------------------------------------
# __repr__
# ------------------------------------------------------------------------------


class _Wrapper(MokitWrapperTensorBase):
    """A wrapper whose four flatten attributes cover every field kind."""

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

    inner_tensor_names = ("payload",)

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
