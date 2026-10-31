# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper tensor base class and its dispatch helpers."""

import functools
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Literal, Protocol, Self, cast, overload

import torch
from torch._custom_class_base import CustomClassBase
from torch._ops import OpOverload

from mokit.core import MokitTensorBase


class HandlerProtocol[
    ClsType: type[Any],
    FuncType: Callable[..., Any] | OpOverload,
    TypesType: tuple[type[Any], ...],
    **Params,
    ReturnType,
](Protocol):
    """The protocol for a handler with unpacked arguments."""

    def __call__(
        self,
        cls: ClsType,
        func: FuncType,
        types: TypesType,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> ReturnType:
        """Dispatch an operation to the handler."""


class PackedHandlerProtocol[
    ClsType: type[Any],
    FuncType: Callable[..., Any] | OpOverload,
    TypesType: tuple[type[Any], ...],
    ReturnType,
](Protocol):
    """The protocol for a handler with packed arguments."""

    def __call__(
        self,
        cls: ClsType,
        func: FuncType,
        types: TypesType,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
    ) -> ReturnType:
        """Invoke the packed handler."""


def unpack_args[
    ClsType: type[Any],
    FuncType: Callable[..., Any] | OpOverload,
    TypesType: tuple[type[Any], ...],
    **Params,
    ReturnType,
](
    handler: HandlerProtocol[ClsType, FuncType, TypesType, Params, ReturnType],
) -> PackedHandlerProtocol[ClsType, FuncType, TypesType, ReturnType]:
    """
    Adapt a handler with unpacked arguments to the packed calling convention.

    Args:
        handler: Handler accepting ``(cls, func, types, *args, **kwargs)``.

    Returns:
        A handler accepting ``(cls, func, types, args, kwargs)`` and preserving the
        handler's metadata parameter types and result type.

    """

    @functools.wraps(handler)
    def packed(
        cls: ClsType,
        func: FuncType,
        types: TypesType,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
    ) -> ReturnType:
        """Unpack the dispatch arguments and invoke the concrete handler."""
        return handler(cls, func, types, *args, **(kwargs if kwargs is not None else {}))

    return packed


class TorchFunctionImpl(
    PackedHandlerProtocol[
        type[Any],
        Callable[..., Any],
        tuple[type[Any], ...],
        Any,
    ],
    Protocol,
):
    """Unbound implementation of a torch function dispatcher."""


class TorchDispatchImpl(
    PackedHandlerProtocol[
        type[Any],
        OpOverload,
        tuple[type[Any], ...],
        Any,
    ],
    Protocol,
):
    """Unbound implementation of an ATen operator dispatcher."""


@overload
def _call_handler_from_table(
    owner_cls: type[MokitWrapperTensorBase],
    table_name: Literal["_TORCH_FN_TABLE"],
    cls: type[MokitWrapperTensorBase],
    func: Callable[..., Any],
    types: tuple[type[Any], ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> tuple[bool, Any]: ...


@overload
def _call_handler_from_table(
    owner_cls: type[MokitWrapperTensorBase],
    table_name: Literal["_ATEN_OP_TABLE"],
    cls: type[MokitWrapperTensorBase],
    func: OpOverload,
    types: tuple[type[Any], ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> tuple[bool, Any]: ...


def _call_handler_from_table(
    owner_cls: type[MokitWrapperTensorBase],
    table_name: Literal["_TORCH_FN_TABLE", "_ATEN_OP_TABLE"],
    cls: type[MokitWrapperTensorBase],
    func: Callable[..., Any] | OpOverload,
    types: tuple[type[Any], ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> tuple[bool, Any]:
    """
    Call a handler for ``func`` from ``owner_cls``'s handler table.

    Args:
        owner_cls: Class whose table is checked.
        table_name: Name of the table attribute.
        cls: Runtime dispatch class passed to the handler.
        func: Dispatched function or operator.
        types: Participating argument types.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        A tuple of whether a handler was found and the handler result.

    """
    table = owner_cls.__dict__[table_name]

    handler = table.get(func)
    if handler is None:
        handler = table.get(None)

    if handler is None:
        return False, None
    else:
        return True, handler(cls, func, types, args, kwargs)


@overload
def _register_handlers(
    cls: type[MokitWrapperTensorBase],
    table_name: Literal["_TORCH_FN_TABLE"],
    method_name: Literal["__torch_function__"],
    keys: tuple[Callable[..., Any] | None, ...],
    func: Callable[..., Any],
    override: bool,
    make_method: Callable[[type[MokitWrapperTensorBase]], TorchFunctionImpl],
) -> None: ...


@overload
def _register_handlers(
    cls: type[MokitWrapperTensorBase],
    table_name: Literal["_ATEN_OP_TABLE"],
    method_name: Literal["__torch_dispatch__"],
    keys: tuple[OpOverload | None, ...],
    func: Callable[..., Any],
    override: bool,
    make_method: Callable[[type[MokitWrapperTensorBase]], TorchDispatchImpl],
) -> None: ...


def _register_handlers(
    cls: type[MokitWrapperTensorBase],
    table_name: Literal["_TORCH_FN_TABLE", "_ATEN_OP_TABLE"],
    method_name: Literal["__torch_function__", "__torch_dispatch__"],
    keys: tuple[Callable[..., Any] | OpOverload | None, ...],
    func: Callable[..., Any],
    override: bool,
    make_method: Callable[
        [type[MokitWrapperTensorBase]],
        TorchFunctionImpl | TorchDispatchImpl,
    ],
) -> None:
    """
    Register all handler keys for one decorator application.

    Validation failures - a non-callable handler, inconsistent method/table state,
    a non-dict table, or key conflicts - leave the class unchanged. Failures while
    installing the generated method or writing the table are fatal.

    Args:
        cls: Class receiving the handlers.
        table_name: Handler table attribute name.
        method_name: Dispatch method attribute name.
        keys: Handler keys, including ``None`` for a wildcard.
        func: Handler callable.
        override: Whether to replace existing handlers.
        make_method: Factory for the generated dispatch implementation.

    Raises:
        TypeError: If the handler or table is invalid.
        KeyError: If the class/table state is inconsistent or keys conflict.

    """
    if not callable(func):
        raise TypeError("handler must be callable")

    has_method = method_name in cls.__dict__
    table = cls.__dict__.get(table_name)

    if has_method and table is None:
        raise KeyError(f"{cls.__name__} defines its own {method_name} but does not define its own {table_name}.")

    if table is not None:
        if type(table) is not dict:
            raise TypeError(f"{table_name} must be a dict.")

        conflicts = tuple(key for key in keys if key in table)
        if conflicts and not override:
            raise KeyError(f"Handlers already registered for {conflicts!r}.")

    if not has_method:
        setattr(cls, method_name, classmethod(make_method(cls)))

    entries = {key: func for key in keys}
    if table is None:
        setattr(cls, table_name, entries)
    else:
        table.update(entries)


def _make_subclass_torch_function(
    owner_cls: type[MokitWrapperTensorBase],
) -> TorchFunctionImpl:
    """
    Create an unbound torch function dispatcher for ``owner_cls``.

    Args:
        owner_cls: Class owning the handler table.

    Returns:
        The unbound dispatcher implementation.

    """

    def __torch_function__(
        cls: type[MokitWrapperTensorBase],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:

        found, result = _call_handler_from_table(owner_cls, "_TORCH_FN_TABLE", cls, func, types, args, kwargs)
        if found:
            return result

        parent = super(owner_cls, cls)  # pyrefly: ignore[invalid-argument]
        return parent.__torch_function__(func, types, args, kwargs)

    return __torch_function__


def _make_subclass_torch_dispatch(
    owner_cls: type[MokitWrapperTensorBase],
) -> TorchDispatchImpl:
    """
    Create an unbound ATen operator dispatcher for ``owner_cls``.

    Args:
        owner_cls: Class owning the handler table.

    Returns:
        The unbound dispatcher implementation.

    """

    def __torch_dispatch__(
        cls: type[MokitWrapperTensorBase],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:

        found, result = _call_handler_from_table(owner_cls, "_ATEN_OP_TABLE", cls, func, types, args, kwargs)
        if found:
            return result

        parent = super(owner_cls, cls)  # pyrefly: ignore[invalid-argument]
        return parent.__torch_dispatch__(func, types, args, kwargs)

    return __torch_dispatch__


class MokitWrapperTensorBase(MokitTensorBase):
    """
    The base class for MOKIT wrapper tensor subclasses.

    Subclasses inherit the dispatch helpers and the PT2 flatten/unflatten helpers. All four
    name tuples below exist by default and are empty; an empty tuple means there are no
    fields of that kind. When overridden, each must be a tuple of strings, no name may
    appear more than once across the four tuples, and none may use the reserved names
    ``outer_size`` or ``outer_stride``.

    Every name listed in any of the four attributes declares a field that must exist on the
    instance. The base does not validate field values; a field may be ``None``. PyTorch's
    tracing protocol additionally requires each inner-tensor name to resolve to a tensor or
    a registered opaque type.

    The attributes follow normal Python MRO inheritance; if multiple bases define them, the
    first definition in the MRO is used. A subclass that adds, removes, or renames a
    flattened field must override the affected tuples explicitly, for example by extending
    the parent tuple. The base cannot detect fields that are not declared in them.

    Attributes:
        inner_tensor_names: Names of inner tensors passed positionally to ``__init__`` and
            ``__new__``, in order.
        metadata_names: Names of non-``Tensor`` attributes passed positionally to
            ``__init__`` and ``__new__``, in order.
        keyword_inner_tensor_names: Names of inner tensors passed as keyword arguments to
            ``__init__`` and ``__new__``.
        keyword_metadata_names: Names of non-``Tensor`` attributes passed as keyword
            arguments to ``__init__`` and ``__new__``.

    Both ``__new__`` and ``__init__`` must accept calls of the form::

        cls(
            *inner_tensor_values,
            *metadata_values,
            outer_size=outer_size,
            outer_stride=outer_stride,
            **keyword_inner_tensor_values,
            **keyword_metadata_values,
        )

    The first ``*`` group contains the values named by ``inner_tensor_names``; the second
    contains the values named by ``metadata_names``. The ``**`` groups use the names from
    the keyword tuples. The two geometry arguments may default to ``None`` for direct
    construction, but ``__tensor_unflatten__`` always passes them explicitly. They are
    reserved keyword names and must not appear in the keyword tuples.

    The outer ``storage_offset`` is synthetic for a wrapper and is ignored. The outer
    ``layout`` is meaningful: it defaults to ``torch.strided`` and is controlled by the
    subclass through ``torch.Tensor._make_wrapper_subclass``. Because
    ``__tensor_unflatten__`` does not receive ``layout``, a subclass that exposes a
    non-strided or variable layout must preserve it itself, for example through metadata
    or class-specific construction. ``layout`` is not a reserved field name. A declared
    field named ``layout`` is allowed as long as ``getattr`` resolves it; for example, a
    subclass can pass the value to ``_make_wrapper_subclass`` without assigning
    ``self.layout``, so ``getattr`` reads the resulting tensor property. Direct assignment
    to ``self.layout`` is not possible because ``Tensor.layout`` is read-only.

    ``__tensor_flatten__`` and ``__tensor_unflatten__`` are PyTorch's PT2 tracing protocol.
    They do not implement pickle support.

    ``is_contiguous`` and ``contiguous`` describe the logical tensor defined by
    ``outer_size`` and ``outer_stride`` by default. Inner tensors are representation
    details and may have different layouts, so the base does not forward ``contiguous``
    to them; an already outer-contiguous wrapper returns itself without copying its inner
    tensors. A subclass that wants representation-level contiguity can override
    ``torch.Tensor.is_contiguous`` and ``torch.Tensor.contiguous`` with
    ``implements_torch_function``, keeping both definitions consistent.

    Dispatch registration and forwarding follow this workflow:

    1. A concrete subclass registers handlers with ``implements_torch_function`` for
       torch functions or ``implements`` for ``OpOverload`` operators. The decorator
       factory is side-effect free; applying the returned decorator validates the request
       and then commits it. Validation failures - invalid, duplicate, or unhashable keys,
       a non-callable handler, inconsistent class state, or key conflicts - leave the
       class unchanged. Failures while installing the generated method or writing the
       table are fatal. Registration ensures the class owns
       ``_TORCH_FN_TABLE`` or ``_ATEN_OP_TABLE`` and installs a class-specific
       ``__torch_function__`` or ``__torch_dispatch__`` when the class does not already
       define one, using the corresponding ``_make_subclass_*`` factory. Each table maps
       a function or operator to its handler; ``None`` is the wildcard key, and
       ``override`` controls replacement of existing keys.

    2. At runtime, PyTorch calls the class-specific dispatch method. That method uses
       ``_call_handler_from_table`` to search the current class's own table, checking the
       exact function or operator first and the wildcard second. A match is invoked as
       ``handler(cls, func, types, args, kwargs)``, where ``cls`` is the runtime subclass.
       Forwarding is plain and performs no argument processing, so a handler must accept
       the same signature as ``__torch_function__`` or ``__torch_dispatch__``.

    3. If the current class has no match, the generated method calls the next
       ``__torch_function__`` or ``__torch_dispatch__`` in the MRO with
       ``super(owner_cls, cls)``. Each ancestor with a generated dispatcher repeats the
       same table lookup. The effective order is therefore: runtime-class exact,
       runtime-class wildcard, parent exact, parent wildcard, continuing through the MRO.

    4. The root ``MokitWrapperTensorBase`` methods perform the final lookup. If no root
       handler matches, ``__torch_function__`` returns ``NotImplemented`` unless ``cls``
       is a subclass of every participating type; otherwise it forwards to the original
       function with subclass dispatch disabled. ``__torch_dispatch__`` instead raises
       ``NotImplementedError``. The handshake is applied only at this final layer, so
       registered handlers in any class can still claim mixed-type calls.

    5. A handler can continue the MRO lookup for the same key by calling
       ``super(Owner, cls).__torch_dispatch__(func, types, args, kwargs)`` or the
       ``__torch_function__`` counterpart. This is how a child handler composes with the
       parent handler it overrides.

    6. A class that defines its own ``__torch_function__`` or ``__torch_dispatch__``
       bypasses the generated dispatcher. The custom method must own the corresponding
       table and consult or delegate to it itself; otherwise registered handlers are
       unreachable.

    Handlers are operator-specific. When registering one, annotate its parameters and
    return value to match that operator's actual dispatch call rather than the general
    ``__torch_dispatch__`` or ``__torch_function__`` signature. For example, the default
    ``aten.clone.default`` handler declares ``args: tuple[WrapperType]``,
    ``kwargs: dict[str, Any]``, and a ``WrapperType`` return value. Registration accepts
    ``Callable[..., Any]`` and does not enforce these annotations, but accurate annotations
    document each operator's contract.

    """

    inner_tensor_names: ClassVar[tuple[str, ...]] = ()
    metadata_names: ClassVar[tuple[str, ...]] = ()
    keyword_inner_tensor_names: ClassVar[tuple[str, ...]] = ()
    keyword_metadata_names: ClassVar[tuple[str, ...]] = ()

    _FLATTEN_ATTR_NAMES: ClassVar[tuple[str, ...]] = (
        "inner_tensor_names",
        "metadata_names",
        "keyword_inner_tensor_names",
        "keyword_metadata_names",
    )

    _TORCH_FN_TABLE: ClassVar[dict[Callable[..., Any] | None, TorchFunctionImpl]] = {}
    _ATEN_OP_TABLE: ClassVar[dict[OpOverload | None, TorchDispatchImpl]] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Validate the four flatten-protocol class attributes after subclass creation.

        Args:
            **kwargs: Class keyword arguments forwarded to the next base class.

        Raises:
            TypeError: If an attribute is not a tuple of strings.
            ValueError: If a name appears more than once across the four tuples or uses a
                reserved name.

        """
        super().__init_subclass__(**kwargs)

        for attr_name in MokitWrapperTensorBase._FLATTEN_ATTR_NAMES:
            attr = getattr(cls, attr_name)
            if not isinstance(attr, tuple):
                raise TypeError(f"The class attribute '{attr_name}' must be a tuple.")
            if not all(isinstance(name, str) for name in attr):
                raise TypeError(f"The class attribute '{attr_name}' must be a tuple of strings.")

        declared_field_counter = Counter(
            field for attr_name in MokitWrapperTensorBase._FLATTEN_ATTR_NAMES for field in getattr(cls, attr_name)
        )

        duplicated_fields = tuple(field for field, freq in declared_field_counter.items() if freq > 1)
        if duplicated_fields:
            raise ValueError(f"Field name(s) {sorted(duplicated_fields)!r} are declared more than once.")

        RESERVED_FIELDS = ("outer_size", "outer_stride")

        conflict_fields = tuple(field for field in RESERVED_FIELDS if field in declared_field_counter)
        if conflict_fields:
            raise ValueError(f"Reserved field name(s) {sorted(conflict_fields)!r} cannot be used.")

    def __tensor_flatten__(self) -> tuple[Sequence[str], Mapping[str, Any]]:
        """
        Flatten the wrapper into inner tensor names and metadata.

        Returns:
            The names of the inner tensors and the metadata mapping.

        """
        cls = type(self)
        tensor_names = (*cls.inner_tensor_names, *cls.keyword_inner_tensor_names)
        metadata = {name: getattr(self, name) for name in (*cls.metadata_names, *cls.keyword_metadata_names)}
        return tensor_names, metadata

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors: Mapping[str, torch.Tensor | CustomClassBase],
        metadata: Mapping[str, Any],
        outer_size: Sequence[int | torch.SymInt],
        outer_stride: Sequence[int | torch.SymInt],
    ) -> Self:
        """
        Rebuild a wrapper from inner tensors, metadata, and outer geometry.

        Args:
            inner_tensors: The inner tensors, keyed by their flattened names.
            metadata: The metadata mapping returned by ``__tensor_flatten__``.
            outer_size: The requested outer size.
            outer_stride: The requested outer stride.

        Returns:
            The rebuilt wrapper tensor.

        """
        cls_inner_tensor = (inner_tensors[name] for name in cls.inner_tensor_names)
        keyword_inner_tensors = {name: inner_tensors[name] for name in cls.keyword_inner_tensor_names}
        cls_metadata = (metadata[name] for name in cls.metadata_names)
        keyword_metadata = {name: metadata[name] for name in cls.keyword_metadata_names}

        constructor = cast(Callable[..., Self], cls)
        return constructor(
            *cls_inner_tensor,
            *cls_metadata,
            outer_size=outer_size,
            outer_stride=outer_stride,
            **keyword_inner_tensors,
            **keyword_metadata,
        )

    def __repr__(self, *, tensor_contents: str | None = None) -> str:
        """
        Return a representation of the wrapper and its declared fields.

        Declared field values are formatted with ``repr``. Tensor attributes are printed
        first; declared fields follow, skipping names already printed as tensor
        attributes.

        Args:
            tensor_contents: Optional pre-rendered content string. If not ``None``, it is
                placed first.

        Returns:
            The wrapper representation.

        """
        fields = [tensor_contents] if tensor_contents is not None else []
        printed_names: set[str] = set()

        device = self.device
        if (
            device.type != torch.get_default_device().type
            or (device.type == "cuda" and torch.cuda.current_device() != device.index)
            or device.type == "mps"
        ):
            fields.append(f"device='{device}'")
            printed_names.add("device")

        # Mirror the dtype suffix logic of ``torch.Tensor.__repr__``.
        default_complex_dtype = torch.cdouble if torch.get_default_dtype() == torch.double else torch.cfloat
        if self.dtype not in (torch.get_default_dtype(), default_complex_dtype, torch.int64, torch.bool):
            fields.append(f"dtype={self.dtype}")
            printed_names.add("dtype")

        if self.layout != torch.strided:
            fields.append(f"layout={self.layout}")
            printed_names.add("layout")

        # Mirror the grad_fn rendering logic of ``torch.Tensor.__repr__``.
        try:
            grad_fn = self.grad_fn
        except RuntimeError:
            grad_fn_name: str | None = "Invalid"
        else:
            if grad_fn is None:
                grad_fn_name = None
            else:
                grad_fn_name = type(grad_fn).__name__
                if grad_fn_name == "CppFunction":
                    grad_fn_name = grad_fn.name().rsplit("::", 1)[-1]

        if grad_fn_name is not None:
            fields.append(f"grad_fn=<{grad_fn_name}>")
            printed_names.add("grad_fn")
        elif self.requires_grad:
            fields.append("requires_grad=True")
            printed_names.add("requires_grad")

        fields.extend(
            f"{name}={getattr(self, name)!r}"
            for attr_name in MokitWrapperTensorBase._FLATTEN_ATTR_NAMES
            for name in getattr(self, attr_name)
            if name not in printed_names
        )

        return f"{type(self).__name__}({', '.join(fields)})"

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """
        Dispatch a torch function through registered handlers.

        Args:
            func: The torch function being called.
            types: Types of the participating arguments.
            args: Positional arguments.
            kwargs: Keyword arguments, or ``None``.

        Returns:
            A registered handler result, ``NotImplemented`` if ``cls`` is not a
            subclass of every participating type, or the default function result.

        """
        found, result = _call_handler_from_table(
            MokitWrapperTensorBase, "_TORCH_FN_TABLE", cls, func, types, args, kwargs
        )
        if found:
            return result

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        if kwargs is None:
            kwargs = {}

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def implements_torch_function[HandlerType: Callable[..., Any]](
        cls,
        *torch_fns: Callable[..., Any],
        override: bool = False,
    ) -> Callable[[HandlerType], HandlerType]:
        """
        Return a decorator that registers torch function handlers on this class.

        The handler must accept ``(cls, func, types, args, kwargs)``. If no functions are
        given, the handler is registered as a wildcard for unmatched functions. The
        factory validates ``override``, key types, hashability, and duplicate keys. The
        returned decorator performs the registration when applied.

        Args:
            *torch_fns: Torch functions to handle. If empty, registers a wildcard handler.
            override: Whether to replace handlers for keys already present in the handler table.

        Returns:
            A decorator that registers and returns the handler.

        Raises:
            TypeError: If any of the following hold:
                * The factory receives a non-bool ``override``.
                * The factory receives a key that is not callable.
                * The factory receives a key that is not hashable.

            KeyError: If the factory receives a duplicate key.

        """
        if not isinstance(override, bool):
            raise TypeError("override must be a bool")

        for fn in torch_fns:
            if not callable(fn):
                raise TypeError(f"torch function key must be callable, got {fn!r}")

        fns: tuple[Callable[..., Any] | None, ...] = torch_fns if torch_fns else (None,)

        try:
            unique_fns = set(fns)
        except TypeError as exc:
            raise TypeError("torch function keys must be hashable") from exc
        if len(unique_fns) != len(fns):
            raise KeyError("Torch function keys must be unique within one registration.")

        def decorator(func: HandlerType) -> HandlerType:
            _register_handlers(
                cls,
                "_TORCH_FN_TABLE",
                "__torch_function__",
                fns,
                func,
                override,
                _make_subclass_torch_function,
            )
            return func

        return decorator

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]
        cls,
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """
        Dispatch an ATen operator through registered handlers.

        Args:
            func: The ATen operator being called.
            types: Types of the participating arguments.
            args: Positional arguments.
            kwargs: Keyword arguments, or ``None``.

        Returns:
            A registered handler result.

        Raises:
            NotImplementedError: If no handler is registered for the operator.

        """
        found, result = _call_handler_from_table(
            MokitWrapperTensorBase, "_ATEN_OP_TABLE", cls, func, types, args, kwargs
        )
        if found:
            return result

        arg_types = tuple(type(arg) for arg in args)
        kwarg_types = {param: type(arg) for param, arg in kwargs.items()} if kwargs is not None else None
        raise NotImplementedError(
            f"{cls.__name__} dispatch: attempting to run unimplemented operator/function: "
            f"{func=}, {types=}, {arg_types=}, {kwarg_types=}"
        )

    @classmethod
    def implements[HandlerType: Callable[..., Any]](
        cls,
        *aten_ops: OpOverload,
        override: bool = False,
    ) -> Callable[[HandlerType], HandlerType]:
        """
        Return a decorator that registers ATen operator handlers on this class.

        The handler must accept ``(cls, func, types, args, kwargs)``. If no operators are
        given, the handler is registered as a wildcard for unmatched operators. The
        factory validates ``override``, key types, hashability, and duplicate keys. The
        returned decorator performs the registration when applied. Each operator must be
        an ``OpOverload``; an ``OpOverloadPacket`` such as ``torch.ops.aten.permute`` is
        rejected, so pass a specific overload such as ``torch.ops.aten.permute.default``.

        Args:
            *aten_ops: ATen operator overloads to handle. If empty, registers a wildcard handler.
            override: Whether to replace handlers for keys already present in the handler table.

        Returns:
            A decorator that registers and returns the handler.

        Raises:
            TypeError: If any of the following hold:
                * The factory receives a non-bool ``override``.
                * The factory receives an operator that is not an ``OpOverload``.
                * The factory receives a key that is not hashable.

            KeyError: If the factory receives a duplicate key.

        """
        if not isinstance(override, bool):
            raise TypeError("override must be a bool")

        for op in aten_ops:
            if not isinstance(op, OpOverload):
                # OpOverloadPacket (e.g. torch.ops.aten.permute) is intentionally rejected.
                raise TypeError(
                    f"aten op must be an OpOverload, got {op!r}; "
                    "register a specific overload such as torch.ops.aten.permute.default"
                )

        ops: tuple[OpOverload | None, ...] = aten_ops if aten_ops else (None,)

        try:
            unique_ops = set(ops)
        except TypeError as exc:
            raise TypeError("aten operator keys must be hashable") from exc
        if len(unique_ops) != len(ops):
            raise KeyError("Aten operator keys must be unique within one registration.")

        def decorator(func: HandlerType) -> HandlerType:
            _register_handlers(
                cls,
                "_ATEN_OP_TABLE",
                "__torch_dispatch__",
                ops,
                func,
                override,
                _make_subclass_torch_dispatch,
            )
            return func

        return decorator


implements = MokitWrapperTensorBase.implements


@implements(torch.ops.aten.clone.default)
def _[WrapperType: MokitWrapperTensorBase](
    cls: type[WrapperType],
    func: OpOverload,
    types: tuple[type[WrapperType]],
    args: tuple[WrapperType],
    kwargs: dict[str, Any],
) -> WrapperType:
    """
    Clone a wrapper tensor by cloning each inner tensor once per object identity.

    Two field names that reference the same tensor object share one clone, preserving
    exact-object aliasing such as tied fields. Distinct tensor objects that share storage
    through views receive independent clones, so their storage-level aliasing is not
    preserved. Detecting aliasing by data pointer would require cloning the shared storage
    once and replaying each view; that is view- and wrapper-specific, so the base class
    deliberately does not attempt it.

    Metadata is passed through by reference, not deep-copied. The outer size is preserved.
    The outer stride is preserved for ``None`` or ``torch.preserve_format``; for an
    explicit memory format it is recomputed from the requested format.

    Args:
        cls: Runtime wrapper class whose ``__tensor_unflatten__`` reconstructs the result.
        func: The ``aten.clone.default`` operator.
        types: Participating dispatch types.
        args: Positional operator arguments; ``args[0]`` is the wrapper to clone.
        kwargs: Keyword operator arguments, including ``memory_format`` when supplied.

    Returns:
        A new wrapper of the same runtime class.

    Raises:
        AssertionError: If assertions are enabled and ``cls`` does not match the runtime
            class of ``args[0]``.

    """
    assert type(args[0]) is cls, (
        f"{func}: dispatch cls={cls.__name__} does not match type(args[0])={type(args[0]).__name__}"
    )

    (wrapper,) = args

    tensor_names, metadata = wrapper.__tensor_flatten__()

    id_cloned_tensor: dict[int, torch.Tensor] = {}
    cloned_inner_tensors: dict[str, torch.Tensor] = {}

    for name in tensor_names:
        tensor: torch.Tensor = getattr(wrapper, name)
        tensor_id: int = id(tensor)

        if tensor_id not in id_cloned_tensor:
            id_cloned_tensor[tensor_id] = func(tensor, **kwargs)

        cloned_inner_tensors[name] = id_cloned_tensor[tensor_id]

    outer_size = wrapper.size()

    memory_format: torch.memory_format | None = kwargs.get("memory_format")
    if memory_format in (None, torch.preserve_format):
        outer_stride = wrapper.stride()
    else:
        # memory_format only defines strides for the strided layout. Passing
        # wrapper.layout to this probe would produce meaningless or unsupported
        # sparse strides; non-strided wrappers with explicit formats are
        # subclass-specific.
        outer_stride = torch.empty(
            outer_size,
            device="meta",
            memory_format=memory_format,
        ).stride()

    return cls.__tensor_unflatten__(cloned_inner_tensors, metadata, outer_size, outer_stride)


@implements(torch.ops.aten.clone.out)
def _[WrapperType: MokitWrapperTensorBase](
    cls: type[WrapperType],
    func: OpOverload,
    types: tuple[type[WrapperType]],
    args: tuple[WrapperType],
    kwargs: dict[str, Any],
) -> WrapperType:
    """
    Clone a wrapper into an existing wrapper of the same runtime class.

    Outer dtype, device, layout, shape, and stride must match. The declared metadata of
    ``out`` must compare equal to the source; it is checked but never rewritten. Inner
    tensors are updated through ``aten.clone.out`` using the corresponding inner tensors of
    ``out``, so their objects and storage remain those of ``out``.

    Policy:

    - ``out`` is authoritative for class-declared metadata.
    - Only exact-object alias topology is checked; distinct views sharing storage are
      outside the base contract.
    - The outer ``storage_offset`` is synthetic and ignored.
    - No outer ``memory_format`` validation is performed; it is forwarded to the inner
      ``aten.clone.out`` calls.
    - The outer version counter is bumped by PyTorch's ``Tensor(a!) out`` handling.

    Args:
        cls: Runtime wrapper class shared by the source and ``out``.
        func: The ``aten.clone.out`` operator.
        types: Participating dispatch types.
        args: Positional operator arguments; ``args[0]`` is the source wrapper.
        kwargs: Keyword operator arguments; ``kwargs["out"]`` is the destination wrapper.

    Returns:
        The ``out`` wrapper after its inner tensors have been updated.

    Raises:
        AssertionError: If assertions are enabled and either the source or ``out`` does
            not match the dispatch class.
        ValueError: If outer dtype, device, layout, shape, or stride differs; if declared
            metadata differs or is not equality-comparable; or if exact-object inner
            alias topology differs.

    """
    (wrapper,) = args
    out = kwargs.pop("out")

    assert type(args[0]) is cls, (
        f"{func}: dispatch cls={cls.__name__} does not match type(args[0])={type(args[0]).__name__}"
    )
    assert type(out) is cls, f"{func}: out type={type(out).__name__} does not match dispatch cls={cls.__name__}"

    if wrapper.dtype is not out.dtype:
        raise ValueError(f"{func}: dtype mismatch: wrapper={wrapper.dtype}, out={out.dtype}")

    if wrapper.device != out.device:
        raise ValueError(f"{func}: device mismatch: wrapper={wrapper.device}, out={out.device}")

    if wrapper.layout is not out.layout:
        raise ValueError(f"{func}: layout mismatch: wrapper={wrapper.layout}, out={out.layout}")

    if wrapper.shape != out.shape:
        raise ValueError(f"{func}: shape mismatch: wrapper={wrapper.shape}, out={out.shape}")

    if wrapper.stride() != out.stride():
        raise ValueError(f"{func}: stride mismatch: wrapper={wrapper.stride()}, out={out.stride()}")

    tensor_names, metadata = wrapper.__tensor_flatten__()

    for name, value in metadata.items():
        out_value = getattr(out, name)

        if value is out_value:
            continue

        try:
            equal = bool(value == out_value)
        except Exception as exc:
            raise ValueError(f"{func}: metadata {name!r} is not equality-comparable") from exc

        if not equal:
            raise ValueError(f"{func}: metadata {name!r} differs: wrapper={value!r}, out={out_value!r}")

    # Check exact-object alias topology: both wrappers must partition their field names
    # into the same groups of identical tensor objects. View-level storage aliasing is
    # intentionally not examined.
    wrapper_first_occurrence: dict[int, int] = {}
    out_first_occurrence: dict[int, int] = {}
    for index, name in enumerate(tensor_names):
        wrapper_index = wrapper_first_occurrence.setdefault(id(getattr(wrapper, name)), index)
        out_index = out_first_occurrence.setdefault(id(getattr(out, name)), index)

        if wrapper_index != out_index:
            raise ValueError(f"{func}: inner-tensor alias topology differs at field {name!r}")

    for first_index in wrapper_first_occurrence.values():
        name = tensor_names[first_index]
        func(getattr(wrapper, name), **kwargs, out=getattr(out, name))

    return out
