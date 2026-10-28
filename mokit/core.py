# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Core utilities for MOKIT."""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, Self, TypeVar, cast, overload

import torch
from torch._custom_class_base import CustomClassBase
from torch._ops import OpOverload


@dataclass(frozen=True, slots=True, kw_only=True)
class MokitConfigBase:
    """The base class for all MOKIT config classes."""


class _TorchFunctionImpl(Protocol):
    """Unbound implementation of a torch function dispatcher."""

    def __call__(
        self,
        cls: type[Any],
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch a torch function."""


class _TorchDispatchImpl(Protocol):
    """Unbound implementation of an ATen operator dispatcher."""

    def __call__(
        self,
        cls: type[Any],
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch an ATen operator."""


class MokitTensorBase(torch.Tensor):
    """The base class for all MOKIT tensor classes."""


_HandlerT = TypeVar("_HandlerT", bound=Callable[..., Any])


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

    ``__tensor_flatten__`` and ``__tensor_unflatten__`` are PyTorch's PT2 tracing protocol.
    They do not implement pickle support.

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

    _TORCH_FN_TABLE: ClassVar[dict[Callable[..., Any] | None, Callable[..., Any]]] = {}
    _ATEN_OP_TABLE: ClassVar[dict[OpOverload | None, Callable[..., Any]]] = {}

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
    def implements_torch_function(
        cls,
        *torch_fns: Callable[..., Any],
        override: bool = False,
    ) -> Callable[[_HandlerT], _HandlerT]:
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

        def decorator(func: _HandlerT) -> _HandlerT:
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
    def implements(
        cls,
        *aten_ops: OpOverload,
        override: bool = False,
    ) -> Callable[[_HandlerT], _HandlerT]:
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

        def decorator(func: _HandlerT) -> _HandlerT:
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
    make_method: Callable[[type[MokitWrapperTensorBase]], _TorchFunctionImpl],
) -> None: ...


@overload
def _register_handlers(
    cls: type[MokitWrapperTensorBase],
    table_name: Literal["_ATEN_OP_TABLE"],
    method_name: Literal["__torch_dispatch__"],
    keys: tuple[OpOverload | None, ...],
    func: Callable[..., Any],
    override: bool,
    make_method: Callable[[type[MokitWrapperTensorBase]], _TorchDispatchImpl],
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
        _TorchFunctionImpl | _TorchDispatchImpl,
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
) -> _TorchFunctionImpl:
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
) -> _TorchDispatchImpl:
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
