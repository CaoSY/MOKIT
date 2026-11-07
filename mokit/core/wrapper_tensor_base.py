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
from torch.utils._python_dispatch import return_and_correct_aliasing

from mokit.core import MokitTensorBase
from mokit.utils import apply_fn


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
        """
        Dispatch an ATen operator through the handler table of ``owner_cls``.

        Registered handler results are corrected with
        ``return_and_correct_aliasing``. See
        :meth:`MokitWrapperTensorBase.__torch_dispatch__` for the detailed aliasing
        behavior.

        Args:
            cls: Runtime wrapper class.
            func: The ATen operator being called.
            types: Types of the participating arguments.
            args: Positional arguments.
            kwargs: Keyword arguments, or ``None``.

        Returns:
            The handler result after schema-declared aliasing correction.

        """
        found, result = _call_handler_from_table(owner_cls, "_ATEN_OP_TABLE", cls, func, types, args, kwargs)
        if found:
            return return_and_correct_aliasing(func, args, kwargs, result)

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
    ``outer_size``, ``outer_stride``, ``metadata``, ``positional_metadata``,
    ``keyword_metadata``, ``inner_tensor_names``, or ``metadata_names``.

    Every name listed in any of the four attributes declares a field that must exist on the
    instance. The base does not validate field values; a field may be ``None``. PyTorch's
    tracing protocol additionally requires each inner-tensor name to resolve to a tensor or
    a registered opaque type.

    The attributes follow normal Python MRO inheritance; if multiple bases define them, the
    first definition in the MRO is used. A subclass that adds, removes, or renames a
    flattened field must override the affected tuples explicitly, for example by extending
    the parent tuple. The base cannot detect fields that are not declared in them.

    Attributes:
        positional_inner_tensor_names: Names of inner tensors passed positionally to ``__init__`` and
            ``__new__``, in order.
        positional_metadata_names: Names of non-``Tensor`` attributes passed positionally to
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

    The first ``*`` group contains the values named by ``positional_inner_tensor_names``; the second
    contains the values named by ``positional_metadata_names``. The ``**`` groups use the names from
    the keyword tuples. The two geometry arguments may default to ``None`` for direct
    construction, but ``__tensor_unflatten__`` always passes them explicitly. They are
    reserved keyword names and must not appear in the keyword tuples.

    ``__new__`` is responsible for creating or customizing the outer shell; ``__init__``
    is responsible for initializing the subclass's Python fields. ``__new__`` receives
    the same arguments as ``__init__`` but does not initialize fields.

    The base implements ``__new__`` as a thin adapter that maps recognized outer metadata
    from the constructor arguments and delegates to ``_make_outer_tensor``. It recognizes
    ``dtype``, ``layout``, ``device``, ``pin_memory``, ``requires_grad``,
    ``storage_offset``, ``memory_format``, ``storage_size``,
    ``dispatch_sizes_strides_policy``, ``dispatch_device``, ``dispatch_layout``, and
    ``_extra_dispatch_keys`` in the constructor keyword arguments and otherwise uses
    factory defaults.

    ``_make_outer_tensor`` is a classmethod and the eventual creator of the outer shell;
    its default implementation forwards its arguments to
    ``torch.Tensor._make_wrapper_subclass``. Both ``__new__`` and ``_clone_outer_tensor``
    delegate to it, and the ``_to_copy.default`` handler uses it directly. Overriding
    ``_make_outer_tensor`` changes shell construction for new construction, cloning, and
    ``_to_copy``.

    ``_clone_outer_tensor`` is an instance method that reads observable outer metadata from
    ``self`` and delegates to ``type(self)._make_outer_tensor``. The
    ``aten.clone.default`` handler uses ``_clone_outer_tensor`` to create the clone's
    outer shell, forwards ``aten.clone.default`` to the inner tensors, and initializes the
    Python fields.

    The default ``_clone_outer_tensor`` copies only observable metadata and does not carry
    ``storage_offset`` or ``storage_size``, matching plain tensor clone behavior. It also
    cannot recover ``dispatch_sizes_strides_policy``, ``dispatch_device``,
    ``dispatch_layout``, or ``_extra_dispatch_keys`` because PyTorch exposes no public
    getters for those shell settings. A subclass that uses them must override
    ``_clone_outer_tensor`` or retain the configuration itself.

    The default ``__new__`` requires both geometry arguments, so a subclass that supports
    direct construction without them must override ``__new__``. A subclass with field
    names that collide with the recognized shell keywords must also override
    ``__new__``. To customize shell creation, override ``_make_outer_tensor``; to
    customize only clone-time shell reconstruction, override ``_clone_outer_tensor``.

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

    The base does not prevent a subclass from giving the outer wrapper its own storage.
    However, the handlers registered on ``MokitWrapperTensorBase`` assume a storage-free
    outer wrapper; subclasses that attach outer storage must override the affected
    handlers.

    ``__tensor_flatten__`` and ``__tensor_unflatten__`` are PyTorch's PT2 tracing protocol.
    They do not implement pickle support. Pickle uses Tensor's default wrapper-subclass
    path, which supports storage-free strided wrappers. Non-strided wrappers reach Tensor's
    layout-specific reduction branches and fail explicitly; subclasses that need pickle
    support for such layouts must override ``__reduce_ex__``. A concrete subclass must be
    registered explicitly with ``mokit.serialization.torch_add_safe_global`` for
    ``weights_only=True`` loading.

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
    ``__torch_dispatch__`` or ``__torch_function__`` signature. A handler may accept the
    packed ``(cls, func, types, args, kwargs)`` arguments directly, or use the optional
    ``unpack_args`` helper to receive the operator's arguments unpacked. For example, the
    default ``aten.clone.default`` handler uses ``unpack_args`` and declares
    ``self: WrapperType``, ``memory_format: torch.memory_format | None``, and a
    ``WrapperType`` return value. Registration accepts ``Callable[..., Any]`` and does not
    enforce these annotations, but accurate annotations document each operator's contract.

    An ATen handler registered with ``implements`` cannot itself replace PyTorch's
    backward. ``__torch_dispatch__`` runs below autograd, so the generated backward for
    the operator may already be attached to the outer wrapper before the handler runs.
    A subclass can provide custom differentiability in one of three ways:

    1. Use the native outer-wrapper backward. Register only the forward handler and rely
       on PyTorch's generated autograd formula. This is appropriate only when that formula
       is compatible with the wrapper representation.

    2. Define the autograd boundary at the public torch-function level. Register a
       ``__torch_function__`` handler for the public function and call a
       ``torch.autograd.Function`` or a custom operator with
       ``torch.library.register_autograd``. The outer wrapper is then the autograd node,
       and the custom ``backward`` is used directly. No reentrant dispatch is needed.

    3. Use the inner-autograd pattern: keep the outer wrapper non-grad, call a custom
       ``torch.autograd.Function`` or custom operator on the inner tensors inside
       ``torch.overrides.enable_reentrant_dispatch()``, and wrap the result in a fresh
       wrapper. See https://github.com/albanD/subclass_zoo/blob/main/inner_autograd_tensor.py.

    """

    positional_inner_tensor_names: ClassVar[tuple[str, ...]] = ()
    positional_metadata_names: ClassVar[tuple[str, ...]] = ()
    keyword_inner_tensor_names: ClassVar[tuple[str, ...]] = ()
    keyword_metadata_names: ClassVar[tuple[str, ...]] = ()

    _FLATTEN_ATTR_NAMES: ClassVar[tuple[str, ...]] = (
        "positional_inner_tensor_names",
        "positional_metadata_names",
        "keyword_inner_tensor_names",
        "keyword_metadata_names",
    )

    _RESERVED_FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "outer_size",
        "outer_stride",
        "metadata",
        "positional_metadata",
        "keyword_metadata",
        "inner_tensor_names",
        "metadata_names",
    )

    _TORCH_FN_TABLE: ClassVar[dict[Callable[..., Any] | None, TorchFunctionImpl]] = {}
    _ATEN_OP_TABLE: ClassVar[dict[OpOverload | None, TorchDispatchImpl]] = {}

    def __new__(
        cls,
        *args: Any,
        outer_size: Sequence[int | torch.SymInt],
        outer_stride: Sequence[int | torch.SymInt] | None,
        **kwargs: Any,
    ) -> Self:
        """
        Create the outer wrapper shell.

        Subclasses that need different shell metadata, or whose field names collide with
        the recognized shell keywords, must override this method. They may reuse it with
        ``super().__new__(cls, *args, outer_size=..., outer_stride=..., **kwargs)``.

        Args:
            *args: Positional constructor field values, accepted for symmetry with
                ``__init__`` and not inspected.
            outer_size: Outer size for the wrapper shell.
            outer_stride: Outer stride, or ``None`` for a contiguous stride.
            **kwargs: Constructor keyword values. Recognized shell keys are ``dtype``,
                ``layout``, ``device``, ``pin_memory``, ``requires_grad``,
                ``storage_offset``, ``memory_format``, ``storage_size``,
                ``dispatch_sizes_strides_policy``, ``dispatch_device``,
                ``dispatch_layout``, and ``_extra_dispatch_keys``.

        Returns:
            The outer wrapper shell.

        Raises:
            TypeError: If ``outer_size`` is not a valid size.

        """
        return cls._make_outer_tensor(
            size=outer_size,
            strides=outer_stride,
            storage_offset=kwargs.get("storage_offset"),
            memory_format=kwargs.get("memory_format"),
            dtype=kwargs.get("dtype"),
            layout=kwargs.get("layout", torch.strided),
            device=kwargs.get("device"),
            pin_memory=kwargs.get("pin_memory", False),
            requires_grad=kwargs.get("requires_grad", False),
            dispatch_sizes_strides_policy=kwargs.get("dispatch_sizes_strides_policy"),
            dispatch_device=kwargs.get("dispatch_device", False),
            dispatch_layout=kwargs.get("dispatch_layout", False),
            _extra_dispatch_keys=kwargs.get("_extra_dispatch_keys"),
            storage_size=kwargs.get("storage_size"),
        )

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

        for attr_name in cls._FLATTEN_ATTR_NAMES:
            attr = getattr(cls, attr_name)
            if not isinstance(attr, tuple):
                raise TypeError(f"The class attribute '{attr_name}' must be a tuple.")
            if not all(isinstance(name, str) for name in attr):
                raise TypeError(f"The class attribute '{attr_name}' must be a tuple of strings.")

        declared_field_counter = Counter(
            field for attr_name in cls._FLATTEN_ATTR_NAMES for field in getattr(cls, attr_name)
        )

        duplicated_fields = tuple(field for field, freq in declared_field_counter.items() if freq > 1)
        if duplicated_fields:
            raise ValueError(f"Field name(s) {sorted(duplicated_fields)!r} are declared more than once.")

        conflict_fields = tuple(field for field in cls._RESERVED_FIELD_NAMES if field in declared_field_counter)
        if conflict_fields:
            raise ValueError(f"Reserved field name(s) {sorted(conflict_fields)!r} cannot be used.")

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
            for attr_name in type(self)._FLATTEN_ATTR_NAMES
            for name in getattr(self, attr_name)
            if name not in printed_names
        )

        return f"{type(self).__name__}({', '.join(fields)})"

    def __tensor_flatten__(self) -> tuple[Sequence[str], Mapping[str, Any]]:
        """
        Flatten the wrapper into inner tensor names and metadata.

        Returns:
            The names of the inner tensors and the metadata mapping.

        """
        cls = type(self)
        tensor_names = cls.inner_tensor_names()
        return tensor_names, self.metadata

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
        (
            inner_tensor_values,
            metadata_values,
            keyword_inner_tensor_values,
            keyword_metadata_values,
        ) = cls._constructor_field_groups(inner_tensors, metadata)

        constructor = cast(Callable[..., Self], cls)
        return constructor(
            *inner_tensor_values,
            *metadata_values,
            outer_size=outer_size,
            outer_stride=outer_stride,
            **keyword_inner_tensor_values,
            **keyword_metadata_values,
        )

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

        Registered handler results are post-processed by
        ``return_and_correct_aliasing`` so the outer tensor objects follow the aliasing
        declared by the operator schema:

        * Non-inplace view operators: link the result's outer storage to the aliased
          input's storage. The result's size, stride, storage offset, inner tensors, and
          metadata are left unchanged. ``MokitWrapperTensorBase`` and its subclasses have
          no outer storage allocation, so this correction only establishes the outer
          TensorImpl alias relationship.
        * Inplace-view operators: replay the metadata-only operator on the original input
          wrapper, then return that wrapper instead of the handler result.
        * In-place and ``out=`` operators: return the mutated input argument instead of
          the handler result, preserving object identity.

        The correction does not establish aliasing between inner tensors or create view
        autograd metadata; handlers remain responsible for forwarding the operator to
        their fields.

        Args:
            func: The ATen operator being called.
            types: Types of the participating arguments.
            args: Positional arguments.
            kwargs: Keyword arguments, or ``None``.

        Returns:
            The registered handler result after schema-declared aliasing correction.

        Raises:
            NotImplementedError: If no handler is registered for the operator.

        """
        found, result = _call_handler_from_table(
            MokitWrapperTensorBase, "_ATEN_OP_TABLE", cls, func, types, args, kwargs
        )
        if found:
            return return_and_correct_aliasing(func, args, kwargs, result)

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

    @classmethod
    def inner_tensor_names(cls: type[Self]) -> tuple[str, ...]:
        """
        Return the inner tensor field names in authoritative flatten order.

        The order is ``positional_inner_tensor_names`` followed by
        ``keyword_inner_tensor_names``. Flattening, field aliasing checks, and constructor
        argument assembly use this order.

        Returns:
            The flattened inner tensor field names.

        """
        return (*cls.positional_inner_tensor_names, *cls.keyword_inner_tensor_names)

    @classmethod
    def metadata_names(cls: type[Self]) -> tuple[str, ...]:
        """
        Return the metadata field names in authoritative flatten order.

        The order is ``positional_metadata_names`` followed by
        ``keyword_metadata_names``. Flattening and constructor argument assembly use this
        order.

        Returns:
            The flattened metadata field names.

        """
        return (*cls.positional_metadata_names, *cls.keyword_metadata_names)

    @property
    def metadata(self: Self) -> dict[str, Any]:
        """
        Return the declared metadata values keyed by field name.

        Returns:
            A new dict containing the declared metadata values by reference.

        """
        return {name: getattr(self, name) for name in type(self).metadata_names()}

    @property
    def positional_metadata(self: Self) -> tuple[Any, ...]:
        """
        Return positional metadata values in declaration order.

        Returns:
            The positional metadata values by reference.

        """
        return tuple(getattr(self, name) for name in type(self).positional_metadata_names)

    @property
    def keyword_metadata(self: Self) -> dict[str, Any]:
        """
        Return keyword metadata values keyed by field name.

        Returns:
            A new dict containing the keyword metadata values by reference.

        """
        return {name: getattr(self, name) for name in type(self).keyword_metadata_names}

    def has_same_attributes(
        self: Self,
        other: Self,
        *,
        strict: bool = True,
        raise_on_mismatch: bool = False,
    ) -> bool:
        """
        Compare declared metadata and, optionally, native tensor attributes.

        Args:
            other: The wrapper to compare against.
            strict: Whether to compare native dtype, device, layout, shape, and stride
                in addition to declared metadata.
            raise_on_mismatch: Whether to raise at the first mismatch instead of returning
                False.

        Returns:
            Whether the selected attributes and metadata match.

        Raises:
            ValueError: If the wrappers have different runtime classes, if metadata
                equality fails, or if ``raise_on_mismatch`` is true and a value differs.

        """
        if type(self) is not type(other):
            raise ValueError(f"type mismatch: self={type(self).__name__}, other={type(other).__name__}")

        def mismatch(
            field: str,
            self_value: object,
            other_value: object,
            *,
            kind: str | None = None,
        ) -> bool:
            """Raise or return False for the current mismatch."""
            if raise_on_mismatch:
                label = f"{kind} {field!r}" if kind is not None else field
                raise ValueError(f"{label} mismatch: self={self_value!r}, other={other_value!r}")
            return False

        if strict:
            for attr_name in ("dtype", "device", "layout", "shape"):
                self_value = getattr(self, attr_name)
                other_value = getattr(other, attr_name)

                if self_value != other_value:
                    return mismatch(attr_name, self_value, other_value)

            self_stride = self.stride()
            other_stride = other.stride()

            if self_stride != other_stride:
                return mismatch("stride", self_stride, other_stride)

        for name in type(self).metadata_names():
            self_value = getattr(self, name)
            other_value = getattr(other, name)

            if self_value is other_value:
                continue

            try:
                equal = bool(self_value == other_value)
            except Exception as exc:
                raise ValueError(f"metadata {name!r} is not equality-comparable") from exc

            if not equal:
                return mismatch(name, self_value, other_value, kind="metadata")

        return True

    def tensor_field_aliasing_groups(self) -> tuple[tuple[str, ...], ...]:
        """
        Group inner tensor field names by exact object identity.

        Returns:
            Field-name groups ordered by first occurrence. Names within each group follow
            the authoritative flattened field order.

        """
        groups: dict[int, list[str]] = {}

        for name in type(self).inner_tensor_names():
            # Python guarantees dict insertion order:
            # https://docs.python.org/3/library/stdtypes.html#typesmapping
            # so groups are ordered by the first field that references each tensor object.
            groups.setdefault(id(getattr(self, name)), []).append(name)

        return tuple(tuple(names) for names in groups.values())

    def has_same_tensor_field_aliasing(
        self: Self,
        field_aliasing_group: tuple[tuple[str, ...], ...],
        *,
        raise_on_mismatch: bool = False,
    ) -> bool:
        """
        Compare exact-object aliasing against the given field aliasing groups.

        Field names are grouped by the identity of the tensor they reference. The groups
        match only when corresponding fields produce identical first-occurrence groups.
        Storage-level view aliasing is not examined.

        Args:
            field_aliasing_group: The field aliasing groups to compare against.
            raise_on_mismatch: Whether to raise at the first mismatch instead of returning
                False.

        Returns:
            Whether this wrapper groups its inner tensor fields identically.

        Raises:
            ValueError: If ``raise_on_mismatch`` is true and a field group differs.

        """
        self_group = self.tensor_field_aliasing_groups()

        if self_group == field_aliasing_group:
            return True

        if not raise_on_mismatch:
            return False

        raise ValueError(f"field aliasing differs: self={self_group!r}, given={field_aliasing_group!r}")

    def outer_stride(
        self: Self,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> Sequence[int | torch.SymInt]:
        """
        Return the outer stride for a memory format.

        ``None`` and ``torch.preserve_format`` preserve the current outer stride. An
        explicit format is probed on a meta tensor using the wrapper's layout.

        Args:
            memory_format: The requested memory format, if any.

        Returns:
            The outer stride to use for the converted wrapper.

        Raises:
            RuntimeError: If ``torch.empty`` rejects the requested layout or memory format.

        """
        if memory_format in (None, torch.preserve_format):
            return self.stride()

        return torch.empty(
            self.size(),
            device="meta",
            layout=self.layout,
            memory_format=memory_format,
        ).stride()

    @classmethod
    def _make_outer_tensor(cls, *args: Any, **kwargs: Any) -> Self:
        """
        Construct an outer wrapper shell.

        The default implementation forwards positional and keyword arguments to
        ``torch.Tensor._make_wrapper_subclass``. Subclasses may override this method to
        provide their own outer wrapper creation logic.

        Args:
            *args: Positional arguments for the wrapper factory.
            **kwargs: Keyword arguments for the wrapper factory.

        Returns:
            The outer wrapper shell.

        Raises:
            TypeError: If the factory rejects the provided arguments.

        """
        return torch.Tensor._make_wrapper_subclass(cls, *args, **kwargs)

    @classmethod
    def _constructor_field_groups(
        cls,
        inner_tensors: Mapping[str, torch.Tensor | CustomClassBase],
        metadata: Mapping[str, Any],
    ) -> tuple[
        tuple[torch.Tensor | CustomClassBase, ...],
        tuple[Any, ...],
        dict[str, torch.Tensor | CustomClassBase],
        dict[str, Any],
    ]:
        """
        Collect the four constructor field groups from flattened data.

        Args:
            inner_tensors: The inner tensors keyed by their flattened names.
            metadata: The metadata mapping from ``__tensor_flatten__``.

        Returns:
            Positional inner tensors, positional metadata, keyword inner tensors, and
            keyword metadata, in constructor-protocol order.

        """
        return (
            tuple(inner_tensors[name] for name in cls.positional_inner_tensor_names),
            tuple(metadata[name] for name in cls.positional_metadata_names),
            {name: inner_tensors[name] for name in cls.keyword_inner_tensor_names},
            {name: metadata[name] for name in cls.keyword_metadata_names},
        )

    def _apply_fn_to_outer_tensor(
        self: Self,
        func: OpOverload | Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """
        Apply a metadata-only operator to the outer tensor shell.

        The operator runs with Python dispatch disabled, so it receives the wrapper as a
        storage-free Tensor instead of re-entering this dispatcher. The result is retyped
        to the wrapper class. Only view and metadata operators are supported;
        data-dependent operators must not use this method because the outer shell has no
        storage.

        Args:
            func: The metadata-only operator to apply.
            *args: Positional operator arguments.
            **kwargs: Keyword operator arguments.

        Returns:
            The wrapper-class shell produced by the operator.

        Raises:
            RuntimeError: If the operator rejects a storage-free Tensor.

        """
        cls = type(self)
        with torch._C._DisableTorchDispatch():
            result = func(self, *args, **kwargs)
            if type(result) is not cls:
                result = result.as_subclass(cls)

        return result

    def _clone_outer_tensor(
        self: Self,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> Self:
        """
        Create a fresh outer shell for a clone.

        This copies observable outer metadata only. It does not copy storage offset or
        storage size, matching plain tensor clone behavior. Advanced dispatch
        configuration is subclass-specific and is not recoverable from the shell.

        Args:
            memory_format: The requested memory format, if any.

        Returns:
            A fresh outer wrapper shell without initialized Python fields.

        Raises:
            RuntimeError: If the memory format is incompatible with the outer size.

        """
        return type(self)._make_outer_tensor(
            size=self.size(),
            strides=self.outer_stride(memory_format=memory_format),
            memory_format=memory_format,
            dtype=self.dtype,
            layout=self.layout,
            device=self.device,
            requires_grad=self.requires_grad,
        )


implements = MokitWrapperTensorBase.implements


@implements(torch.ops.aten.clone.default)
@unpack_args
def _[WrapperType: MokitWrapperTensorBase](
    cls: type[WrapperType],
    func: OpOverload,
    types: tuple[type[WrapperType]],
    self: WrapperType,
    *,
    memory_format: torch.memory_format | None = None,
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
    explicit memory format it is recomputed from the requested format. The outer layout
    and other observable shell metadata are preserved by ``_clone_outer_tensor``. Storage
    offset and storage size are reset as for a plain tensor clone.

    Args:
        cls: Runtime wrapper class used to construct the clone.
        func: The ``aten.clone.default`` operator.
        types: Participating dispatch types.
        self: The wrapper to clone.
        memory_format: The requested memory format, if any.

    Returns:
        A new wrapper of the same runtime class.

    Raises:
        AssertionError: If assertions are enabled and ``cls`` does not match the runtime
            class of ``self``.

    """
    assert type(self) is cls, (
        f"{func}: dispatch cls={cls.__name__} does not match type(self)={type(self).__name__}; types={types}"
    )

    groups = self.tensor_field_aliasing_groups()
    cloned_inner_tensor_values = apply_fn(
        [getattr(self, group[0]) for group in groups],
        func,
        memory_format=memory_format,
    )

    cloned_inner_tensors: dict[str, torch.Tensor] = {}
    for group, cloned in zip(groups, cloned_inner_tensor_values, strict=True):
        for name in group:
            cloned_inner_tensors[name] = cloned

    outer_tensor = self._clone_outer_tensor(memory_format=memory_format)

    initializer = cast(Callable[..., None], outer_tensor.__init__)
    initializer(
        *(cloned_inner_tensors[name] for name in cls.positional_inner_tensor_names),
        *self.positional_metadata,
        outer_size=outer_tensor.size(),
        outer_stride=outer_tensor.stride(),
        **{name: cloned_inner_tensors[name] for name in cls.keyword_inner_tensor_names},
        **self.keyword_metadata,
    )

    return outer_tensor


@implements(torch.ops.aten.clone.out)
@unpack_args
def _[WrapperType: MokitWrapperTensorBase](
    cls: type[WrapperType],
    func: OpOverload,
    types: tuple[type[WrapperType]],
    self: WrapperType,
    *,
    memory_format: torch.memory_format | None = None,
    out: WrapperType,
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
        self: The source wrapper.
        memory_format: The requested memory format, if any.
        out: The destination wrapper.

    Returns:
        The ``out`` wrapper after its inner tensors have been updated.

    Raises:
        AssertionError: If assertions are enabled and either ``self`` or ``out`` does
            not match the dispatch class.
        ValueError: If outer dtype, device, layout, shape, or stride differs; if declared
            metadata differs or is not equality-comparable; or if exact-object inner
            alias topology differs.

    """
    assert type(self) is cls, (
        f"{func}: dispatch cls={cls.__name__} does not match type(self)={type(self).__name__}; types={types}"
    )
    assert type(out) is cls, f"{func}: out type={type(out).__name__} does not match dispatch cls={cls.__name__}"

    self.has_same_attributes(out, strict=True, raise_on_mismatch=True)

    field_aliasing_group = self.tensor_field_aliasing_groups()
    out.has_same_tensor_field_aliasing(
        field_aliasing_group,
        raise_on_mismatch=True,
    )

    apply_fn(
        [getattr(self, group[0]) for group in field_aliasing_group],
        lambda source, *, dest: func(source, memory_format=memory_format, out=dest),
        dest=[getattr(out, group[0]) for group in field_aliasing_group],
    )

    return out


@implements(torch.ops.aten.alias.default, torch.ops.aten.detach.default)
@unpack_args
def _[WrapperType: MokitWrapperTensorBase](
    cls: type[WrapperType],
    func: OpOverload,
    types: tuple[type[WrapperType]],
    self: WrapperType,
) -> WrapperType:
    """
    Create an alias or detach result for a wrapper tensor.

    The outer wrapper shell is re-created by forwarding ``func`` to the shell as a
    storage-free Tensor. The declared inner tensors are forwarded to ``func`` once per
    object identity. Inner dispatch follows PyTorch's default behavior for operations
    invoked inside ``__torch_dispatch__``: plain tensors share storage without
    necessarily gaining view, version, or autograd metadata, while custom tensor
    subclasses dispatch to their own handlers.

    The shell is created with ``as_subclass`` while Python dispatch is disabled, which
    bypasses ``__new__`` and ``__tensor_unflatten__``. The subclass ``__init__`` is then
    called with the transformed inner tensors and the original metadata references,
    allowing each wrapper class to finish its own Python initialization.

    Args:
        cls: Runtime wrapper class returned by the operation.
        func: The ``aten.alias.default`` or ``aten.detach.default`` operator.
        types: Participating dispatch types.
        self: The wrapper receiving the operation.

    Returns:
        A new wrapper produced by forwarding ``func`` to the outer shell and inner
        tensors, with declared metadata passed by reference.

    Raises:
        AssertionError: If assertions are enabled and ``cls`` does not match the runtime
            class of ``self``.

    """
    assert type(self) is cls, (
        f"{func}: dispatch cls={cls.__name__} does not match type(self)={type(self).__name__}; types={types}"
    )

    aliased = self._apply_fn_to_outer_tensor(func)

    groups = self.tensor_field_aliasing_groups()
    aliased_inner_values = apply_fn(
        [getattr(self, group[0]) for group in groups],
        func,
    )

    aliased_inner_tensors: dict[str, torch.Tensor] = {}
    for group, aliased_value in zip(groups, aliased_inner_values, strict=True):
        for name in group:
            aliased_inner_tensors[name] = aliased_value

    initializer = cast(Callable[..., None], aliased.__init__)
    initializer(
        *(aliased_inner_tensors[name] for name in cls.positional_inner_tensor_names),
        *self.positional_metadata,
        outer_size=aliased.size(),
        outer_stride=aliased.stride(),
        **{name: aliased_inner_tensors[name] for name in cls.keyword_inner_tensor_names},
        **self.keyword_metadata,
    )

    return aliased


@implements(torch.ops.aten.copy_.default)
@unpack_args
def _[WrapperType: MokitWrapperTensorBase](
    cls: type[WrapperType],
    func: OpOverload,
    types: tuple[type[WrapperType]],
    self: WrapperType,
    src: WrapperType,
    non_blocking: bool = False,
) -> WrapperType:
    """
    Copy a source wrapper into an existing wrapper in place.

    Both wrappers must have the same runtime class. Outer dtype, device, layout,
    shape, stride, declared metadata, and inner tensor field aliasing must match.
    Inner tensors are copied in place through ``aten.copy_`` once per alias group.

    The scalar-fill overloads ``copy_.int`` and ``copy_.float`` are not handled;
    filling a wrapper representation with one scalar is subclass-specific.

    Args:
        cls: Runtime wrapper class shared by the source and destination.
        func: The ``aten.copy_.default`` operator.
        types: Participating dispatch types.
        self: The destination wrapper.
        src: The source wrapper.
        non_blocking: Whether to forward the asynchronous-copy hint.

    Returns:
        The destination wrapper, ``self``.

    Raises:
        AssertionError: If assertions are enabled and ``cls`` does not match the runtime
            class of ``self``.
        ValueError: If the wrappers have different runtime classes, outer attributes,
            declared metadata, or inner tensor field aliasing.

    """
    assert type(self) is cls, (
        f"{func}: dispatch cls={cls.__name__} does not match type(self)={type(self).__name__}; types={types}"
    )

    self.has_same_attributes(src, strict=True, raise_on_mismatch=True)

    field_aliasing_group = src.tensor_field_aliasing_groups()
    self.has_same_tensor_field_aliasing(
        field_aliasing_group,
        raise_on_mismatch=True,
    )

    apply_fn(
        [getattr(src, group[0]) for group in field_aliasing_group],
        lambda source, *, dest: func(dest, source, non_blocking=non_blocking),
        dest=[getattr(self, group[0]) for group in field_aliasing_group],
    )

    return self


@implements(torch.ops.aten._to_copy.default)
@unpack_args
def _[WrapperType: MokitWrapperTensorBase](
    cls: type[WrapperType],
    func: OpOverload,
    types: tuple[type[WrapperType]],
    self: WrapperType,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> WrapperType:
    """
    Copy a wrapper tensor with device, pinning, and memory-format conversion.

    Supported conversions are ``device``, ``pin_memory``, ``non_blocking``, and
    ``memory_format``. ``dtype`` and ``layout`` are not supported by the base handler and
    must be handled by a subclass. Inner tensors are copied once per alias group, and
    declared metadata is passed through by reference.

    Memory-format handling matches ``clone.default``: ``None`` and
    ``torch.preserve_format`` preserve the source outer stride, while an explicit format
    derives the outer stride from that format. The format is also forwarded to each inner
    ``_to_copy`` call.

    A subclass can selectively reuse this handler while implementing one of the supported
    conversions itself: call the base handler with the arguments it should process and
    omit the argument the subclass has already handled. The base handler only applies the
    values it receives.

    The new outer shell is created by ``_make_outer_tensor`` with the target device,
    outer size and stride, dtype, layout, and requires-grad state. The Python fields are
    then initialized directly on that shell. Storage offset and storage size follow plain
    copy semantics, and non-strided layouts are preserved. The base assumes all inner
    tensors share the outer device; heterogeneous wrappers must override the handler.
    Explicit memory-format conversion assumes a strided wrapper and compatible inner
    ranks, and unsupported native conversions propagate their errors.

    Args:
        cls: Runtime wrapper class used to construct the copy.
        func: The ``aten._to_copy.default`` operator.
        types: Participating dispatch types.
        self: The wrapper to copy.
        dtype: Unsupported by the base handler; must be handled by a subclass.
        layout: Unsupported by the base handler; must be handled by a subclass.
        device: The target device, if conversion is requested.
        pin_memory: The pin-memory hint forwarded to the inner copies.
        non_blocking: The asynchronous-copy hint forwarded to the inner copies.
        memory_format: The requested memory format, if any.

    Returns:
        A new wrapper of the same runtime class.

    Raises:
        AssertionError: If assertions are enabled and ``cls`` does not match the runtime
            class of ``self``.
        NotImplementedError: If ``dtype`` or ``layout`` is not ``None``.

    """
    assert type(self) is cls, (
        f"{func}: dispatch cls={cls.__name__} does not match type(self)={type(self).__name__}; types={types}"
    )

    requested_conversions = tuple(
        name
        for name, value in (
            ("dtype", dtype),
            ("layout", layout),
        )
        if value is not None
    )
    if requested_conversions:
        raise NotImplementedError(
            f"{func}: base handler supports only device, pin_memory, non_blocking, "
            f"and memory_format; handle and remove {requested_conversions} before delegating"
        )

    field_aliasing_group = self.tensor_field_aliasing_groups()
    new_unique_tensors = apply_fn(
        (getattr(self, group[0]) for group in field_aliasing_group),
        lambda src: func(
            src,
            device=device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=memory_format,
        ),
    )

    new_inner_tensors = {
        name: new_unique_tensors[group_id] for group_id, group in enumerate(field_aliasing_group) for name in group
    }

    target_device = self.device if device is None else device
    outer_tensor = cls._make_outer_tensor(
        size=self.size(),
        strides=self.outer_stride(memory_format=memory_format),
        memory_format=memory_format,
        dtype=self.dtype,
        layout=self.layout,
        device=target_device,
        requires_grad=self.requires_grad,
    )

    initializer = cast(Callable[..., None], outer_tensor.__init__)
    initializer(
        *(new_inner_tensors[name] for name in cls.positional_inner_tensor_names),
        *self.positional_metadata,
        outer_size=outer_tensor.size(),
        outer_stride=outer_tensor.stride(),
        **{name: new_inner_tensors[name] for name in cls.keyword_inner_tensor_names},
        **self.keyword_metadata,
    )

    return outer_tensor
