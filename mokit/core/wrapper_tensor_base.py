# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper tensor base class and its dispatch helpers."""

import functools
import itertools
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, ClassVar, Final, Literal, Protocol, Self, cast, overload

import torch
from torch._custom_class_base import CustomClassBase
from torch._ops import OpOverload
from torch.utils._python_dispatch import return_and_correct_aliasing

from mokit.core import MokitTensorBase
from mokit.utils import apply_fn_by_id


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
        """Invoke the handler with unpacked arguments."""


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
        """Invoke the handler with packed arguments."""


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
        A handler accepting ``(cls, func, types, args, kwargs)``. ``functools.wraps``
        preserves the wrapped handler's name, docstring, and annotations, and the result
        type is preserved as ``ReturnType``.

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
def _register_handlers(
    cls: type[MokitWrapperTensorBase],
    table_name: Literal["_TORCH_FN_TABLE"],
    dispatcher_name: Literal["__torch_function__"],
    keys: tuple[Callable[..., Any] | None, ...],
    func: Callable[..., Any],
    override: bool,
    make_method: Callable[[type[MokitWrapperTensorBase]], TorchFunctionImpl],
) -> None: ...


@overload
def _register_handlers(
    cls: type[MokitWrapperTensorBase],
    table_name: Literal["_ATEN_OP_TABLE"],
    dispatcher_name: Literal["__torch_dispatch__"],
    keys: tuple[OpOverload | None, ...],
    func: Callable[..., Any],
    override: bool,
    make_method: Callable[[type[MokitWrapperTensorBase]], TorchDispatchImpl],
) -> None: ...


def _register_handlers(
    cls: type[MokitWrapperTensorBase],
    table_name: Literal["_TORCH_FN_TABLE", "_ATEN_OP_TABLE"],
    dispatcher_name: Literal["__torch_function__", "__torch_dispatch__"],
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

    Validation failures - a non-callable handler, inconsistent dispatcher/table state,
    a non-dict table, or key conflicts - leave the class unchanged. Failures while
    installing the generated dispatcher or writing the table are fatal.

    Args:
        cls: Class receiving the handlers.
        table_name: Handler table attribute name.
        dispatcher_name: Attribute name of the dispatcher.
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

    has_dispatcher = dispatcher_name in cls.__dict__
    table = cls.__dict__.get(table_name)

    if has_dispatcher and table is None:
        raise KeyError(f"{cls.__name__} defines its own {dispatcher_name} but does not define its own {table_name}.")

    if table is not None:
        if type(table) is not dict:
            raise TypeError(f"{table_name} must be a dict.")

        conflicts = tuple(key for key in keys if key in table)
        if conflicts and not override:
            raise KeyError(f"Handlers already registered for {conflicts!r}.")

    if not has_dispatcher:
        setattr(cls, dispatcher_name, classmethod(make_method(cls)))

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
        """
        Dispatch a torch function through the handler table of ``owner_cls``.

        Args:
            cls: Runtime wrapper class.
            func: The torch function being called.
            types: Types of the participating arguments.
            args: Positional arguments.
            kwargs: Keyword arguments, or ``None``.

        Returns:
            The handler result, or the result after delegating to the parent dispatcher.

        """
        # The generated dispatcher is installed only on classes that own their table.
        table = owner_cls._TORCH_FN_TABLE
        handler = table.get(func)
        if handler is None:
            handler = table.get(None)
        if handler is not None:
            return handler(cls, func, types, args, kwargs)

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
        # The generated dispatcher is installed only on classes that own their table.
        table = owner_cls._ATEN_OP_TABLE
        handler = table.get(func)
        if handler is None:
            handler = table.get(None)
        if handler is not None:
            return return_and_correct_aliasing(func, args, kwargs, handler(cls, func, types, args, kwargs))

        parent = super(owner_cls, cls)  # pyrefly: ignore[invalid-argument]
        return parent.__torch_dispatch__(func, types, args, kwargs)

    return __torch_dispatch__


_MAKE_WRAPPER_SUBCLASS_KEYWORD_NAMES: Final[tuple[str, ...]] = (
    "storage_offset",
    "memory_format",
    "dtype",
    "layout",
    "device",
    "pin_memory",
    "requires_grad",
    "dispatch_sizes_strides_policy",
    "dispatch_device",
    "dispatch_layout",
    "_extra_dispatch_keys",
    "storage_size",
)


class MokitWrapperTensorBase(MokitTensorBase):
    """
    The base class for MOKIT wrapper tensor subclasses.

    **Inner / Outer Tensors and Metadata**

    A wrapper tensor is a ``torch.Tensor`` (usually storage-free) whose data and metadata are
    exposed through Python attributes. The outer representative ``torch.Tensor`` is referred
    to as the outer tensor or the shell tensor. Fields are split into inner tensors and
    metadata.

    Inner tensors store real data and participate in PyTorch's tracing protocol; metadata values
    are opaque to PyTorch. Each kind is also split into positional and keyword constructor groups.
    Their names are held in four class attributes:

    Class Attributes:
        positional_inner_tensor_names: Names of inner tensors passed positionally to
            ``__new__`` and ``__init__``, in order.
        positional_metadata_names: Names of metadata values passed positionally to
            ``__new__`` and ``__init__``, in order.
        keyword_inner_tensor_names: Names of inner tensors passed as keyword arguments to
            ``__new__`` and ``__init__``.
        keyword_metadata_names: Names of metadata values passed as keyword arguments to
            ``__new__`` and ``__init__``.

    All four default to ``()``. Every listed name must exist on the instance; the base does
    not validate field values, so a field may be ``None``. PyTorch's tracing protocol
    additionally requires each inner-tensor name to resolve to a tensor or a registered
    opaque type. Each attribute must be a tuple of strings; names must be unique across the
    four tuples and must not use a reserved name listed in ``_RESERVED_FIELD_NAMES``.
    ``__init_subclass__`` enforces these rules when each subclass is created.

    The following helpers expose these names and field values on an instance::

        def __repr__(self, *, tensor_contents: str | None = None) -> str: ...

        @classmethod
        def inner_tensor_names(cls: type[Self]) -> Iterator[str]: ...

        @classmethod
        def metadata_names(cls: type[Self]) -> Iterator[str]: ...

        @property
        def inner_tensors(self: Self) -> Iterator[torch.Tensor | CustomClassBase]: ...

        @property
        def positional_inner_tensors(self: Self) -> Iterator[torch.Tensor | CustomClassBase]: ...

        @property
        def keyword_inner_tensors(self: Self) -> Iterator[tuple[str, torch.Tensor | CustomClassBase]]: ...

        @property
        def metadata(self: Self) -> Iterator[Any]: ...

        @property
        def positional_metadata(self: Self) -> Iterator[Any]: ...

        @property
        def keyword_metadata(self: Self) -> Iterator[tuple[str, Any]]: ...


    For each kind, the canonical flatten order is the positional names followed by the
    keyword names: ``inner_tensor_names()`` chains ``positional_inner_tensor_names`` and
    ``keyword_inner_tensor_names``, while ``metadata_names()`` does the same for metadata.
    The separate ``_FLATTEN_ATTR_NAMES`` attribute does not affect flattening; it fixes the
    order in which the four tuples are validated and printed by ``__repr__``.

    The accessors listed above return a fresh iterator on every access and yield values by
    reference; ``__repr__`` consumes these values and returns a string. The class tuple is
    read when the accessor creates the iterator; each instance field is read with
    ``getattr`` as the iterator advances. Reassigning an instance field before its slot is
    consumed changes the yielded value; rebinding a class tuple after the iterator is created
    does not. ``_split_inner_tensor_values`` is a private helper for handlers: given processed
    values in canonical order, it returns the positional values and a name-keyed mapping
    of the keyword values.


    **Constructor**

    In ``MokitWrapperTensorBase`` and its subclasses, both ``__new__`` and ``__init__`` must
    accept the constructor call::

        cls(
            *inner_tensor_values,
            *metadata_values,
            outer_size=outer_size,
            outer_stride=outer_stride,
            **keyword_inner_tensor_values,
            **keyword_metadata_values,
        )

    The two ``*`` groups and two ``**`` groups map to the four class attribute name groups one
    by one. ``outer_size`` and ``outer_stride`` are reserved geometry keywords; direct
    construction may default them to ``None``. These six argument groups carry exactly the
    information that ``__tensor_unflatten__`` receives: the split of ``inner_tensors`` and
    ``metadata`` plus the two geometry values. The positional signature must therefore place
    all positional inner tensors, in ``positional_inner_tensor_names`` order, before all
    positional metadata, in ``positional_metadata_names`` order. By default, the outer wrapper
    is created in ``__new__`` through ``_make_outer_tensor``. It carries size, stride, dtype,
    device, layout, and requires-grad state but no storage. Inner tensors hold the data that
    operations are forwarded to, and metadata fields hold opaque Python state; the subclass's
    ``__init__`` initializes both. Python passes the same arguments to both ``__new__`` and
    ``__init__``. The base's methods and operation handlers assume this storage-free design.

    The base always reads declared field values through ``getattr(self, ...)`` and never
    assigns individual fields itself; it delegates field initialization to the subclass
    constructor. The ``self.field = value`` pattern is therefore not the only valid way to
    consume a declared field value: a field may be backed by any attribute that
    ``getattr(self, name)`` resolves. For example, a declared ``dtype`` field may be consumed
    by shell construction or other subclass logic without assigning ``self.dtype``; the helper
    methods still read it through ``getattr(self, "dtype")``.

    A subclass may need finer control over outer-shell creation. The following private
    helpers are provided for this purpose:

    - ``_make_outer_tensor`` creates a new outer tensor. Its default implementation forwards
      its arguments to ``torch.Tensor._make_wrapper_subclass``. The default ``__new__`` maps
      ``outer_size``/``outer_stride`` and recognized shell keywords to this helper.

    - ``_clone_outer_tensor`` rebuilds a shell for a clone from observable metadata. It
      resolves the requested memory format first, then calls ``_make_outer_tensor``
      internally.

    - ``_apply_fn_to_outer_tensor`` applies view ops to the outer tensor. Using it for other
      ops may fail because they may access storage that the outer tensor does not have.

    Shell construction has two further constraints. The default ``__new__`` maps
    ``outer_size``/``outer_stride`` to ``size``/``strides`` and recognizes the remaining
    keyword arguments of ``torch.Tensor._make_wrapper_subclass``: ``storage_offset``,
    ``memory_format``, ``dtype``, ``layout``, ``device``, ``pin_memory``, ``requires_grad``,
    ``dispatch_sizes_strides_policy``, ``dispatch_device``, ``dispatch_layout``,
    ``_extra_dispatch_keys``, and ``storage_size``. Only recognized keys supplied by the
    caller are forwarded to ``_make_outer_tensor``; omitted keys use the shell factory's
    defaults. Therefore,

    1. A field whose value is passed under one of these keyword names is consumed by
       ``__new__`` as shell metadata. Declaring one of these names is acceptable only when
       ``getattr(self, name)`` exposes the corresponding shell value (for example, ``dtype``
       or ``layout``); using the name to shadow or repurpose that value is not recommended.

    2. The default ``_clone_outer_tensor`` copies only observable metadata, so it cannot
       recover ``dispatch_sizes_strides_policy``, ``dispatch_device``, ``dispatch_layout``,
       or ``_extra_dispatch_keys`` in the new shell. A subclass that needs to keep these
       configurations must override ``_clone_outer_tensor`` or retain the configuration
       itself. The default also does not pass ``storage_offset`` or ``storage_size`` to the
       new shell: ``storage_offset`` is observable through ``Tensor.storage_offset()`` but
       is intentionally reset, and ``storage_size`` has no public getter; the reset matches
       plain tensor clone behavior.


    **PT2 Support**

    The base provides a default implementation for ``__tensor_flatten__`` and ``__tensor_unflatten__``
    required by ``torch.compile`` and ``torch.export``. It relies on the inner / outer tensors
    and metadata protocol introduced above.

    ``__tensor_flatten__`` returns the inner tensor names in ``inner_tensor_names()`` order and a flat
    tuple of metadata values in ``metadata_names()`` order. PyTorch treats the metadata as opaque; the
    flat representation is private to the flatten/unflatten pair and may change as long as the pair
    stays consistent.

    ``__tensor_unflatten__`` rebuilds the constructor call from the class attributes: the first
    ``len(positional_metadata_names)`` metadata values are positional, the remaining values
    are paired with ``keyword_metadata_names``, and inner tensors are looked up in the
    supplied mapping and split the same way. ``__tensor_unflatten__`` receives ``outer_size`` and
    ``outer_stride`` separately.

    PyTorch's AOT autograd cache can warn that a wrapper subclass does not implement
    ``_stable_hash_for_caching``. The warning is emitted when the cache pickler hashes a
    traceable wrapper subclass for a cache key and finds no hook. Without further context,
    the base class deliberately relies on PyTorch's inherited fallback: it hashes the outer
    shape and ``requires_grad``, the metadata returned by ``__tensor_flatten__`` after
    stabilizing tuples/lists/dicts and replacing opaque custom classes with their type names,
    and the metadata of each inner tensor. Subclasses can get rid of the warning by
    implementing their own ``_stable_hash_for_caching``. The fallback is best-effort: PyTorch
    cannot know whether ``__tensor_flatten__`` fully captures a subclass's semantics or
    whether its metadata pickles deterministically. DTensor is the standard example - it
    implements the hook to fold in its spec hash and local device
    (``torch/distributed/tensor/_api.py:465``), because its identity is not recoverable from
    inner-tensor metadata alone. The warning is a one-time nudge per process for subclasses
    in that situation.


    **Implementation, Registration, and Dispatch of ATen Ops and Torch Functions**

    ``MokitWrapperTensorBase`` provides a mechanism for adding customized implementations of
    ATen ops and torch functions, both at class definition time and after class creation.
    Handlers registered on a class are reachable from its subclasses through the dispatcher's
    MRO delegation. Registration uses two classmethod decorator factories, ``implements`` and
    ``implements_torch_function``; they work in parallel and share the same pattern.

    A customized implementation for an ATen op or a torch function is called a handler. For
    torch functions, registration involves three objects:

    - ``_TORCH_FN_TABLE`` maps torch functions to their handlers. A class that does not yet
      own one installs it when its first handler is registered. A ``None`` key is a wildcard
      match. Handler tables are read and written directly from ``cls.__dict__``, so they are
      effectively not inherited; inheritance of registered handlers is simulated by the
      lookup in ``__torch_function__`` described below.

    - ``__torch_function__`` is generated from a template and installed lazily by the first
      registration on the class. At runtime it looks up the given torch function in its
      table; if found, it runs the handler; otherwise it delegates to the next
      ``__torch_function__`` in the MRO.

    - ``implements_torch_function`` accepts torch functions and returns a decorator for the
      handler. With no torch function it returns a decorator that registers the handler under
      the ``None`` key. All validation happens before the class is mutated, so invalid keys
      and conflicts leave the class unchanged; a failure while installing the generated
      method or writing the table is fatal. On successful registration, the decorator writes
      to the class's own ``_TORCH_FN_TABLE``.


    The ATen side has the same three counterparts, which cooperate in the same way:

    - ``_ATEN_OP_TABLE``
    - ``__torch_dispatch__``
    - ``implements``


    Both dispatchers forward their arguments to handlers unchanged. The ``unpack_args``
    decorator adapts a handler that wants the forwarded ``*args`` and ``**kwargs`` unpacked.
    As a convention, handlers should annotate parameters and return values to match the
    operator's dispatch signature, although registration does not enforce the annotations.
    For ATen handlers, results are passed through ``return_and_correct_aliasing``, so
    schema-declared view, inplace, and ``out`` aliasing is applied to the outer wrappers.

    For a generated dispatcher, lookup follows the runtime class's MRO:

    1. The exact key in the class owning the dispatcher.
    2. The wildcard key in the class owning the dispatcher.
    3. The exact key in the next class in the MRO that owns a generated dispatcher.
    4. That class's wildcard key.
    5. Continue through the MRO all the way to ``MokitWrapperTensorBase``.
    6. If no handler matches, ``MokitWrapperTensorBase.__torch_function__`` returns
       ``NotImplemented`` unless the runtime class is a subclass of every participating
       type; otherwise it forwards the call with subclass torch-function dispatch disabled.
       The handshake is applied only at this final layer, so any registered handler can still
       claim mixed-type calls. The root ``__torch_dispatch__`` raises ``NotImplementedError``.

    A custom ``__torch_function__`` or ``__torch_dispatch__`` in the MRO receives the
    delegation and may change this order.

    The base registers ATen handlers for

    - ``aten.clone.default``,
    - ``aten.clone.out``,
    - ``aten.alias.default``,
    - ``aten.detach.default``,
    - ``aten.copy_.default``,
    - and ``aten._to_copy.default``.

    They assume a storage-free outer wrapper. Each handler forwards the operation to the
    declared inner tensors and creates, checks, or updates the outer shell as appropriate:
    ``clone.default`` and ``_to_copy.default`` create a new shell, ``clone.out`` and
    ``copy_.default`` update the destination, and ``alias.default`` and ``detach.default``
    apply the operator to the shell as well. Inner tensors are processed once per exact-object
    identity, so newly created results preserve exact-object aliasing among inner tensors; out
    and in-place handlers check the destination's alias topology instead. Handlers that create
    a new wrapper initialize it with the original metadata references; out and in-place
    handlers leave the destination's metadata unchanged.


    **Pickle Support**

    Pickle uses ``__reduce_ex__`` instead of the PT2 flattening protocol. The base does not
    implement a custom ``__reduce_ex__``; it uses the one inherited from ``torch.Tensor``,
    which preserves a wrapper's Python fields automatically, including inner tensors and
    metadata. In particular, it handles nested wrappers (inner tensors that are wrappers) and
    shared inner tensor identities correctly. Only strided and ``_mkldnn`` layouts are
    supported; sparse COO and sparse compressed layouts fail explicitly during reduction.
    Loading with ``torch.load(..., weights_only=True)`` requires the concrete subclass to be
    allowlisted, for example with ``mokit.serialization.torch_add_safe_global``.


    **Deepcopy Support**

    Deepcopy uses the inherited ``torch.Tensor.__deepcopy__`` path, which clones the wrapper
    through ``aten.clone.default`` and deep-copies its Python fields. It preserves layout,
    outer stride, and nested wrapper and inner-tensor relationships. A non-zero synthetic
    storage offset is not supported and fails.


    **Safetensors Support**

    TODO: add Safetensors support.


    **Defining a Subclass**

    A subclass normally:

    1. Declares the four field-name tuples, or inherits them. The tuples follow normal MRO
       inheritance: each attribute resolves independently, so a subclass may inherit all
       four, override a subset, or redeclare them. Overriding replaces the inherited tuple
       rather than extending it; to remove a group, override it with ``()``.
       ``__init_subclass__`` is itself inherited and validates the resolved values,
       including inherited names, when each subclass is created.

    2. Implements ``__init__`` to store every declared field, accepting the constructor
       call above. It optionally implements ``__new__``. The default ``__new__`` requires
       both geometry arguments, so a subclass supporting direct construction should derive
       them from its fields and delegate to ``super().__new__``. A subclass that needs a
       recognized shell keyword to carry an ordinary field value must override
       ``__new__``.

    3. Optionally overrides ``_make_outer_tensor`` to change shell creation, or
       ``_clone_outer_tensor`` to change clone-time shell reconstruction.

    4. Registers additional handlers through the concrete class, for example
       ``@Subclass.implements(op)`` or ``@Subclass.implements_torch_function(func)``,
       usually after the class body. The module-level ``implements`` in this module is the
       base's classmethod bound to the base; using it registers on the base.

    The internal ``_FLATTEN_ATTR_NAMES`` and ``_RESERVED_FIELD_NAMES`` are ordinary class attributes;
    they may be overridden, but normally are not.

    A subclass can define its own ``__torch_function__`` or ``__torch_dispatch__``; this
    prevents installation of the generated dispatcher. If the subclass still wants to use the
    registration mechanism, it must also define the corresponding table on the class, and the
    custom dispatcher must consult or delegate to that table itself. A custom dispatcher can
    still delegate to a parent class's dispatcher if needed.

    During construction, the outer ``layout`` defaults to ``torch.strided`` and is controlled
    through ``torch.Tensor._make_wrapper_subclass``. However, because ``__tensor_unflatten__``
    does not receive ``layout``, unlike ``outer_size`` and ``outer_stride``, ``layout`` is not
    automatically preserved in the default flatten/unflatten round trip. A subclass that
    exposes a non-strided or variable layout must preserve it itself, for example through
    metadata or class-specific construction.

    The base does not prevent a subclass from giving the outer wrapper its own storage,
    but its handlers assume a storage-free outer wrapper; such a subclass must override the
    affected handlers. A subclass that needs pickle support for an unsupported layout must
    override ``__reduce_ex__``.


    **Examples**

    A wrapper with one inner tensor and one metadata field::

        from collections.abc import Sequence
        from typing import Self

        import torch
        from torch._ops import OpOverload

        from mokit.core import MokitWrapperTensorBase
        from mokit.core.wrapper_tensor_base import unpack_args


        class ScaledTensor(MokitWrapperTensorBase):
            positional_inner_tensor_names = ("payload",)
            keyword_metadata_names = ("scale",)

            def __new__(
                cls,
                payload: torch.Tensor,
                *,
                scale: float,
                outer_size: Sequence[int | torch.SymInt] | None = None,
                outer_stride: Sequence[int | torch.SymInt] | None = None,
            ) -> Self:
                return super().__new__(
                    cls,
                    outer_size=payload.size() if outer_size is None else outer_size,
                    outer_stride=payload.stride() if outer_stride is None else outer_stride,
                    dtype=payload.dtype,
                    device=payload.device,
                )

            def __init__(
                self,
                payload: torch.Tensor,
                *,
                scale: float,
                outer_size: Sequence[int | torch.SymInt] | None = None,
                outer_stride: Sequence[int | torch.SymInt] | None = None,
            ) -> None:
                self.payload = payload
                self.scale = scale


        @ScaledTensor.implements(torch.ops.aten.neg.default)
        @unpack_args
        def _(
            cls: type[ScaledTensor],
            func: OpOverload,
            types: tuple[type[ScaledTensor], ...],
            self: ScaledTensor,
        ) -> ScaledTensor:
            return cls(
                self.payload.neg(),
                scale=self.scale,
                outer_size=self.size(),
                outer_stride=self.stride(),
            )


        x = ScaledTensor(torch.ones(2, 3), scale=0.5)
        y = x.clone()        # handled by the inherited clone.default handler
        z = torch.neg(x)     # handled by the handler registered above

        assert isinstance(y, ScaledTensor) and y.scale == 0.5
        assert isinstance(z, ScaledTensor) and torch.equal(z.payload, -x.payload)
        assert x.__tensor_flatten__() == (("payload",), (0.5,))
        assert list(x.inner_tensor_names()) == ["payload"]
        assert list(x.keyword_metadata) == [("scale", 0.5)]

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
        "inner_tensors",
        "positional_inner_tensors",
        "keyword_inner_tensors",
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

        Subclasses that need different shell metadata, or that repurpose a recognized shell
        keyword as an ordinary field value, must override this method. They may reuse it
        with ``super().__new__(cls, *args, outer_size=..., outer_stride=..., **kwargs)``.

        Args:
            *args: Positional constructor field values, accepted for symmetry with
                ``__init__`` and not inspected.
            outer_size: Outer size for the wrapper shell.
            outer_stride: Outer stride, or ``None`` for a contiguous stride.
            **kwargs: Constructor keyword values. Recognized shell keys are ``dtype``,
                ``layout``, ``device``, ``pin_memory``, ``requires_grad``,
                ``storage_offset``, ``memory_format``, ``storage_size``,
                ``dispatch_sizes_strides_policy``, ``dispatch_device``,
                ``dispatch_layout``, and ``_extra_dispatch_keys``. Only recognized keys
                that are present are forwarded to ``_make_outer_tensor``; omitted keys
                use the shell factory's defaults.

        Returns:
            The outer wrapper shell.

        Raises:
            TypeError: If the shell factory rejects the provided geometry or shell metadata.

        """
        outer_kwargs = {name: kwargs[name] for name in _MAKE_WRAPPER_SUBCLASS_KEYWORD_NAMES if name in kwargs}
        return cls._make_outer_tensor(size=outer_size, strides=outer_stride, **outer_kwargs)

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Validate the resolved flatten-protocol class attributes after subclass creation.

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
        first; declared fields follow in ``_FLATTEN_ATTR_NAMES`` order, skipping names
        already printed as tensor attributes.

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

    def __tensor_flatten__(self) -> tuple[Sequence[str], tuple[Any, ...]]:
        """
        Flatten the wrapper into inner tensor names and metadata.

        Returns:
            The inner tensor names in ``inner_tensor_names()`` order and the metadata
            values in ``metadata_names()`` order.

        """
        cls = type(self)
        return tuple(cls.inner_tensor_names()), tuple(self.metadata)

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors: Mapping[str, torch.Tensor | CustomClassBase],
        metadata: Sequence[Any],
        outer_size: Sequence[int | torch.SymInt],
        outer_stride: Sequence[int | torch.SymInt],
    ) -> Self:
        """
        Rebuild a wrapper from inner tensors, metadata, and outer geometry.

        Args:
            inner_tensors: The inner tensors, keyed by their flattened names.
            metadata: The metadata values returned by ``__tensor_flatten__``.
            outer_size: The requested outer size.
            outer_stride: The requested outer stride.

        Returns:
            The rebuilt wrapper tensor.

        """
        positional_metadata_count = len(cls.positional_metadata_names)

        inner_tensor_values = (inner_tensors[name] for name in cls.positional_inner_tensor_names)
        metadata_values = itertools.islice(metadata, positional_metadata_count)
        keyword_inner_tensor_values = {name: inner_tensors[name] for name in cls.keyword_inner_tensor_names}
        keyword_metadata_values = dict(
            zip(
                cls.keyword_metadata_names,
                itertools.islice(metadata, positional_metadata_count, None),
                strict=True,
            )
        )

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
        table = MokitWrapperTensorBase._TORCH_FN_TABLE
        handler = table.get(func)
        if handler is None:
            handler = table.get(None)
        if handler is not None:
            return handler(cls, func, types, args, kwargs)

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

        The registered handler accepts ``(cls, func, types, args, kwargs)``. If no functions
        are given, the handler is registered as a wildcard for unmatched functions. The
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
                * The decorated handler is not callable.
                * The class's handler table is not a dict.

            KeyError: If any of the following hold:
                * The factory receives a duplicate key.
                * Registration conflicts with an existing key and ``override`` is false.
                * The class defines its own ``__torch_function__`` without its own ``_TORCH_FN_TABLE``.

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
        table = MokitWrapperTensorBase._ATEN_OP_TABLE
        handler = table.get(func)
        if handler is None:
            handler = table.get(None)
        if handler is not None:
            return return_and_correct_aliasing(func, args, kwargs, handler(cls, func, types, args, kwargs))

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

        The registered handler accepts ``(cls, func, types, args, kwargs)``. If no operators
        are given, the handler is registered as a wildcard for unmatched operators. The
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
                * The decorated handler is not callable.
                * The class's handler table is not a dict.

            KeyError: If any of the following hold:
                * The factory receives a duplicate key.
                * Registration conflicts with an existing key and ``override`` is false.
                * The class defines its own ``__torch_dispatch__`` without its own ``_ATEN_OP_TABLE``.

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
    def inner_tensor_names(cls: type[Self]) -> Iterator[str]:
        """
        Return an iterator over the inner tensor field names in canonical flatten order.

        The order is ``positional_inner_tensor_names`` followed by ``keyword_inner_tensor_names``.
        The class attributes are read when this method is called.

        Returns:
            An iterator over the flattened inner tensor field names.

        """
        return itertools.chain(cls.positional_inner_tensor_names, cls.keyword_inner_tensor_names)

    @classmethod
    def metadata_names(cls: type[Self]) -> Iterator[str]:
        """
        Return an iterator over the metadata field names in canonical flatten order.

        The order is ``positional_metadata_names`` followed by ``keyword_metadata_names``.
        The class attributes are read when this method is called.

        Returns:
            An iterator over the flattened metadata field names.

        """
        return itertools.chain(cls.positional_metadata_names, cls.keyword_metadata_names)

    @property
    def inner_tensors(self: Self) -> Iterator[torch.Tensor | CustomClassBase]:
        """
        Return an iterator over the declared inner tensor values.

        Returns:
            An iterator over the declared inner tensor values in canonical flatten order, by
            reference.

        """
        return (getattr(self, name) for name in type(self).inner_tensor_names())

    @property
    def positional_inner_tensors(self: Self) -> Iterator[torch.Tensor | CustomClassBase]:
        """
        Return an iterator over positional inner tensor values in declaration order.

        Returns:
            An iterator over the positional inner tensor values by reference.

        """
        return (getattr(self, name) for name in type(self).positional_inner_tensor_names)

    @property
    def keyword_inner_tensors(self: Self) -> Iterator[tuple[str, torch.Tensor | CustomClassBase]]:
        """
        Return an iterator over keyword inner tensor field names and values.

        Returns:
            An iterator over ``(name, value)`` pairs, with values by reference.

        """
        return ((name, getattr(self, name)) for name in type(self).keyword_inner_tensor_names)

    @property
    def metadata(self: Self) -> Iterator[Any]:
        """
        Return an iterator over the declared metadata values.

        Returns:
            An iterator over the declared metadata values in canonical flatten order, by
            reference.

        """
        return (getattr(self, name) for name in type(self).metadata_names())

    @property
    def positional_metadata(self: Self) -> Iterator[Any]:
        """
        Return an iterator over positional metadata values in declaration order.

        Returns:
            An iterator over the positional metadata values by reference.

        """
        return (getattr(self, name) for name in type(self).positional_metadata_names)

    @property
    def keyword_metadata(self: Self) -> Iterator[tuple[str, Any]]:
        """
        Return an iterator over keyword metadata field names and values.

        Returns:
            An iterator over ``(name, value)`` pairs, with values by reference.

        """
        return ((name, getattr(self, name)) for name in type(self).keyword_metadata_names)

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
        ``torch.Tensor._make_wrapper_subclass``. ``__new__`` forwards only the recognized
        shell keywords that the caller supplied, so overrides must accept a subset of them
        and preserve the shell factory's defaults for omitted keys. Subclasses may override
        this method to provide their own outer wrapper creation logic.

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
    def _split_inner_tensor_values(
        cls: type[Self],
        values: Iterator[Any] | Sequence[Any],
    ) -> tuple[Sequence[Any], Mapping[str, Any]]:
        """
        Split processed inner tensor values into positional and keyword groups.

        Args:
            values: The processed inner tensor values in ``inner_tensor_names()`` order.

        Returns:
            The positional inner tensor values and the keyword inner tensor values keyed
            by field name.

        Raises:
            ValueError: If ``values`` does not yield one value per declared inner
                tensor.

        """
        positional_count = len(cls.positional_inner_tensor_names)

        if isinstance(values, Sequence):
            positional_inner_tensors = values[:positional_count]
            remaining_iterator = itertools.islice(values, positional_count, None)
        else:
            positional_inner_tensors = tuple(itertools.islice(values, positional_count))
            remaining_iterator = values

        if len(positional_inner_tensors) != positional_count:
            raise ValueError(
                f"expected {positional_count} positional inner tensor values, got {len(positional_inner_tensors)}"
            )

        keyword_inner_tensors = dict(zip(cls.keyword_inner_tensor_names, remaining_iterator, strict=True))

        return positional_inner_tensors, keyword_inner_tensors

    def _apply_fn_to_outer_tensor(
        self: Self,
        func: OpOverload | Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """
        Apply a metadata-only operator to the outer tensor shell.

        The operator runs with ``__torch_dispatch__`` disabled, so it receives the wrapper
        as a storage-free Tensor instead of re-entering this dispatcher. The result is
        retyped to the wrapper class. Only view and metadata operators are supported;
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
    types: tuple[type[Any], ...],
    self: WrapperType,
    *,
    memory_format: torch.memory_format | None = None,
) -> WrapperType:
    """
    Clone a wrapper tensor by cloning each inner tensor once per object identity.

    Two fields that reference the same tensor object share one clone, preserving
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

    cloned_inner_tensor_values = apply_fn_by_id(
        tuple(self.inner_tensors),
        func,
        memory_format=memory_format,
    )
    positional_inner_values, keyword_inner_values = cls._split_inner_tensor_values(cloned_inner_tensor_values)

    outer_tensor = self._clone_outer_tensor(memory_format=memory_format)

    initializer = cast(Callable[..., None], outer_tensor.__init__)
    initializer(
        *positional_inner_values,
        *self.positional_metadata,
        outer_size=outer_tensor.size(),
        outer_stride=outer_tensor.stride(),
        **keyword_inner_values,
        **dict(self.keyword_metadata),
    )

    return outer_tensor


@implements(torch.ops.aten.clone.out)
@unpack_args
def _[WrapperType: MokitWrapperTensorBase](
    cls: type[WrapperType],
    func: OpOverload,
    types: tuple[type[Any], ...],
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

    apply_fn_by_id(
        tuple(self.inner_tensors),
        lambda source, *, dest: func(source, memory_format=memory_format, out=dest),
        dest=tuple(out.inner_tensors),
    )

    return out


@implements(torch.ops.aten.alias.default, torch.ops.aten.detach.default)
@unpack_args
def _[WrapperType: MokitWrapperTensorBase](
    cls: type[WrapperType],
    func: OpOverload,
    types: tuple[type[Any], ...],
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

    The shell is created with ``as_subclass`` while ``__torch_dispatch__`` is disabled,
    which bypasses ``__new__`` and ``__tensor_unflatten__``. The subclass ``__init__`` is
    then called with the transformed inner tensors and the original metadata references,
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

    aliased_inner_values = apply_fn_by_id(
        tuple(self.inner_tensors),
        func,
    )
    positional_inner_values, keyword_inner_values = cls._split_inner_tensor_values(aliased_inner_values)

    initializer = cast(Callable[..., None], aliased.__init__)
    initializer(
        *positional_inner_values,
        *self.positional_metadata,
        outer_size=aliased.size(),
        outer_stride=aliased.stride(),
        **keyword_inner_values,
        **dict(self.keyword_metadata),
    )

    return aliased


@implements(torch.ops.aten.copy_.default)
@unpack_args
def _[WrapperType: MokitWrapperTensorBase](
    cls: type[WrapperType],
    func: OpOverload,
    types: tuple[type[Any], ...],
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

    apply_fn_by_id(
        tuple(src.inner_tensors),
        lambda source, *, dest: func(dest, source, non_blocking=non_blocking),
        dest=tuple(self.inner_tensors),
    )

    return self


@implements(torch.ops.aten._to_copy.default)
@unpack_args
def _[WrapperType: MokitWrapperTensorBase](
    cls: type[WrapperType],
    func: OpOverload,
    types: tuple[type[Any], ...],
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

    if dtype is not None or layout is not None:
        requested_conversions = tuple(
            name
            for name, value in (
                ("dtype", dtype),
                ("layout", layout),
            )
            if value is not None
        )
        raise NotImplementedError(
            f"{func}: base handler supports only device, pin_memory, non_blocking, "
            f"and memory_format; handle and remove {requested_conversions} before delegating"
        )

    new_inner_tensor_values = apply_fn_by_id(
        tuple(self.inner_tensors),
        lambda src: func(
            src,
            device=device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=memory_format,
        ),
    )
    positional_inner_values, keyword_inner_values = cls._split_inner_tensor_values(new_inner_tensor_values)

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
        *positional_inner_values,
        *self.positional_metadata,
        outer_size=outer_tensor.size(),
        outer_stride=outer_tensor.stride(),
        **keyword_inner_values,
        **dict(self.keyword_metadata),
    )

    return outer_tensor
