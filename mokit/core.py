# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Core utilities for MOKIT."""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Self, cast

import torch
from torch._custom_class_base import CustomClassBase
from torch._ops import OpOverload


@dataclass(frozen=True, slots=True, kw_only=True)
class MokitConfigBase:
    """The base class for all MOKIT config classes."""


class MokitTensorBase(torch.Tensor):
    """The base class for all MOKIT tensor classes."""


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

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch torch functions for MOKIT tensors."""
        if kwargs is None:
            kwargs = {}

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]
        cls,
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch torch operations for MOKIT tensors."""
        if kwargs is None:
            kwargs = {}

        arg_types = tuple(type(arg) for arg in args)
        kwarg_types = {param: type(arg) for param, arg in kwargs.items()}
        raise NotImplementedError(
            f"{cls.__name__} dispatch: attempting to run unimplemented operator/function: "
            f"{func=}, {types=}, {arg_types=}, {kwarg_types=}"
        )

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

        Args:
            tensor_contents: Accepted for compatibility with ``torch.Tensor.__repr__``;
                unused by this implementation.

        Returns:
            The wrapper representation.

        """
        fields = [
            f"{name}={getattr(self, name)}"
            for attr_name in MokitWrapperTensorBase._FLATTEN_ATTR_NAMES
            for name in getattr(self, attr_name)
        ]
        return f"{type(self).__name__}({', '.join(fields)})"
