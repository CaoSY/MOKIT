# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Core utilities for MOKIT."""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self, cast

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

    Subclasses inherit the dispatch helpers and the PT2 flatten/unflatten helpers. The
    flatten helpers are opt-in: a subclass that does not use them does not need to define
    any of the class attributes below. To use the helpers, a subclass must define
    ``inner_tensor_names`` and ``metadata_names``; an empty tuple is allowed. The optional
    variants may be omitted. When defined, all four attributes must be tuples; they must
    not be ``None``.

    A name listed in one of these class attributes is a declaration by the subclass that the
    corresponding field belongs to the wrapper. The base class does not validate field
    values; a field may be ``None``. PyTorch's tracing protocol additionally requires each
    inner-tensor name to resolve to a tensor or a registered opaque type.

    These attributes are inherited through normal Python MRO semantics; if multiple bases
    define them, the first definition in the MRO is used. Because of inheritance, once a parent
    class opts in, a subclass remains opted in and cannot opt out by omission. If it adds,
    removes, or renames a flattened field, it must override the affected tuples explicitly,
    for example by extending the parent tuple. The base validates the effective tuples but
    cannot detect fields that are not declared in them.

    Attributes:
        inner_tensor_names: Names of the required inner tensors, in the order of the
            positional tensor arguments of ``__init__`` and ``__new__``. Omitting this
            attribute means the subclass has not opted in to the flatten helpers.
        metadata_names: Names of the required non-``Tensor`` attributes, in the order of
            the positional metadata arguments of ``__init__`` and ``__new__``. Omitting
            this attribute means the subclass has not opted in to the flatten helpers.
        optional_inner_tensor_names: Names of optional inner tensors, passed as keyword
            arguments to ``__init__`` and ``__new__``.
        optional_metadata_names: Names of optional non-``Tensor`` attributes, passed as
            keyword arguments to ``__init__`` and ``__new__``.

    The constructors receive one flat positional argument list: the values named by
    ``inner_tensor_names`` followed by the values named by ``metadata_names``. Optional
    values are keyword arguments. The inherited ``__tensor_unflatten__`` also passes
    ``outer_size`` and ``outer_stride`` as keyword arguments, so ``__init__`` and
    ``__new__`` must accept them; use them when the outer shape or stride is not derived
    from the inner tensors. ``outer_size`` and ``outer_stride`` are reserved keyword names
    and must not appear in the optional name tuples.

    ``__tensor_flatten__`` and ``__tensor_unflatten__`` are PyTorch's PT2 tracing protocol.
    They do not implement pickle support.

    """

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Validate the flatten-protocol class attributes after subclass creation.

        Args:
            **kwargs: Class keyword arguments forwarded to the next base class.

        Raises:
            TypeError: If a defined attribute is not a tuple of strings.
            ValueError: If a name is duplicated or reserved, or appears in more than one
                class attribute tuple.

        """
        super().__init_subclass__(**kwargs)

        CLS_ATTR_NAMES = (
            "inner_tensor_names",
            "metadata_names",
            "optional_inner_tensor_names",
            "optional_metadata_names",
        )

        defined_cls_attr_names = tuple(name for name in CLS_ATTR_NAMES if hasattr(cls, name))

        for attr_name in defined_cls_attr_names:
            attr = getattr(cls, attr_name)
            if not isinstance(attr, tuple):
                raise TypeError(f"The class attribute '{attr_name}' must be a tuple when defined.")
            if not all(isinstance(name, str) for name in attr):
                raise TypeError(f"The class attribute '{attr_name}' must be a tuple of str when defined.")

        all_declared_fields = tuple(field for attr_name in defined_cls_attr_names for field in getattr(cls, attr_name))

        duplicated_fields = tuple(field for field, freq in Counter(all_declared_fields).items() if freq > 1)
        if duplicated_fields:
            raise ValueError(f"Field name(s) {sorted(duplicated_fields)!r} is (are) declared more than once.")

        RESERVED_FIELDS = ("outer_size", "outer_stride")

        conflict_fields = tuple(field for field in RESERVED_FIELDS if field in all_declared_fields)
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

        Raises:
            NotImplementedError: If the subclass has not defined ``inner_tensor_names``
                and ``metadata_names``.

        """
        cls = type(self)
        if hasattr(cls, "inner_tensor_names") and hasattr(cls, "metadata_names"):
            tensor_names = list(cls.inner_tensor_names)

            if hasattr(cls, "optional_inner_tensor_names"):
                tensor_names.extend(name for name in cls.optional_inner_tensor_names if hasattr(self, name))

            metadata = {name: getattr(self, name) for name in cls.metadata_names}

            if hasattr(cls, "optional_metadata_names"):
                metadata.update(
                    (name, getattr(self, name)) for name in cls.optional_metadata_names if hasattr(self, name)
                )

            return tensor_names, metadata

        else:
            raise NotImplementedError(
                f"{cls.__name__} must define inner_tensor_names and metadata_names to use "
                "the inherited __tensor_flatten__."
            )

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

        Raises:
            NotImplementedError: If the subclass has not defined ``inner_tensor_names``
                and ``metadata_names``.

        """
        if hasattr(cls, "inner_tensor_names") and hasattr(cls, "metadata_names"):
            cls_inner_tensor = (inner_tensors[name] for name in cls.inner_tensor_names)

            if hasattr(cls, "optional_inner_tensor_names"):
                optional_inner_tensors = {
                    name: inner_tensors[name] for name in cls.optional_inner_tensor_names if name in inner_tensors
                }
            else:
                optional_inner_tensors = {}

            cls_metadata = (metadata[name] for name in cls.metadata_names)

            if hasattr(cls, "optional_metadata_names"):
                optional_metadata = {name: metadata[name] for name in cls.optional_metadata_names if name in metadata}
            else:
                optional_metadata = {}

            constructor = cast(Callable[..., Self], cls)
            return constructor(
                *cls_inner_tensor,
                *cls_metadata,
                outer_size=outer_size,
                outer_stride=outer_stride,
                **optional_inner_tensors,
                **optional_metadata,
            )

        else:
            raise NotImplementedError(
                f"{cls.__name__} must define inner_tensor_names and metadata_names to use "
                "the inherited __tensor_unflatten__."
            )
