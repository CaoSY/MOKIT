# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Floating-point format descriptions."""

import math
from dataclasses import dataclass
from typing import Final

import torch

from mokit.core import MokitConfigBase

_REQUIRED_SPECIAL_VALUE_NAMES: Final[tuple[str, ...]] = (
    "positive_largest_normal",
    "positive_smallest_normal",
    "positive_largest_power_of_two",
    "positive_smallest_normal_power_of_two",
    "positive_largest_integer",
    "positive_number_example",
)

_REQUIRED_SIGNED_SPECIAL_VALUE_NAMES: Final[tuple[str, ...]] = (
    "negative_largest_normal",
    "negative_smallest_normal",
    "negative_largest_power_of_two",
    "negative_smallest_normal_power_of_two",
    "negative_largest_integer",
    "negative_number_example",
)

_OPTIONAL_SPECIAL_VALUE_NAMES: Final[tuple[str, ...]] = (
    "positive_zero",
    "negative_zero",
    "positive_infinity",
    "negative_infinity",
    "positive_largest_subnormal",
    "positive_smallest_subnormal",
    "negative_largest_subnormal",
    "negative_smallest_subnormal",
    "positive_smallest_subnormal_power_of_two",
    "negative_smallest_subnormal_power_of_two",
    "positive_largest_consecutive_integer",
    "negative_largest_consecutive_integer",
)

_SPECIAL_VALUE_NAMES: Final[tuple[str, ...]] = (
    *_REQUIRED_SPECIAL_VALUE_NAMES,
    *_REQUIRED_SIGNED_SPECIAL_VALUE_NAMES,
    *_OPTIONAL_SPECIAL_VALUE_NAMES,
)


def _validate_int(name: str, value: int) -> None:
    """Validate that a value is an integer and not a boolean."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")


def _validate_binary(name: str, value: str) -> None:
    """Validate that a value is a non-empty binary string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a binary string, got {type(value).__name__}")
    if not value or set(value) - {"0", "1"}:
        raise ValueError(f"{name} must be a non-empty binary string, got {value!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class BitField(MokitConfigBase):
    """
    A contiguous range of bits in a logical bit string.

    Both ends are inclusive, following the hardware convention: ``BitField(start=0, end=0)`` is a
    one-bit field. Bit 0 is the least significant bit; the physical bit order in storage is not recorded.

    Attributes:
        start: Index of the least significant bit in the field.
        end: Index of the most significant bit in the field, inclusive.

    """

    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate the bit range."""
        _validate_int("start", self.start)
        _validate_int("end", self.end)

        if self.start < 0:
            raise ValueError(f"start must be non-negative, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end must be greater than or equal to start, got [{self.start}, {self.end}]")

    @property
    def width(self) -> int:
        """Return the number of bits in the field."""
        return self.end - self.start + 1


@dataclass(frozen=True, slots=True, kw_only=True)
class SpecialFloatValue(MokitConfigBase):
    """
    A special value of a floating-point format and its bit encoding.

    Encodings are binary strings in most-significant-bit-first order. The last character corresponds to
    the least significant bit of the field, i.e. ``BitField.start``.

    Attributes:
        ref_float_value: Reference value in Python's ``float`` type.
        sign_encoding: Binary encoding of the sign field, or ``None`` if the format has no sign.
        exponent_encoding: Binary encoding of the exponent field.
        significand_encoding: Binary encoding of the significand field, or ``None`` if the format has
            no significand.
        description: Human-readable description of the value.

    """

    ref_float_value: float
    sign_encoding: str | None
    exponent_encoding: str
    significand_encoding: str | None
    description: str

    def __post_init__(self) -> None:
        """Validate the reference value and the encodings."""
        if not isinstance(self.ref_float_value, float):
            raise TypeError(f"ref_float_value must be a float, got {type(self.ref_float_value).__name__}")

        if self.sign_encoding is not None:
            _validate_binary("sign_encoding", self.sign_encoding)
        _validate_binary("exponent_encoding", self.exponent_encoding)
        if self.significand_encoding is not None:
            _validate_binary("significand_encoding", self.significand_encoding)

        if not isinstance(self.description, str):
            raise TypeError(f"description must be a str, got {type(self.description).__name__}")


@dataclass(frozen=True, slots=True, kw_only=True)
class FloatSpec(MokitConfigBase):
    """
    A description of a floating-point format.

    Values are described by their logical bit layout: bit 0 is the least significant bit, and both ends
    of a ``BitField`` are inclusive. The sign, exponent, and significand fields must tile the entire
    logical bit width without gaps or overlaps, in that order from the most significant bit down.

    For the negative special values, ``largest`` and ``smallest`` refer to magnitude: ``largest`` is the
    most negative value, and ``smallest`` is the negative value closest to zero. All fields whose value
    does not exist in a format are set to ``None``; for example, E8M0 has no sign, zero, infinity, or
    subnormals.

    Attributes:
        torch_dtype: PyTorch dtype this format describes.

        is_packed: Whether ``torch_dtype`` packs multiple logical elements into one storage unit, as
            ``torch.float4_e2m1fn_x2`` does.

        logical_bit_width: Number of bits in the logical representation at full precision.
        storage_bit_width: Number of bits occupied by one element in storage.

        sign: Layout of the sign field, or ``None`` for an unsigned format.
        exponent: Layout of the exponent field.
        significand: Layout of the significand field, or ``None`` for a format without a significand.

        exponent_bias: Bias subtracted from the stored exponent to obtain the unbiased exponent.

        has_implicit_leading_bit: Whether normal values have an implicit leading one. This can be True
            even when the format has no significand field, as in E8M0.

        exponent_inf_value: Value of the exponent field used by infinity, or ``None`` if the format has
            no infinity.
        exponent_nan_value: Value of the exponent field used by NaN, or ``None`` if the format has no NaN.

        is_signed: Whether the format has a sign bit.
        has_inf: Whether the format can represent infinities.
        has_nan: Whether the format can represent NaNs.

        positive_zero: Positive zero, or ``None`` if the format has no zero.
        negative_zero: Negative zero, or ``None`` if the format has no negative zero.

        positive_infinity: Positive infinity, or ``None`` if the format has no infinity.
        negative_infinity: Negative infinity, or ``None`` if the format has no infinity.

        positive_largest_normal: Positive normal with the largest magnitude.
        positive_smallest_normal: Positive normal with the smallest magnitude.
        negative_largest_normal: Negative normal with the largest magnitude.
        negative_smallest_normal: Negative normal with the smallest magnitude.

        positive_largest_subnormal: Positive subnormal with the largest magnitude, or ``None`` if the
            format has no subnormals.
        positive_smallest_subnormal: Positive subnormal with the smallest magnitude, or ``None`` if the
            format has no subnormals.
        negative_largest_subnormal: Negative subnormal with the largest magnitude, or ``None`` if the
            format has no subnormals.
        negative_smallest_subnormal: Negative subnormal with the smallest magnitude, or ``None`` if the
            format has no subnormals.

        positive_largest_power_of_two: Largest positive power of two representable.
        positive_smallest_normal_power_of_two: Smallest positive normal power of two.
        positive_smallest_subnormal_power_of_two: Smallest positive subnormal power of two, or
            ``None`` if the format has no subnormals.
        negative_largest_power_of_two: Negative power of two with the largest magnitude.
        negative_smallest_normal_power_of_two: Negative normal power of two with the smallest magnitude.
        negative_smallest_subnormal_power_of_two: Negative subnormal power of two with the smallest
            magnitude, or ``None`` if the format has no subnormals.

        positive_largest_integer: Largest positive integer-valued float.
        negative_largest_integer: Most negative integer-valued float.
        positive_largest_consecutive_integer: Largest positive integer ``n`` such that every integer
            in ``[0, n]`` is exactly representable, or ``None`` if the format has no such integer.
        negative_largest_consecutive_integer: Most negative integer ``n`` such that every integer in
            ``[n, 0]`` is exactly representable, or ``None`` if the format has no such integer.

        positive_number_example: A positive value used to exercise the encoding.
        negative_number_example: A negative value used to exercise the encoding, or ``None`` for an
            unsigned format.

    """

    torch_dtype: torch.dtype
    is_packed: bool

    logical_bit_width: int
    storage_bit_width: int

    sign: BitField | None
    exponent: BitField
    significand: BitField | None

    exponent_bias: int
    has_implicit_leading_bit: bool

    exponent_inf_value: int | None
    exponent_nan_value: int | None

    is_signed: bool
    has_inf: bool
    has_nan: bool

    positive_zero: SpecialFloatValue | None
    negative_zero: SpecialFloatValue | None

    positive_infinity: SpecialFloatValue | None
    negative_infinity: SpecialFloatValue | None

    positive_largest_normal: SpecialFloatValue
    positive_smallest_normal: SpecialFloatValue
    negative_largest_normal: SpecialFloatValue | None
    negative_smallest_normal: SpecialFloatValue | None

    positive_largest_subnormal: SpecialFloatValue | None
    positive_smallest_subnormal: SpecialFloatValue | None
    negative_largest_subnormal: SpecialFloatValue | None
    negative_smallest_subnormal: SpecialFloatValue | None

    positive_largest_power_of_two: SpecialFloatValue
    positive_smallest_normal_power_of_two: SpecialFloatValue
    positive_smallest_subnormal_power_of_two: SpecialFloatValue | None
    negative_largest_power_of_two: SpecialFloatValue | None
    negative_smallest_normal_power_of_two: SpecialFloatValue | None
    negative_smallest_subnormal_power_of_two: SpecialFloatValue | None

    positive_largest_integer: SpecialFloatValue
    negative_largest_integer: SpecialFloatValue | None
    positive_largest_consecutive_integer: SpecialFloatValue | None
    negative_largest_consecutive_integer: SpecialFloatValue | None

    positive_number_example: SpecialFloatValue
    negative_number_example: SpecialFloatValue | None

    def __post_init__(self) -> None:
        """Validate the bit layout, flags, and special values."""
        for name in ("logical_bit_width", "storage_bit_width", "exponent_bias"):
            _validate_int(name, getattr(self, name))
        for name in ("is_packed", "is_signed", "has_inf", "has_nan", "has_implicit_leading_bit"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool, got {type(getattr(self, name)).__name__}")

        if not isinstance(self.torch_dtype, torch.dtype):
            raise TypeError(f"torch_dtype must be a torch.dtype, got {type(self.torch_dtype).__name__}")

        if self.logical_bit_width <= 0:
            raise ValueError(f"logical_bit_width must be positive, got {self.logical_bit_width}")
        if self.storage_bit_width <= 0:
            raise ValueError(f"storage_bit_width must be positive, got {self.storage_bit_width}")

        if not isinstance(self.exponent, BitField):
            raise TypeError(f"exponent must be a BitField, got {type(self.exponent).__name__}")
        for name in ("sign", "significand"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, BitField):
                raise TypeError(f"{name} must be a BitField or None, got {type(value).__name__}")

        if self.is_signed != (self.sign is not None):
            raise ValueError("is_signed must be True exactly when sign is not None")
        if self.sign is not None and self.sign.width != 1:
            raise ValueError(f"the sign field must be one bit wide, got {self.sign.width}")

        self._validate_layout()
        self._validate_exponent_special_values()
        self._validate_special_values()

    def _validate_layout(self) -> None:
        """Validate that the bit fields tile the logical bit width in sign-exponent-significand order."""
        expected_end = self.logical_bit_width - 1
        for name in ("sign", "exponent", "significand"):
            field: BitField | None = getattr(self, name)
            if field is None:
                continue
            if field.end != expected_end:
                raise ValueError(f"{name} must end at bit {expected_end}, got [{field.start}, {field.end}]")
            expected_end = field.start - 1

        if expected_end != -1:
            raise ValueError(
                f"bit fields must cover exactly {self.logical_bit_width} bits, "
                f"but bits 0..{expected_end} are unassigned"
            )

    def _validate_exponent_special_values(self) -> None:
        """Validate the exponent values used by infinities and NaNs."""
        exponent_max = (1 << self.exponent.width) - 1

        if self.has_inf != (self.exponent_inf_value is not None):
            raise ValueError("exponent_inf_value must be set exactly when has_inf is True")
        if self.has_nan != (self.exponent_nan_value is not None):
            raise ValueError("exponent_nan_value must be set exactly when has_nan is True")

        for name in ("exponent_inf_value", "exponent_nan_value"):
            value = getattr(self, name)
            if value is None:
                continue
            _validate_int(name, value)
            if not 0 <= value <= exponent_max:
                raise ValueError(f"{name} must be in [0, {exponent_max}], got {value}")

    def _validate_special_values(self) -> None:
        """Validate the presence, encodings, and signs of the special values."""
        for name in _REQUIRED_SPECIAL_VALUE_NAMES:
            if getattr(self, name) is None:
                raise ValueError(f"{name} is required")

        if not self.is_signed:
            for name in _SPECIAL_VALUE_NAMES:
                if name.startswith("negative_") and getattr(self, name) is not None:
                    raise ValueError(f"{name} must be None for an unsigned format")
        else:
            for name in _REQUIRED_SIGNED_SPECIAL_VALUE_NAMES:
                if getattr(self, name) is None:
                    raise ValueError(f"{name} is required for a signed format")

        if self.positive_zero is None and self.negative_zero is not None:
            raise ValueError("negative_zero cannot be set when positive_zero is None")

        if self.positive_largest_consecutive_integer is None:
            if self.negative_largest_consecutive_integer is not None:
                raise ValueError(
                    "negative_largest_consecutive_integer cannot be set when "
                    "positive_largest_consecutive_integer is None"
                )
        elif self.is_signed and self.negative_largest_consecutive_integer is None:
            raise ValueError("negative_largest_consecutive_integer is required for a signed format")

        self._validate_infinities()
        self._validate_subnormals()

        for name in _SPECIAL_VALUE_NAMES:
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, SpecialFloatValue):
                raise TypeError(f"{name} must be a SpecialFloatValue or None, got {type(value).__name__}")
            self._validate_special_value_encoding(name, value)

    def _validate_infinities(self) -> None:
        """Validate the infinity values against ``has_inf`` and ``is_signed``."""
        if self.has_inf != (self.positive_infinity is not None):
            raise ValueError("positive_infinity must be set exactly when has_inf is True")
        if (self.negative_infinity is not None) != (self.has_inf and self.is_signed):
            raise ValueError("negative_infinity must be set exactly when has_inf is True and the format is signed")

    def _validate_subnormals(self) -> None:
        """Validate that subnormal values are present as complete groups."""
        positive_present = self.positive_largest_subnormal is not None
        if (self.positive_smallest_subnormal is not None) != positive_present:
            raise ValueError("positive subnormal values must be either both set or both None")
        if (self.positive_smallest_subnormal_power_of_two is not None) != positive_present:
            raise ValueError("positive_smallest_subnormal_power_of_two must be set exactly when subnormal values are")

        if self.is_signed:
            negative_present = self.negative_largest_subnormal is not None
            if (self.negative_smallest_subnormal is not None) != negative_present:
                raise ValueError("negative subnormal values must be either both set or both None")
            if negative_present != positive_present:
                raise ValueError("negative subnormal values must be present exactly when positive ones are")
            if (self.negative_smallest_subnormal_power_of_two is not None) != negative_present:
                raise ValueError(
                    "negative_smallest_subnormal_power_of_two must be set exactly when negative subnormal values are"
                )

    def _validate_special_value_encoding(self, name: str, value: SpecialFloatValue) -> None:
        """Validate one special value's encodings against the bit layout and its sign."""
        expected_negative = name.startswith("negative_")

        if self.sign is None:
            if value.sign_encoding is not None:
                raise ValueError(f"{name}.sign_encoding must be None for an unsigned format")
        else:
            if value.sign_encoding is None:
                raise ValueError(f"{name}.sign_encoding is required for a signed format")
            if len(value.sign_encoding) != self.sign.width:
                raise ValueError(f"{name}.sign_encoding must be {self.sign.width} bits, got {len(value.sign_encoding)}")
            expected_sign = "1" if expected_negative else "0"
            if value.sign_encoding != expected_sign:
                raise ValueError(f"{name}.sign_encoding must be {expected_sign!r}, got {value.sign_encoding!r}")

        if len(value.exponent_encoding) != self.exponent.width:
            raise ValueError(
                f"{name}.exponent_encoding must be {self.exponent.width} bits, got {len(value.exponent_encoding)}"
            )

        if self.significand is None:
            if value.significand_encoding is not None:
                raise ValueError(f"{name}.significand_encoding must be None for a format without a significand")
        else:
            if value.significand_encoding is None:
                raise ValueError(f"{name}.significand_encoding is required")
            if len(value.significand_encoding) != self.significand.width:
                raise ValueError(
                    f"{name}.significand_encoding must be {self.significand.width} bits, "
                    f"got {len(value.significand_encoding)}"
                )

        if expected_negative != (math.copysign(1.0, value.ref_float_value) < 0):
            kind = "negative" if expected_negative else "positive"
            raise ValueError(f"{name}.ref_float_value must be {kind}")


# The dtypes for which a FloatSpec is provided.
FLOAT_SPEC_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
    torch.float4_e2m1fn_x2,
)
