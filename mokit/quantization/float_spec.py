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
    "positive_nan",
    "negative_nan",
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

        register_bit_width: Number of bits occupied by one element in the register.

        sign: Layout of the sign field, or ``None`` for an unsigned format.
        exponent: Layout of the exponent field.
        significand: Layout of the significand field, or ``None`` for a format without a significand.

        exponent_bias: Bias subtracted from the stored exponent to obtain the unbiased exponent.

        has_implicit_leading_bit: Whether normal values have an implicit leading one. This can be True
            even when the format has no significand field, as in E8M0.

        eps: Difference between 1.0 and the next representable value.

        positive_zero: Positive zero, or ``None`` if the format has no zero.
        negative_zero: Negative zero, or ``None`` if the format has no negative zero.

        positive_infinity: Positive infinity, or ``None`` if the format has no infinity.
        negative_infinity: Negative infinity, or ``None`` if the format has no infinity.

        positive_nan: Positive NaN, or ``None`` if the format has no NaN.
        negative_nan: Negative NaN, or ``None`` if the format has no NaN.

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

    Properties:
        logical_bit_width: Total width of the logical bit string.

        is_signed: Whether the format has a sign bit.

        has_inf: Whether the format can represent infinities.

        has_nan: Whether the format can represent NaNs.

    """

    torch_dtype: torch.dtype
    is_packed: bool

    @property
    def logical_bit_width(self) -> int:
        """Return the total width of the logical bit string."""
        return sum(f.width for f in (self.sign, self.exponent, self.significand) if f is not None)

    register_bit_width: int

    sign: BitField | None
    exponent: BitField
    significand: BitField | None

    exponent_bias: int
    has_implicit_leading_bit: bool

    eps: float

    @property
    def is_signed(self) -> bool:
        """Return whether the format has a sign bit."""
        return self.sign is not None

    @property
    def has_inf(self) -> bool:
        """Return whether the format can represent infinities."""
        return self.positive_infinity is not None or self.negative_infinity is not None

    @property
    def has_nan(self) -> bool:
        """Return whether the format can represent NaNs."""
        return self.positive_nan is not None or self.negative_nan is not None

    positive_zero: SpecialFloatValue | None
    negative_zero: SpecialFloatValue | None

    positive_infinity: SpecialFloatValue | None
    negative_infinity: SpecialFloatValue | None

    positive_nan: SpecialFloatValue | None
    negative_nan: SpecialFloatValue | None

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
        for name in ("register_bit_width", "exponent_bias"):
            _validate_int(name, getattr(self, name))
        for name in ("is_packed", "has_implicit_leading_bit"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool, got {type(getattr(self, name)).__name__}")

        if not isinstance(self.torch_dtype, torch.dtype):
            raise TypeError(f"torch_dtype must be a torch.dtype, got {type(self.torch_dtype).__name__}")

        if self.register_bit_width <= 0:
            raise ValueError(f"register_bit_width must be positive, got {self.register_bit_width}")

        if not isinstance(self.eps, float):
            raise TypeError(f"eps must be a float, got {type(self.eps).__name__}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

        if not isinstance(self.exponent, BitField):
            raise TypeError(f"exponent must be a BitField, got {type(self.exponent).__name__}")
        for name in ("sign", "significand"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, BitField):
                raise TypeError(f"{name} must be a BitField or None, got {type(value).__name__}")

        if self.sign is not None and self.sign.width != 1:
            raise ValueError(f"the sign field must be one bit wide, got {self.sign.width}")

        self._validate_layout()

        if self.register_bit_width < self.logical_bit_width:
            raise ValueError(
                f"register_bit_width must be at least logical_bit_width, got {self.register_bit_width} < "
                f"{self.logical_bit_width}"
            )

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
        """Validate the infinity values against signedness."""
        if (self.negative_infinity is not None) != (self.positive_infinity is not None and self.is_signed):
            raise ValueError(
                "negative_infinity must be set exactly when positive_infinity is set and the format is signed"
            )

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


# IEEE 754-2019, Section 3.4 and 3.6 (Table 3.5); reference PDF:
# https://www-users.cse.umn.edu/~vinals/tspot_files/phys4041/2020/IEEE%20Standard%20754-2019.pdf

# Binary64 encoding constants.
_FLOAT64_EXPONENT_ALL_ONES: Final[int] = (1 << 11) - 1
_FLOAT64_MAX_FINITE_EXPONENT: Final[int] = _FLOAT64_EXPONENT_ALL_ONES - 1
_FLOAT64_SIGNIFICAND_ALL_ONES: Final[int] = (1 << 52) - 1
_FLOAT64_SMALLEST_SUBNORMAL: Final[float] = math.ldexp(1.0, -1074)
_FLOAT64_LARGEST_SUBNORMAL: Final[float] = math.ldexp(_FLOAT64_SIGNIFICAND_ALL_ONES, -1074)
_FLOAT64_LARGEST_POWER_OF_TWO: Final[float] = math.ldexp(1.0, 1023)
_FLOAT64_LARGEST_CONSECUTIVE_INTEGER: Final[float] = math.ldexp(1.0, 53)


FLOAT64_SPEC: Final[FloatSpec] = FloatSpec(
    torch_dtype=torch.float64,
    is_packed=False,
    register_bit_width=torch.finfo(torch.float64).bits,
    sign=BitField(start=63, end=63),
    exponent=BitField(start=52, end=62),
    significand=BitField(start=0, end=51),
    exponent_bias=1023,
    has_implicit_leading_bit=True,
    eps=torch.finfo(torch.float64).eps,
    positive_zero=SpecialFloatValue(
        ref_float_value=0.0,
        sign_encoding=str(0),
        exponent_encoding=format(0, "011b"),
        significand_encoding=format(0, "052b"),
        description="positive zero",
    ),
    negative_zero=SpecialFloatValue(
        ref_float_value=-0.0,
        sign_encoding=str(1),
        exponent_encoding=format(0, "011b"),
        significand_encoding=format(0, "052b"),
        description="negative zero",
    ),
    positive_infinity=SpecialFloatValue(
        ref_float_value=math.inf,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT64_EXPONENT_ALL_ONES, "011b"),
        significand_encoding=format(0, "052b"),
        description="positive infinity",
    ),
    negative_infinity=SpecialFloatValue(
        ref_float_value=-math.inf,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT64_EXPONENT_ALL_ONES, "011b"),
        significand_encoding=format(0, "052b"),
        description="negative infinity",
    ),
    positive_nan=SpecialFloatValue(
        ref_float_value=math.nan,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT64_EXPONENT_ALL_ONES, "011b"),
        significand_encoding=format(1 << 51, "052b"),
        description="positive quiet NaN",
    ),
    negative_nan=SpecialFloatValue(
        ref_float_value=-math.nan,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT64_EXPONENT_ALL_ONES, "011b"),
        significand_encoding=format(1 << 51, "052b"),
        description="negative quiet NaN",
    ),
    positive_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float64).max,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT64_MAX_FINITE_EXPONENT, "011b"),
        significand_encoding=format(_FLOAT64_SIGNIFICAND_ALL_ONES, "052b"),
        description="largest positive normal",
    ),
    positive_smallest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float64).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "011b"),
        significand_encoding=format(0, "052b"),
        description="smallest positive normal",
    ),
    negative_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float64).min,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT64_MAX_FINITE_EXPONENT, "011b"),
        significand_encoding=format(_FLOAT64_SIGNIFICAND_ALL_ONES, "052b"),
        description="most negative normal",
    ),
    negative_smallest_normal=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float64).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "011b"),
        significand_encoding=format(0, "052b"),
        description="negative normal closest to zero",
    ),
    positive_largest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT64_LARGEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "011b"),
        significand_encoding=format(_FLOAT64_SIGNIFICAND_ALL_ONES, "052b"),
        description="largest positive subnormal",
    ),
    positive_smallest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT64_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "011b"),
        significand_encoding=format(1, "052b"),
        description="smallest positive subnormal",
    ),
    negative_largest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT64_LARGEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "011b"),
        significand_encoding=format(_FLOAT64_SIGNIFICAND_ALL_ONES, "052b"),
        description="most negative subnormal",
    ),
    negative_smallest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT64_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "011b"),
        significand_encoding=format(1, "052b"),
        description="negative subnormal closest to zero",
    ),
    positive_largest_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT64_LARGEST_POWER_OF_TWO,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT64_MAX_FINITE_EXPONENT, "011b"),
        significand_encoding=format(0, "052b"),
        description="largest positive power of two",
    ),
    positive_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float64).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "011b"),
        significand_encoding=format(0, "052b"),
        description="smallest positive normal power of two",
    ),
    positive_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT64_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "011b"),
        significand_encoding=format(1, "052b"),
        description="smallest positive subnormal power of two",
    ),
    negative_largest_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT64_LARGEST_POWER_OF_TWO,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT64_MAX_FINITE_EXPONENT, "011b"),
        significand_encoding=format(0, "052b"),
        description="negative power of two with largest magnitude",
    ),
    negative_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float64).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "011b"),
        significand_encoding=format(0, "052b"),
        description="negative normal power of two with smallest magnitude",
    ),
    negative_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT64_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "011b"),
        significand_encoding=format(1, "052b"),
        description="negative subnormal power of two with smallest magnitude",
    ),
    positive_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float64).max,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT64_MAX_FINITE_EXPONENT, "011b"),
        significand_encoding=format(_FLOAT64_SIGNIFICAND_ALL_ONES, "052b"),
        description="largest positive integer-valued float",
    ),
    negative_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float64).min,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT64_MAX_FINITE_EXPONENT, "011b"),
        significand_encoding=format(_FLOAT64_SIGNIFICAND_ALL_ONES, "052b"),
        description="most negative integer-valued float",
    ),
    positive_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=_FLOAT64_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(1076, "011b"),
        significand_encoding=format(0, "052b"),
        description="largest positive consecutive integer",
    ),
    negative_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT64_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(1076, "011b"),
        significand_encoding=format(0, "052b"),
        description="most negative consecutive integer",
    ),
    positive_number_example=SpecialFloatValue(
        ref_float_value=41.0,
        sign_encoding=str(0),
        exponent_encoding=format(1028, "011b"),
        significand_encoding=format(0x4800000000000, "052b"),
        description="positive example",
    ),
    negative_number_example=SpecialFloatValue(
        ref_float_value=-37.0,
        sign_encoding=str(1),
        exponent_encoding=format(1028, "011b"),
        significand_encoding=format(0x2800000000000, "052b"),
        description="negative example",
    ),
)


# IEEE 754-2019, Section 3.4 and 3.6 (Table 3.5); reference PDF:
# https://www-users.cse.umn.edu/~vinals/tspot_files/phys4041/2020/IEEE%20Standard%20754-2019.pdf

# Binary32 encoding constants.
_FLOAT32_EXPONENT_ALL_ONES: Final[int] = (1 << 8) - 1
_FLOAT32_MAX_FINITE_EXPONENT: Final[int] = _FLOAT32_EXPONENT_ALL_ONES - 1
_FLOAT32_SIGNIFICAND_ALL_ONES: Final[int] = (1 << 23) - 1
_FLOAT32_SMALLEST_SUBNORMAL: Final[float] = math.ldexp(1.0, -149)
_FLOAT32_LARGEST_SUBNORMAL: Final[float] = math.ldexp(_FLOAT32_SIGNIFICAND_ALL_ONES, -149)
_FLOAT32_LARGEST_POWER_OF_TWO: Final[float] = math.ldexp(1.0, 127)
_FLOAT32_LARGEST_CONSECUTIVE_INTEGER: Final[float] = math.ldexp(1.0, 24)


FLOAT32_SPEC: Final[FloatSpec] = FloatSpec(
    torch_dtype=torch.float32,
    is_packed=False,
    register_bit_width=torch.finfo(torch.float32).bits,
    sign=BitField(start=31, end=31),
    exponent=BitField(start=23, end=30),
    significand=BitField(start=0, end=22),
    exponent_bias=127,
    has_implicit_leading_bit=True,
    eps=torch.finfo(torch.float32).eps,
    positive_zero=SpecialFloatValue(
        ref_float_value=0.0,
        sign_encoding=str(0),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(0, "023b"),
        description="positive zero",
    ),
    negative_zero=SpecialFloatValue(
        ref_float_value=-0.0,
        sign_encoding=str(1),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(0, "023b"),
        description="negative zero",
    ),
    positive_infinity=SpecialFloatValue(
        ref_float_value=math.inf,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT32_EXPONENT_ALL_ONES, "08b"),
        significand_encoding=format(0, "023b"),
        description="positive infinity",
    ),
    negative_infinity=SpecialFloatValue(
        ref_float_value=-math.inf,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT32_EXPONENT_ALL_ONES, "08b"),
        significand_encoding=format(0, "023b"),
        description="negative infinity",
    ),
    positive_nan=SpecialFloatValue(
        ref_float_value=math.nan,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT32_EXPONENT_ALL_ONES, "08b"),
        significand_encoding=format(1 << 22, "023b"),
        description="positive quiet NaN",
    ),
    negative_nan=SpecialFloatValue(
        ref_float_value=-math.nan,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT32_EXPONENT_ALL_ONES, "08b"),
        significand_encoding=format(1 << 22, "023b"),
        description="negative quiet NaN",
    ),
    positive_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float32).max,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT32_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=format(_FLOAT32_SIGNIFICAND_ALL_ONES, "023b"),
        description="largest positive normal",
    ),
    positive_smallest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float32).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "08b"),
        significand_encoding=format(0, "023b"),
        description="smallest positive normal",
    ),
    negative_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float32).min,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT32_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=format(_FLOAT32_SIGNIFICAND_ALL_ONES, "023b"),
        description="most negative normal",
    ),
    negative_smallest_normal=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float32).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "08b"),
        significand_encoding=format(0, "023b"),
        description="negative normal closest to zero",
    ),
    positive_largest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT32_LARGEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(_FLOAT32_SIGNIFICAND_ALL_ONES, "023b"),
        description="largest positive subnormal",
    ),
    positive_smallest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT32_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(1, "023b"),
        description="smallest positive subnormal",
    ),
    negative_largest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT32_LARGEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(_FLOAT32_SIGNIFICAND_ALL_ONES, "023b"),
        description="most negative subnormal",
    ),
    negative_smallest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT32_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(1, "023b"),
        description="negative subnormal closest to zero",
    ),
    positive_largest_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT32_LARGEST_POWER_OF_TWO,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT32_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=format(0, "023b"),
        description="largest positive power of two",
    ),
    positive_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float32).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "08b"),
        significand_encoding=format(0, "023b"),
        description="smallest positive normal power of two",
    ),
    positive_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT32_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(1, "023b"),
        description="smallest positive subnormal power of two",
    ),
    negative_largest_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT32_LARGEST_POWER_OF_TWO,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT32_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=format(0, "023b"),
        description="negative power of two with largest magnitude",
    ),
    negative_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float32).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "08b"),
        significand_encoding=format(0, "023b"),
        description="negative normal power of two with smallest magnitude",
    ),
    negative_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT32_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(1, "023b"),
        description="negative subnormal power of two with smallest magnitude",
    ),
    positive_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float32).max,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT32_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=format(_FLOAT32_SIGNIFICAND_ALL_ONES, "023b"),
        description="largest positive integer-valued float",
    ),
    negative_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float32).min,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT32_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=format(_FLOAT32_SIGNIFICAND_ALL_ONES, "023b"),
        description="most negative integer-valued float",
    ),
    positive_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=_FLOAT32_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(151, "08b"),
        significand_encoding=format(0, "023b"),
        description="largest positive consecutive integer",
    ),
    negative_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT32_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(151, "08b"),
        significand_encoding=format(0, "023b"),
        description="most negative consecutive integer",
    ),
    positive_number_example=SpecialFloatValue(
        ref_float_value=41.0,
        sign_encoding=str(0),
        exponent_encoding=format(132, "08b"),
        significand_encoding=format(0x240000, "023b"),
        description="positive example",
    ),
    negative_number_example=SpecialFloatValue(
        ref_float_value=-37.0,
        sign_encoding=str(1),
        exponent_encoding=format(132, "08b"),
        significand_encoding=format(0x140000, "023b"),
        description="negative example",
    ),
)


# IEEE 754-2019, Section 3.4 and 3.6 (Table 3.5); reference PDF:
# https://www-users.cse.umn.edu/~vinals/tspot_files/phys4041/2020/IEEE%20Standard%20754-2019.pdf

# Binary16 encoding constants.
_FLOAT16_EXPONENT_ALL_ONES: Final[int] = (1 << 5) - 1
_FLOAT16_MAX_FINITE_EXPONENT: Final[int] = _FLOAT16_EXPONENT_ALL_ONES - 1
_FLOAT16_SIGNIFICAND_ALL_ONES: Final[int] = (1 << 10) - 1
_FLOAT16_SMALLEST_SUBNORMAL: Final[float] = math.ldexp(1.0, -24)
_FLOAT16_LARGEST_SUBNORMAL: Final[float] = math.ldexp(_FLOAT16_SIGNIFICAND_ALL_ONES, -24)
_FLOAT16_LARGEST_POWER_OF_TWO: Final[float] = math.ldexp(1.0, 15)
_FLOAT16_LARGEST_CONSECUTIVE_INTEGER: Final[float] = math.ldexp(1.0, 11)


FLOAT16_SPEC: Final[FloatSpec] = FloatSpec(
    torch_dtype=torch.float16,
    is_packed=False,
    register_bit_width=torch.finfo(torch.float16).bits,
    sign=BitField(start=15, end=15),
    exponent=BitField(start=10, end=14),
    significand=BitField(start=0, end=9),
    exponent_bias=15,
    has_implicit_leading_bit=True,
    eps=torch.finfo(torch.float16).eps,
    positive_zero=SpecialFloatValue(
        ref_float_value=0.0,
        sign_encoding=str(0),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(0, "010b"),
        description="positive zero",
    ),
    negative_zero=SpecialFloatValue(
        ref_float_value=-0.0,
        sign_encoding=str(1),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(0, "010b"),
        description="negative zero",
    ),
    positive_infinity=SpecialFloatValue(
        ref_float_value=math.inf,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT16_EXPONENT_ALL_ONES, "05b"),
        significand_encoding=format(0, "010b"),
        description="positive infinity",
    ),
    negative_infinity=SpecialFloatValue(
        ref_float_value=-math.inf,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT16_EXPONENT_ALL_ONES, "05b"),
        significand_encoding=format(0, "010b"),
        description="negative infinity",
    ),
    positive_nan=SpecialFloatValue(
        ref_float_value=math.nan,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT16_EXPONENT_ALL_ONES, "05b"),
        significand_encoding=format(1 << 9, "010b"),
        description="positive quiet NaN",
    ),
    negative_nan=SpecialFloatValue(
        ref_float_value=-math.nan,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT16_EXPONENT_ALL_ONES, "05b"),
        significand_encoding=format(1 << 9, "010b"),
        description="negative quiet NaN",
    ),
    positive_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float16).max,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT16_MAX_FINITE_EXPONENT, "05b"),
        significand_encoding=format(_FLOAT16_SIGNIFICAND_ALL_ONES, "010b"),
        description="largest positive normal",
    ),
    positive_smallest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float16).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "05b"),
        significand_encoding=format(0, "010b"),
        description="smallest positive normal",
    ),
    negative_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float16).min,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT16_MAX_FINITE_EXPONENT, "05b"),
        significand_encoding=format(_FLOAT16_SIGNIFICAND_ALL_ONES, "010b"),
        description="most negative normal",
    ),
    negative_smallest_normal=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float16).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "05b"),
        significand_encoding=format(0, "010b"),
        description="negative normal closest to zero",
    ),
    positive_largest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT16_LARGEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(_FLOAT16_SIGNIFICAND_ALL_ONES, "010b"),
        description="largest positive subnormal",
    ),
    positive_smallest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT16_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(1, "010b"),
        description="smallest positive subnormal",
    ),
    negative_largest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT16_LARGEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(_FLOAT16_SIGNIFICAND_ALL_ONES, "010b"),
        description="most negative subnormal",
    ),
    negative_smallest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT16_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(1, "010b"),
        description="negative subnormal closest to zero",
    ),
    positive_largest_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT16_LARGEST_POWER_OF_TWO,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT16_MAX_FINITE_EXPONENT, "05b"),
        significand_encoding=format(0, "010b"),
        description="largest positive power of two",
    ),
    positive_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float16).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "05b"),
        significand_encoding=format(0, "010b"),
        description="smallest positive normal power of two",
    ),
    positive_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT16_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(1, "010b"),
        description="smallest positive subnormal power of two",
    ),
    negative_largest_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT16_LARGEST_POWER_OF_TWO,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT16_MAX_FINITE_EXPONENT, "05b"),
        significand_encoding=format(0, "010b"),
        description="negative power of two with largest magnitude",
    ),
    negative_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float16).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "05b"),
        significand_encoding=format(0, "010b"),
        description="negative normal power of two with smallest magnitude",
    ),
    negative_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT16_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(1, "010b"),
        description="negative subnormal power of two with smallest magnitude",
    ),
    positive_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float16).max,
        sign_encoding=str(0),
        exponent_encoding=format(_FLOAT16_MAX_FINITE_EXPONENT, "05b"),
        significand_encoding=format(_FLOAT16_SIGNIFICAND_ALL_ONES, "010b"),
        description="largest positive integer-valued float",
    ),
    negative_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float16).min,
        sign_encoding=str(1),
        exponent_encoding=format(_FLOAT16_MAX_FINITE_EXPONENT, "05b"),
        significand_encoding=format(_FLOAT16_SIGNIFICAND_ALL_ONES, "010b"),
        description="most negative integer-valued float",
    ),
    positive_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=_FLOAT16_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(26, "05b"),
        significand_encoding=format(0, "010b"),
        description="largest positive consecutive integer",
    ),
    negative_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT16_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(26, "05b"),
        significand_encoding=format(0, "010b"),
        description="most negative consecutive integer",
    ),
    positive_number_example=SpecialFloatValue(
        ref_float_value=41.0,
        sign_encoding=str(0),
        exponent_encoding=format(20, "05b"),
        significand_encoding=format(0x120, "010b"),
        description="positive example",
    ),
    negative_number_example=SpecialFloatValue(
        ref_float_value=-37.0,
        sign_encoding=str(1),
        exponent_encoding=format(20, "05b"),
        significand_encoding=format(0xA0, "010b"),
        description="negative example",
    ),
)


# bfloat16 is a vendor-originated, de facto format; no published IEEE standard defines it.
# The 1/8/7 layout (sign / exponent / significand) is shared by:
#   Intel:
#   https://www.intel.com/content/dam/develop/external/us/en/documents/bf16-hardware-numerics-definition-white-paper.pdf
#   Arm (BFloat16 floating-point number format):
#   https://support.arm.com/documentation/101754/0620/armclang-Reference/Other-Compiler-specific-Features/BFloat16-floating-point-number-format  # noqa: E501
#   NVIDIA (PTX ISA, Section 5.2.3, "Alternate Floating-Point Data Formats"):
#   https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats
#   Google (1/8/7 layout, flushes subnormals to zero):
#   https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
#   bfloat16 study (Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training", 2019):
#   https://arxiv.org/pdf/1905.12322
# Implementations differ in subnormal handling (Google TPU flushes to zero; Intel's proposal
# uses FTZ) and conversion rounding: Intel's white paper proposes truncation with optional
# RNE; Google TPU uses RNE; NVIDIA uses RNE by default but PTX exposes explicit rounding
# modes; Arm BFCVT follows FPCR rounding (default RNE). This spec describes the encoding,
# including subnormals.
#   Google Cloud TPU behavior (round-to-nearest-even, no subnormals):
#   https://docs.cloud.google.com/tpu/docs/bfloat16
# IEEE P3109 is an active project, not a published standard:
#   https://standards.ieee.org/ieee/3109/11165/

# bfloat16 encoding constants.
_BFLOAT16_EXPONENT_ALL_ONES: Final[int] = (1 << 8) - 1
_BFLOAT16_MAX_FINITE_EXPONENT: Final[int] = _BFLOAT16_EXPONENT_ALL_ONES - 1
_BFLOAT16_SIGNIFICAND_ALL_ONES: Final[int] = (1 << 7) - 1
_BFLOAT16_SMALLEST_SUBNORMAL: Final[float] = math.ldexp(1.0, -133)
_BFLOAT16_LARGEST_SUBNORMAL: Final[float] = math.ldexp(_BFLOAT16_SIGNIFICAND_ALL_ONES, -133)
_BFLOAT16_LARGEST_POWER_OF_TWO: Final[float] = math.ldexp(1.0, 127)
_BFLOAT16_LARGEST_CONSECUTIVE_INTEGER: Final[float] = math.ldexp(1.0, 8)


BFLOAT16_SPEC: Final[FloatSpec] = FloatSpec(
    torch_dtype=torch.bfloat16,
    is_packed=False,
    register_bit_width=torch.finfo(torch.bfloat16).bits,
    sign=BitField(start=15, end=15),
    exponent=BitField(start=7, end=14),
    significand=BitField(start=0, end=6),
    exponent_bias=127,
    has_implicit_leading_bit=True,
    eps=torch.finfo(torch.bfloat16).eps,
    positive_zero=SpecialFloatValue(
        ref_float_value=0.0,
        sign_encoding=str(0),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(0, "07b"),
        description="positive zero",
    ),
    negative_zero=SpecialFloatValue(
        ref_float_value=-0.0,
        sign_encoding=str(1),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(0, "07b"),
        description="negative zero",
    ),
    positive_infinity=SpecialFloatValue(
        ref_float_value=math.inf,
        sign_encoding=str(0),
        exponent_encoding=format(_BFLOAT16_EXPONENT_ALL_ONES, "08b"),
        significand_encoding=format(0, "07b"),
        description="positive infinity",
    ),
    negative_infinity=SpecialFloatValue(
        ref_float_value=-math.inf,
        sign_encoding=str(1),
        exponent_encoding=format(_BFLOAT16_EXPONENT_ALL_ONES, "08b"),
        significand_encoding=format(0, "07b"),
        description="negative infinity",
    ),
    positive_nan=SpecialFloatValue(
        ref_float_value=math.nan,
        sign_encoding=str(0),
        exponent_encoding=format(_BFLOAT16_EXPONENT_ALL_ONES, "08b"),
        significand_encoding=format(1 << 6, "07b"),
        description="positive quiet NaN",
    ),
    negative_nan=SpecialFloatValue(
        ref_float_value=-math.nan,
        sign_encoding=str(1),
        exponent_encoding=format(_BFLOAT16_EXPONENT_ALL_ONES, "08b"),
        significand_encoding=format(1 << 6, "07b"),
        description="negative quiet NaN",
    ),
    positive_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.bfloat16).max,
        sign_encoding=str(0),
        exponent_encoding=format(_BFLOAT16_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=format(_BFLOAT16_SIGNIFICAND_ALL_ONES, "07b"),
        description="largest positive normal",
    ),
    positive_smallest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.bfloat16).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "08b"),
        significand_encoding=format(0, "07b"),
        description="smallest positive normal",
    ),
    negative_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.bfloat16).min,
        sign_encoding=str(1),
        exponent_encoding=format(_BFLOAT16_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=format(_BFLOAT16_SIGNIFICAND_ALL_ONES, "07b"),
        description="most negative normal",
    ),
    negative_smallest_normal=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.bfloat16).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "08b"),
        significand_encoding=format(0, "07b"),
        description="negative normal closest to zero",
    ),
    positive_largest_subnormal=SpecialFloatValue(
        ref_float_value=_BFLOAT16_LARGEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(_BFLOAT16_SIGNIFICAND_ALL_ONES, "07b"),
        description="largest positive subnormal",
    ),
    positive_smallest_subnormal=SpecialFloatValue(
        ref_float_value=_BFLOAT16_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(1, "07b"),
        description="smallest positive subnormal",
    ),
    negative_largest_subnormal=SpecialFloatValue(
        ref_float_value=-_BFLOAT16_LARGEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(_BFLOAT16_SIGNIFICAND_ALL_ONES, "07b"),
        description="most negative subnormal",
    ),
    negative_smallest_subnormal=SpecialFloatValue(
        ref_float_value=-_BFLOAT16_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(1, "07b"),
        description="negative subnormal closest to zero",
    ),
    positive_largest_power_of_two=SpecialFloatValue(
        ref_float_value=_BFLOAT16_LARGEST_POWER_OF_TWO,
        sign_encoding=str(0),
        exponent_encoding=format(_BFLOAT16_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=format(0, "07b"),
        description="largest positive power of two",
    ),
    positive_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.bfloat16).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "08b"),
        significand_encoding=format(0, "07b"),
        description="smallest positive normal power of two",
    ),
    positive_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=_BFLOAT16_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(1, "07b"),
        description="smallest positive subnormal power of two",
    ),
    negative_largest_power_of_two=SpecialFloatValue(
        ref_float_value=-_BFLOAT16_LARGEST_POWER_OF_TWO,
        sign_encoding=str(1),
        exponent_encoding=format(_BFLOAT16_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=format(0, "07b"),
        description="negative power of two with largest magnitude",
    ),
    negative_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.bfloat16).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "08b"),
        significand_encoding=format(0, "07b"),
        description="negative normal power of two with smallest magnitude",
    ),
    negative_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=-_BFLOAT16_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "08b"),
        significand_encoding=format(1, "07b"),
        description="negative subnormal power of two with smallest magnitude",
    ),
    positive_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.bfloat16).max,
        sign_encoding=str(0),
        exponent_encoding=format(_BFLOAT16_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=format(_BFLOAT16_SIGNIFICAND_ALL_ONES, "07b"),
        description="largest positive integer-valued float",
    ),
    negative_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.bfloat16).min,
        sign_encoding=str(1),
        exponent_encoding=format(_BFLOAT16_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=format(_BFLOAT16_SIGNIFICAND_ALL_ONES, "07b"),
        description="most negative integer-valued float",
    ),
    positive_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=_BFLOAT16_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(135, "08b"),
        significand_encoding=format(0, "07b"),
        description="largest positive consecutive integer",
    ),
    negative_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=-_BFLOAT16_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(135, "08b"),
        significand_encoding=format(0, "07b"),
        description="most negative consecutive integer",
    ),
    positive_number_example=SpecialFloatValue(
        ref_float_value=41.0,
        sign_encoding=str(0),
        exponent_encoding=format(132, "08b"),
        significand_encoding=format(0x24, "07b"),
        description="positive example",
    ),
    negative_number_example=SpecialFloatValue(
        ref_float_value=-37.0,
        sign_encoding=str(1),
        exponent_encoding=format(132, "08b"),
        significand_encoding=format(0x14, "07b"),
        description="negative example",
    ),
)

# FLOAT8_E4M3FN: no infinity; NaN encodings 0x7F and 0xFF; max finite 448.
# OCP 8-bit Floating Point Specification (OFP8), Rev. 1.1:
# https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf
# FP8 formats paper (Micikevicius et al.):
# https://arxiv.org/pdf/2209.05433
# FLOAT8_E4M3FN encoding constants.
_FLOAT8_E4M3FN_SIGNIFICAND_ALL_ONES: Final[int] = (1 << 3) - 1
_FLOAT8_E4M3FN_SMALLEST_SUBNORMAL: Final[float] = math.ldexp(1.0, -9)
_FLOAT8_E4M3FN_LARGEST_SUBNORMAL: Final[float] = math.ldexp(_FLOAT8_E4M3FN_SIGNIFICAND_ALL_ONES, -9)
_FLOAT8_E4M3FN_LARGEST_POWER_OF_TWO: Final[float] = math.ldexp(1.0, 8)
_FLOAT8_E4M3FN_LARGEST_CONSECUTIVE_INTEGER: Final[float] = math.ldexp(1.0, 4)


FLOAT8_E4M3FN_SPEC: Final[FloatSpec] = FloatSpec(
    torch_dtype=torch.float8_e4m3fn,
    is_packed=False,
    register_bit_width=torch.finfo(torch.float8_e4m3fn).bits,
    sign=BitField(start=7, end=7),
    exponent=BitField(start=3, end=6),
    significand=BitField(start=0, end=2),
    exponent_bias=7,
    has_implicit_leading_bit=True,
    eps=torch.finfo(torch.float8_e4m3fn).eps,
    positive_zero=SpecialFloatValue(
        ref_float_value=0.0,
        sign_encoding=str(0),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(0, "03b"),
        description="positive zero",
    ),
    negative_zero=SpecialFloatValue(
        ref_float_value=-0.0,
        sign_encoding=str(1),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(0, "03b"),
        description="negative zero",
    ),
    positive_infinity=None,
    negative_infinity=None,
    positive_nan=SpecialFloatValue(
        ref_float_value=math.nan,
        sign_encoding=str(0),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(7, "03b"),
        description="positive quiet NaN",
    ),
    negative_nan=SpecialFloatValue(
        ref_float_value=-math.nan,
        sign_encoding=str(1),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(7, "03b"),
        description="negative quiet NaN",
    ),
    positive_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e4m3fn).max,
        sign_encoding=str(0),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(6, "03b"),
        description="largest positive normal",
    ),
    positive_smallest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e4m3fn).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "04b"),
        significand_encoding=format(0, "03b"),
        description="smallest positive normal",
    ),
    negative_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e4m3fn).min,
        sign_encoding=str(1),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(6, "03b"),
        description="most negative normal",
    ),
    negative_smallest_normal=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float8_e4m3fn).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "04b"),
        significand_encoding=format(0, "03b"),
        description="negative normal closest to zero",
    ),
    positive_largest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT8_E4M3FN_LARGEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(7, "03b"),
        description="largest positive subnormal",
    ),
    positive_smallest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT8_E4M3FN_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(1, "03b"),
        description="smallest positive subnormal",
    ),
    negative_largest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E4M3FN_LARGEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(7, "03b"),
        description="most negative subnormal",
    ),
    negative_smallest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E4M3FN_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(1, "03b"),
        description="negative subnormal closest to zero",
    ),
    positive_largest_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT8_E4M3FN_LARGEST_POWER_OF_TWO,
        sign_encoding=str(0),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(0, "03b"),
        description="largest positive power of two",
    ),
    positive_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e4m3fn).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "04b"),
        significand_encoding=format(0, "03b"),
        description="smallest positive normal power of two",
    ),
    positive_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT8_E4M3FN_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(1, "03b"),
        description="smallest positive subnormal power of two",
    ),
    negative_largest_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E4M3FN_LARGEST_POWER_OF_TWO,
        sign_encoding=str(1),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(0, "03b"),
        description="negative power of two with largest magnitude",
    ),
    negative_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float8_e4m3fn).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "04b"),
        significand_encoding=format(0, "03b"),
        description="negative normal power of two with smallest magnitude",
    ),
    negative_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E4M3FN_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(1, "03b"),
        description="negative subnormal power of two with smallest magnitude",
    ),
    positive_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e4m3fn).max,
        sign_encoding=str(0),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(6, "03b"),
        description="largest positive integer-valued float",
    ),
    negative_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e4m3fn).min,
        sign_encoding=str(1),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(6, "03b"),
        description="most negative integer-valued float",
    ),
    positive_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=_FLOAT8_E4M3FN_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(11, "04b"),
        significand_encoding=format(0, "03b"),
        description="largest positive consecutive integer",
    ),
    negative_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E4M3FN_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(11, "04b"),
        significand_encoding=format(0, "03b"),
        description="most negative consecutive integer",
    ),
    positive_number_example=SpecialFloatValue(
        ref_float_value=40.0,
        sign_encoding=str(0),
        exponent_encoding=format(12, "04b"),
        significand_encoding=format(2, "03b"),
        description="positive example",
    ),
    negative_number_example=SpecialFloatValue(
        ref_float_value=-32.0,
        sign_encoding=str(1),
        exponent_encoding=format(12, "04b"),
        significand_encoding=format(0, "03b"),
        description="negative example",
    ),
)

# FLOAT8_E4M3FNUZ: no infinity; only NaN 0x80; no negative zero; max finite 240.
# FNUZ variants (Graphcore/AMD/Qualcomm):
# https://github.com/openxla/stablehlo/blob/main/rfcs/20230321-fp8_fnuz.md
# ml_dtypes format description:
# https://github.com/jax-ml/ml_dtypes/blob/main/README.md#float8_e4m3fnuz
# FLOAT8_E4M3FNUZ encoding constants.
_FLOAT8_E4M3FNUZ_SIGNIFICAND_ALL_ONES: Final[int] = (1 << 3) - 1
_FLOAT8_E4M3FNUZ_SMALLEST_SUBNORMAL: Final[float] = math.ldexp(1.0, -10)
_FLOAT8_E4M3FNUZ_LARGEST_SUBNORMAL: Final[float] = math.ldexp(_FLOAT8_E4M3FNUZ_SIGNIFICAND_ALL_ONES, -10)
_FLOAT8_E4M3FNUZ_LARGEST_POWER_OF_TWO: Final[float] = math.ldexp(1.0, 7)
_FLOAT8_E4M3FNUZ_LARGEST_CONSECUTIVE_INTEGER: Final[float] = math.ldexp(1.0, 4)


FLOAT8_E4M3FNUZ_SPEC: Final[FloatSpec] = FloatSpec(
    torch_dtype=torch.float8_e4m3fnuz,
    is_packed=False,
    register_bit_width=torch.finfo(torch.float8_e4m3fnuz).bits,
    sign=BitField(start=7, end=7),
    exponent=BitField(start=3, end=6),
    significand=BitField(start=0, end=2),
    exponent_bias=8,
    has_implicit_leading_bit=True,
    eps=torch.finfo(torch.float8_e4m3fnuz).eps,
    positive_zero=SpecialFloatValue(
        ref_float_value=0.0,
        sign_encoding=str(0),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(0, "03b"),
        description="positive zero",
    ),
    negative_zero=None,
    positive_infinity=None,
    negative_infinity=None,
    positive_nan=None,
    negative_nan=SpecialFloatValue(
        ref_float_value=-math.nan,
        sign_encoding=str(1),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(0, "03b"),
        description="negative quiet NaN",
    ),
    positive_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e4m3fnuz).max,
        sign_encoding=str(0),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(7, "03b"),
        description="largest positive normal",
    ),
    positive_smallest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e4m3fnuz).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "04b"),
        significand_encoding=format(0, "03b"),
        description="smallest positive normal",
    ),
    negative_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e4m3fnuz).min,
        sign_encoding=str(1),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(7, "03b"),
        description="most negative normal",
    ),
    negative_smallest_normal=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float8_e4m3fnuz).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "04b"),
        significand_encoding=format(0, "03b"),
        description="negative normal closest to zero",
    ),
    positive_largest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT8_E4M3FNUZ_LARGEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(7, "03b"),
        description="largest positive subnormal",
    ),
    positive_smallest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT8_E4M3FNUZ_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(1, "03b"),
        description="smallest positive subnormal",
    ),
    negative_largest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E4M3FNUZ_LARGEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(7, "03b"),
        description="most negative subnormal",
    ),
    negative_smallest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E4M3FNUZ_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(1, "03b"),
        description="negative subnormal closest to zero",
    ),
    positive_largest_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT8_E4M3FNUZ_LARGEST_POWER_OF_TWO,
        sign_encoding=str(0),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(0, "03b"),
        description="largest positive power of two",
    ),
    positive_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e4m3fnuz).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "04b"),
        significand_encoding=format(0, "03b"),
        description="smallest positive normal power of two",
    ),
    positive_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT8_E4M3FNUZ_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(1, "03b"),
        description="smallest positive subnormal power of two",
    ),
    negative_largest_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E4M3FNUZ_LARGEST_POWER_OF_TWO,
        sign_encoding=str(1),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(0, "03b"),
        description="negative power of two with largest magnitude",
    ),
    negative_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float8_e4m3fnuz).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "04b"),
        significand_encoding=format(0, "03b"),
        description="negative normal power of two with smallest magnitude",
    ),
    negative_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E4M3FNUZ_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "04b"),
        significand_encoding=format(1, "03b"),
        description="negative subnormal power of two with smallest magnitude",
    ),
    positive_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e4m3fnuz).max,
        sign_encoding=str(0),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(7, "03b"),
        description="largest positive integer-valued float",
    ),
    negative_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e4m3fnuz).min,
        sign_encoding=str(1),
        exponent_encoding=format(15, "04b"),
        significand_encoding=format(7, "03b"),
        description="most negative integer-valued float",
    ),
    positive_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=_FLOAT8_E4M3FNUZ_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(12, "04b"),
        significand_encoding=format(0, "03b"),
        description="largest positive consecutive integer",
    ),
    negative_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E4M3FNUZ_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(12, "04b"),
        significand_encoding=format(0, "03b"),
        description="most negative consecutive integer",
    ),
    positive_number_example=SpecialFloatValue(
        ref_float_value=40.0,
        sign_encoding=str(0),
        exponent_encoding=format(13, "04b"),
        significand_encoding=format(2, "03b"),
        description="positive example",
    ),
    negative_number_example=SpecialFloatValue(
        ref_float_value=-32.0,
        sign_encoding=str(1),
        exponent_encoding=format(13, "04b"),
        significand_encoding=format(0, "03b"),
        description="negative example",
    ),
)

# FLOAT8_E5M2: IEEE-like infinity and NaNs; max finite 57344.
# OCP 8-bit Floating Point Specification (OFP8), Rev. 1.1:
# https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf
# FP8 formats paper (Micikevicius et al.):
# https://arxiv.org/pdf/2209.05433
# FLOAT8_E5M2 encoding constants.
_FLOAT8_E5M2_SIGNIFICAND_ALL_ONES: Final[int] = (1 << 2) - 1
_FLOAT8_E5M2_SMALLEST_SUBNORMAL: Final[float] = math.ldexp(1.0, -16)
_FLOAT8_E5M2_LARGEST_SUBNORMAL: Final[float] = math.ldexp(_FLOAT8_E5M2_SIGNIFICAND_ALL_ONES, -16)
_FLOAT8_E5M2_LARGEST_POWER_OF_TWO: Final[float] = math.ldexp(1.0, 15)
_FLOAT8_E5M2_LARGEST_CONSECUTIVE_INTEGER: Final[float] = math.ldexp(1.0, 3)


FLOAT8_E5M2_SPEC: Final[FloatSpec] = FloatSpec(
    torch_dtype=torch.float8_e5m2,
    is_packed=False,
    register_bit_width=torch.finfo(torch.float8_e5m2).bits,
    sign=BitField(start=7, end=7),
    exponent=BitField(start=2, end=6),
    significand=BitField(start=0, end=1),
    exponent_bias=15,
    has_implicit_leading_bit=True,
    eps=torch.finfo(torch.float8_e5m2).eps,
    positive_zero=SpecialFloatValue(
        ref_float_value=0.0,
        sign_encoding=str(0),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(0, "02b"),
        description="positive zero",
    ),
    negative_zero=SpecialFloatValue(
        ref_float_value=-0.0,
        sign_encoding=str(1),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(0, "02b"),
        description="negative zero",
    ),
    positive_infinity=SpecialFloatValue(
        ref_float_value=math.inf,
        sign_encoding=str(0),
        exponent_encoding=format(31, "05b"),
        significand_encoding=format(0, "02b"),
        description="positive infinity",
    ),
    negative_infinity=SpecialFloatValue(
        ref_float_value=-math.inf,
        sign_encoding=str(1),
        exponent_encoding=format(31, "05b"),
        significand_encoding=format(0, "02b"),
        description="negative infinity",
    ),
    positive_nan=SpecialFloatValue(
        ref_float_value=math.nan,
        sign_encoding=str(0),
        exponent_encoding=format(31, "05b"),
        significand_encoding=format(3, "02b"),
        description="positive quiet NaN",
    ),
    negative_nan=SpecialFloatValue(
        ref_float_value=-math.nan,
        sign_encoding=str(1),
        exponent_encoding=format(31, "05b"),
        significand_encoding=format(3, "02b"),
        description="negative quiet NaN",
    ),
    positive_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e5m2).max,
        sign_encoding=str(0),
        exponent_encoding=format(30, "05b"),
        significand_encoding=format(3, "02b"),
        description="largest positive normal",
    ),
    positive_smallest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e5m2).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "05b"),
        significand_encoding=format(0, "02b"),
        description="smallest positive normal",
    ),
    negative_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e5m2).min,
        sign_encoding=str(1),
        exponent_encoding=format(30, "05b"),
        significand_encoding=format(3, "02b"),
        description="most negative normal",
    ),
    negative_smallest_normal=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float8_e5m2).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "05b"),
        significand_encoding=format(0, "02b"),
        description="negative normal closest to zero",
    ),
    positive_largest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT8_E5M2_LARGEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(3, "02b"),
        description="largest positive subnormal",
    ),
    positive_smallest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT8_E5M2_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(1, "02b"),
        description="smallest positive subnormal",
    ),
    negative_largest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E5M2_LARGEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(3, "02b"),
        description="most negative subnormal",
    ),
    negative_smallest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E5M2_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(1, "02b"),
        description="negative subnormal closest to zero",
    ),
    positive_largest_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT8_E5M2_LARGEST_POWER_OF_TWO,
        sign_encoding=str(0),
        exponent_encoding=format(30, "05b"),
        significand_encoding=format(0, "02b"),
        description="largest positive power of two",
    ),
    positive_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e5m2).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "05b"),
        significand_encoding=format(0, "02b"),
        description="smallest positive normal power of two",
    ),
    positive_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT8_E5M2_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(1, "02b"),
        description="smallest positive subnormal power of two",
    ),
    negative_largest_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E5M2_LARGEST_POWER_OF_TWO,
        sign_encoding=str(1),
        exponent_encoding=format(30, "05b"),
        significand_encoding=format(0, "02b"),
        description="negative power of two with largest magnitude",
    ),
    negative_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float8_e5m2).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "05b"),
        significand_encoding=format(0, "02b"),
        description="negative normal power of two with smallest magnitude",
    ),
    negative_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E5M2_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(1, "02b"),
        description="negative subnormal power of two with smallest magnitude",
    ),
    positive_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e5m2).max,
        sign_encoding=str(0),
        exponent_encoding=format(30, "05b"),
        significand_encoding=format(3, "02b"),
        description="largest positive integer-valued float",
    ),
    negative_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e5m2).min,
        sign_encoding=str(1),
        exponent_encoding=format(30, "05b"),
        significand_encoding=format(3, "02b"),
        description="most negative integer-valued float",
    ),
    positive_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=_FLOAT8_E5M2_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(18, "05b"),
        significand_encoding=format(0, "02b"),
        description="largest positive consecutive integer",
    ),
    negative_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E5M2_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(18, "05b"),
        significand_encoding=format(0, "02b"),
        description="most negative consecutive integer",
    ),
    positive_number_example=SpecialFloatValue(
        ref_float_value=40.0,
        sign_encoding=str(0),
        exponent_encoding=format(20, "05b"),
        significand_encoding=format(1, "02b"),
        description="positive example",
    ),
    negative_number_example=SpecialFloatValue(
        ref_float_value=-32.0,
        sign_encoding=str(1),
        exponent_encoding=format(20, "05b"),
        significand_encoding=format(0, "02b"),
        description="negative example",
    ),
)

# FLOAT8_E5M2FNUZ: no infinity; only NaN 0x80; no negative zero; max finite 57344.
# torch.finfo(float8_e5m2fnuz).eps reports 0.125; the actual ULP at 1.0 is 0.25.
# This spec stores the correct 0.25.
# PyTorch bug: numeric_limits uses e5m2's bit patterns without the FNUZ bias adjustment.
# https://github.com/pytorch/pytorch/issues/198854
# Fix: https://github.com/pytorch/pytorch/pull/198855
# Fixed upstream in commit 74c343b (2026-09-29); torch 2.14.0 and 2.14.1 still report 0.125.
# FNUZ variants (Graphcore/AMD/Qualcomm):
# https://github.com/openxla/stablehlo/blob/main/rfcs/20230321-fp8_fnuz.md
# ml_dtypes format description:
# https://github.com/jax-ml/ml_dtypes/blob/main/README.md#float8_e5m2fnuz
# FLOAT8_E5M2FNUZ encoding constants.
_FLOAT8_E5M2FNUZ_SIGNIFICAND_ALL_ONES: Final[int] = (1 << 2) - 1
_FLOAT8_E5M2FNUZ_SMALLEST_SUBNORMAL: Final[float] = math.ldexp(1.0, -17)
_FLOAT8_E5M2FNUZ_LARGEST_SUBNORMAL: Final[float] = math.ldexp(_FLOAT8_E5M2FNUZ_SIGNIFICAND_ALL_ONES, -17)
_FLOAT8_E5M2FNUZ_LARGEST_POWER_OF_TWO: Final[float] = math.ldexp(1.0, 15)
_FLOAT8_E5M2FNUZ_LARGEST_CONSECUTIVE_INTEGER: Final[float] = math.ldexp(1.0, 3)


FLOAT8_E5M2FNUZ_SPEC: Final[FloatSpec] = FloatSpec(
    torch_dtype=torch.float8_e5m2fnuz,
    is_packed=False,
    register_bit_width=torch.finfo(torch.float8_e5m2fnuz).bits,
    sign=BitField(start=7, end=7),
    exponent=BitField(start=2, end=6),
    significand=BitField(start=0, end=1),
    exponent_bias=16,
    has_implicit_leading_bit=True,
    eps=0.25,
    positive_zero=SpecialFloatValue(
        ref_float_value=0.0,
        sign_encoding=str(0),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(0, "02b"),
        description="positive zero",
    ),
    negative_zero=None,
    positive_infinity=None,
    negative_infinity=None,
    positive_nan=None,
    negative_nan=SpecialFloatValue(
        ref_float_value=-math.nan,
        sign_encoding=str(1),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(0, "02b"),
        description="negative quiet NaN",
    ),
    positive_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e5m2fnuz).max,
        sign_encoding=str(0),
        exponent_encoding=format(31, "05b"),
        significand_encoding=format(3, "02b"),
        description="largest positive normal",
    ),
    positive_smallest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e5m2fnuz).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "05b"),
        significand_encoding=format(0, "02b"),
        description="smallest positive normal",
    ),
    negative_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e5m2fnuz).min,
        sign_encoding=str(1),
        exponent_encoding=format(31, "05b"),
        significand_encoding=format(3, "02b"),
        description="most negative normal",
    ),
    negative_smallest_normal=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float8_e5m2fnuz).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "05b"),
        significand_encoding=format(0, "02b"),
        description="negative normal closest to zero",
    ),
    positive_largest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT8_E5M2FNUZ_LARGEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(3, "02b"),
        description="largest positive subnormal",
    ),
    positive_smallest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT8_E5M2FNUZ_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(1, "02b"),
        description="smallest positive subnormal",
    ),
    negative_largest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E5M2FNUZ_LARGEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(3, "02b"),
        description="most negative subnormal",
    ),
    negative_smallest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E5M2FNUZ_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(1, "02b"),
        description="negative subnormal closest to zero",
    ),
    positive_largest_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT8_E5M2FNUZ_LARGEST_POWER_OF_TWO,
        sign_encoding=str(0),
        exponent_encoding=format(31, "05b"),
        significand_encoding=format(0, "02b"),
        description="largest positive power of two",
    ),
    positive_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e5m2fnuz).tiny,
        sign_encoding=str(0),
        exponent_encoding=format(1, "05b"),
        significand_encoding=format(0, "02b"),
        description="smallest positive normal power of two",
    ),
    positive_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT8_E5M2FNUZ_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(1, "02b"),
        description="smallest positive subnormal power of two",
    ),
    negative_largest_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E5M2FNUZ_LARGEST_POWER_OF_TWO,
        sign_encoding=str(1),
        exponent_encoding=format(31, "05b"),
        significand_encoding=format(0, "02b"),
        description="negative power of two with largest magnitude",
    ),
    negative_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=-torch.finfo(torch.float8_e5m2fnuz).tiny,
        sign_encoding=str(1),
        exponent_encoding=format(1, "05b"),
        significand_encoding=format(0, "02b"),
        description="negative normal power of two with smallest magnitude",
    ),
    negative_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E5M2FNUZ_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "05b"),
        significand_encoding=format(1, "02b"),
        description="negative subnormal power of two with smallest magnitude",
    ),
    positive_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e5m2fnuz).max,
        sign_encoding=str(0),
        exponent_encoding=format(31, "05b"),
        significand_encoding=format(3, "02b"),
        description="largest positive integer-valued float",
    ),
    negative_largest_integer=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e5m2fnuz).min,
        sign_encoding=str(1),
        exponent_encoding=format(31, "05b"),
        significand_encoding=format(3, "02b"),
        description="most negative integer-valued float",
    ),
    positive_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=_FLOAT8_E5M2FNUZ_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(19, "05b"),
        significand_encoding=format(0, "02b"),
        description="largest positive consecutive integer",
    ),
    negative_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT8_E5M2FNUZ_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(19, "05b"),
        significand_encoding=format(0, "02b"),
        description="most negative consecutive integer",
    ),
    positive_number_example=SpecialFloatValue(
        ref_float_value=40.0,
        sign_encoding=str(0),
        exponent_encoding=format(21, "05b"),
        significand_encoding=format(1, "02b"),
        description="positive example",
    ),
    negative_number_example=SpecialFloatValue(
        ref_float_value=-32.0,
        sign_encoding=str(1),
        exponent_encoding=format(21, "05b"),
        significand_encoding=format(0, "02b"),
        description="negative example",
    ),
)


# OCP Microscaling Formats (MX) Specification v1.0:
# https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
# E8M0: unsigned exponent-only scale, 2^(E-127); E=255 is NaN; no zero, infinity, or subnormals.
# FLOAT8_E8M0FNU encoding constants.
_FLOAT8_E8M0FNU_EXPONENT_ALL_ONES: Final[int] = (1 << 8) - 1
_FLOAT8_E8M0FNU_MAX_FINITE_EXPONENT: Final[int] = 254
_FLOAT8_E8M0FNU_LARGEST_POWER_OF_TWO: Final[float] = math.ldexp(1.0, 127)


FLOAT8_E8M0FNU_SPEC: Final[FloatSpec] = FloatSpec(
    torch_dtype=torch.float8_e8m0fnu,
    is_packed=False,
    register_bit_width=torch.finfo(torch.float8_e8m0fnu).bits,
    sign=None,
    exponent=BitField(start=0, end=7),
    significand=None,
    exponent_bias=127,
    has_implicit_leading_bit=True,
    eps=torch.finfo(torch.float8_e8m0fnu).eps,
    positive_zero=None,
    negative_zero=None,
    positive_infinity=None,
    negative_infinity=None,
    positive_nan=SpecialFloatValue(
        ref_float_value=math.nan,
        sign_encoding=None,
        exponent_encoding=format(_FLOAT8_E8M0FNU_EXPONENT_ALL_ONES, "08b"),
        significand_encoding=None,
        description="NaN",
    ),
    negative_nan=None,
    positive_largest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e8m0fnu).max,
        sign_encoding=None,
        exponent_encoding=format(_FLOAT8_E8M0FNU_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=None,
        description="largest positive normal",
    ),
    positive_smallest_normal=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e8m0fnu).tiny,
        sign_encoding=None,
        exponent_encoding=format(0, "08b"),
        significand_encoding=None,
        description="smallest positive normal",
    ),
    negative_largest_normal=None,
    negative_smallest_normal=None,
    positive_largest_subnormal=None,
    positive_smallest_subnormal=None,
    negative_largest_subnormal=None,
    negative_smallest_subnormal=None,
    positive_largest_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT8_E8M0FNU_LARGEST_POWER_OF_TWO,
        sign_encoding=None,
        exponent_encoding=format(_FLOAT8_E8M0FNU_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=None,
        description="largest positive power of two",
    ),
    positive_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=torch.finfo(torch.float8_e8m0fnu).tiny,
        sign_encoding=None,
        exponent_encoding=format(0, "08b"),
        significand_encoding=None,
        description="smallest positive normal power of two",
    ),
    positive_smallest_subnormal_power_of_two=None,
    negative_largest_power_of_two=None,
    negative_smallest_normal_power_of_two=None,
    negative_smallest_subnormal_power_of_two=None,
    positive_largest_integer=SpecialFloatValue(
        ref_float_value=_FLOAT8_E8M0FNU_LARGEST_POWER_OF_TWO,
        sign_encoding=None,
        exponent_encoding=format(_FLOAT8_E8M0FNU_MAX_FINITE_EXPONENT, "08b"),
        significand_encoding=None,
        description="largest positive integer-valued float",
    ),
    negative_largest_integer=None,
    positive_largest_consecutive_integer=None,
    negative_largest_consecutive_integer=None,
    positive_number_example=SpecialFloatValue(
        ref_float_value=0.125,
        sign_encoding=None,
        exponent_encoding=format(124, "08b"),
        significand_encoding=None,
        description="positive example",
    ),
    negative_number_example=None,
)


# FLOAT6_E2M3: 1/2/3, bias 1; max finite 7.5; stored one per uint8 (upper 2 bits unused).
# OCP Microscaling Formats (MX) Specification v1.0; FP6 has no Inf or NaN:
# https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
# ml_dtypes: https://github.com/jax-ml/ml_dtypes/blob/main/README.md#float6_e2m3fn
# ONNX: https://onnx.ai/onnx/technical/float6.html
# FLOAT6_E2M3 encoding constants.
_FLOAT6_E2M3_SIGNIFICAND_ALL_ONES: Final[int] = (1 << 3) - 1
_FLOAT6_E2M3_SMALLEST_SUBNORMAL: Final[float] = math.ldexp(1.0, -3)
_FLOAT6_E2M3_LARGEST_SUBNORMAL: Final[float] = math.ldexp(_FLOAT6_E2M3_SIGNIFICAND_ALL_ONES, -3)
_FLOAT6_E2M3_LARGEST_POWER_OF_TWO: Final[float] = math.ldexp(1.0, 2)
_FLOAT6_E2M3_LARGEST_INTEGER: Final[float] = 7.0
_FLOAT6_E2M3_LARGEST_CONSECUTIVE_INTEGER: Final[float] = 7.0


FLOAT6_E2M3_SPEC: Final[FloatSpec] = FloatSpec(
    torch_dtype=torch.uint8,
    is_packed=False,
    register_bit_width=torch.iinfo(torch.uint8).bits,
    sign=BitField(start=5, end=5),
    exponent=BitField(start=3, end=4),
    significand=BitField(start=0, end=2),
    exponent_bias=1,
    has_implicit_leading_bit=True,
    eps=0.125,
    positive_zero=SpecialFloatValue(
        ref_float_value=0.0,
        sign_encoding=str(0),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(0, "03b"),
        description="positive zero",
    ),
    negative_zero=SpecialFloatValue(
        ref_float_value=-0.0,
        sign_encoding=str(1),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(0, "03b"),
        description="negative zero",
    ),
    positive_infinity=None,
    negative_infinity=None,
    positive_nan=None,
    negative_nan=None,
    positive_largest_normal=SpecialFloatValue(
        ref_float_value=7.5,
        sign_encoding=str(0),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(7, "03b"),
        description="largest positive normal",
    ),
    positive_smallest_normal=SpecialFloatValue(
        ref_float_value=1.0,
        sign_encoding=str(0),
        exponent_encoding=format(1, "02b"),
        significand_encoding=format(0, "03b"),
        description="smallest positive normal",
    ),
    negative_largest_normal=SpecialFloatValue(
        ref_float_value=-7.5,
        sign_encoding=str(1),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(7, "03b"),
        description="most negative normal",
    ),
    negative_smallest_normal=SpecialFloatValue(
        ref_float_value=-1.0,
        sign_encoding=str(1),
        exponent_encoding=format(1, "02b"),
        significand_encoding=format(0, "03b"),
        description="negative normal closest to zero",
    ),
    positive_largest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT6_E2M3_LARGEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(7, "03b"),
        description="largest positive subnormal",
    ),
    positive_smallest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT6_E2M3_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(1, "03b"),
        description="smallest positive subnormal",
    ),
    negative_largest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT6_E2M3_LARGEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(7, "03b"),
        description="most negative subnormal",
    ),
    negative_smallest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT6_E2M3_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(1, "03b"),
        description="negative subnormal closest to zero",
    ),
    positive_largest_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT6_E2M3_LARGEST_POWER_OF_TWO,
        sign_encoding=str(0),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(0, "03b"),
        description="largest positive power of two",
    ),
    positive_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=1.0,
        sign_encoding=str(0),
        exponent_encoding=format(1, "02b"),
        significand_encoding=format(0, "03b"),
        description="smallest positive normal power of two",
    ),
    positive_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT6_E2M3_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(1, "03b"),
        description="smallest positive subnormal power of two",
    ),
    negative_largest_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT6_E2M3_LARGEST_POWER_OF_TWO,
        sign_encoding=str(1),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(0, "03b"),
        description="negative power of two with largest magnitude",
    ),
    negative_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=-1.0,
        sign_encoding=str(1),
        exponent_encoding=format(1, "02b"),
        significand_encoding=format(0, "03b"),
        description="negative normal power of two with smallest magnitude",
    ),
    negative_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT6_E2M3_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(1, "03b"),
        description="negative subnormal power of two with smallest magnitude",
    ),
    positive_largest_integer=SpecialFloatValue(
        ref_float_value=_FLOAT6_E2M3_LARGEST_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(6, "03b"),
        description="largest positive integer-valued float",
    ),
    negative_largest_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT6_E2M3_LARGEST_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(6, "03b"),
        description="most negative integer-valued float",
    ),
    positive_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=_FLOAT6_E2M3_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(6, "03b"),
        description="largest positive consecutive integer",
    ),
    negative_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT6_E2M3_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(6, "03b"),
        description="most negative consecutive integer",
    ),
    positive_number_example=SpecialFloatValue(
        ref_float_value=3.5,
        sign_encoding=str(0),
        exponent_encoding=format(2, "02b"),
        significand_encoding=format(6, "03b"),
        description="positive example",
    ),
    negative_number_example=SpecialFloatValue(
        ref_float_value=-2.0,
        sign_encoding=str(1),
        exponent_encoding=format(2, "02b"),
        significand_encoding=format(0, "03b"),
        description="negative example",
    ),
)

# FLOAT6_E3M2: 1/3/2, bias 3; max finite 28; stored one per uint8 (upper 2 bits unused).
# OCP Microscaling Formats (MX) Specification v1.0; FP6 has no Inf or NaN:
# https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
# ml_dtypes: https://github.com/jax-ml/ml_dtypes/blob/main/README.md#float6_e3m2fn
# ONNX: https://onnx.ai/onnx/technical/float6.html
# FLOAT6_E3M2 encoding constants.
_FLOAT6_E3M2_SIGNIFICAND_ALL_ONES: Final[int] = (1 << 2) - 1
_FLOAT6_E3M2_SMALLEST_SUBNORMAL: Final[float] = math.ldexp(1.0, -4)
_FLOAT6_E3M2_LARGEST_SUBNORMAL: Final[float] = math.ldexp(_FLOAT6_E3M2_SIGNIFICAND_ALL_ONES, -4)
_FLOAT6_E3M2_LARGEST_POWER_OF_TWO: Final[float] = math.ldexp(1.0, 4)
_FLOAT6_E3M2_LARGEST_INTEGER: Final[float] = 28.0
_FLOAT6_E3M2_LARGEST_CONSECUTIVE_INTEGER: Final[float] = 8.0


FLOAT6_E3M2_SPEC: Final[FloatSpec] = FloatSpec(
    torch_dtype=torch.uint8,
    is_packed=False,
    register_bit_width=torch.iinfo(torch.uint8).bits,
    sign=BitField(start=5, end=5),
    exponent=BitField(start=2, end=4),
    significand=BitField(start=0, end=1),
    exponent_bias=3,
    has_implicit_leading_bit=True,
    eps=0.25,
    positive_zero=SpecialFloatValue(
        ref_float_value=0.0,
        sign_encoding=str(0),
        exponent_encoding=format(0, "03b"),
        significand_encoding=format(0, "02b"),
        description="positive zero",
    ),
    negative_zero=SpecialFloatValue(
        ref_float_value=-0.0,
        sign_encoding=str(1),
        exponent_encoding=format(0, "03b"),
        significand_encoding=format(0, "02b"),
        description="negative zero",
    ),
    positive_infinity=None,
    negative_infinity=None,
    positive_nan=None,
    negative_nan=None,
    positive_largest_normal=SpecialFloatValue(
        ref_float_value=28.0,
        sign_encoding=str(0),
        exponent_encoding=format(7, "03b"),
        significand_encoding=format(3, "02b"),
        description="largest positive normal",
    ),
    positive_smallest_normal=SpecialFloatValue(
        ref_float_value=0.25,
        sign_encoding=str(0),
        exponent_encoding=format(1, "03b"),
        significand_encoding=format(0, "02b"),
        description="smallest positive normal",
    ),
    negative_largest_normal=SpecialFloatValue(
        ref_float_value=-28.0,
        sign_encoding=str(1),
        exponent_encoding=format(7, "03b"),
        significand_encoding=format(3, "02b"),
        description="most negative normal",
    ),
    negative_smallest_normal=SpecialFloatValue(
        ref_float_value=-0.25,
        sign_encoding=str(1),
        exponent_encoding=format(1, "03b"),
        significand_encoding=format(0, "02b"),
        description="negative normal closest to zero",
    ),
    positive_largest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT6_E3M2_LARGEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "03b"),
        significand_encoding=format(3, "02b"),
        description="largest positive subnormal",
    ),
    positive_smallest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT6_E3M2_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "03b"),
        significand_encoding=format(1, "02b"),
        description="smallest positive subnormal",
    ),
    negative_largest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT6_E3M2_LARGEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "03b"),
        significand_encoding=format(3, "02b"),
        description="most negative subnormal",
    ),
    negative_smallest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT6_E3M2_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "03b"),
        significand_encoding=format(1, "02b"),
        description="negative subnormal closest to zero",
    ),
    positive_largest_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT6_E3M2_LARGEST_POWER_OF_TWO,
        sign_encoding=str(0),
        exponent_encoding=format(7, "03b"),
        significand_encoding=format(0, "02b"),
        description="largest positive power of two",
    ),
    positive_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=0.25,
        sign_encoding=str(0),
        exponent_encoding=format(1, "03b"),
        significand_encoding=format(0, "02b"),
        description="smallest positive normal power of two",
    ),
    positive_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT6_E3M2_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "03b"),
        significand_encoding=format(1, "02b"),
        description="smallest positive subnormal power of two",
    ),
    negative_largest_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT6_E3M2_LARGEST_POWER_OF_TWO,
        sign_encoding=str(1),
        exponent_encoding=format(7, "03b"),
        significand_encoding=format(0, "02b"),
        description="negative power of two with largest magnitude",
    ),
    negative_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=-0.25,
        sign_encoding=str(1),
        exponent_encoding=format(1, "03b"),
        significand_encoding=format(0, "02b"),
        description="negative normal power of two with smallest magnitude",
    ),
    negative_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT6_E3M2_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "03b"),
        significand_encoding=format(1, "02b"),
        description="negative subnormal power of two with smallest magnitude",
    ),
    positive_largest_integer=SpecialFloatValue(
        ref_float_value=_FLOAT6_E3M2_LARGEST_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(7, "03b"),
        significand_encoding=format(3, "02b"),
        description="largest positive integer-valued float",
    ),
    negative_largest_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT6_E3M2_LARGEST_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(7, "03b"),
        significand_encoding=format(3, "02b"),
        description="most negative integer-valued float",
    ),
    positive_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=_FLOAT6_E3M2_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(6, "03b"),
        significand_encoding=format(0, "02b"),
        description="largest positive consecutive integer",
    ),
    negative_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT6_E3M2_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(6, "03b"),
        significand_encoding=format(0, "02b"),
        description="most negative consecutive integer",
    ),
    positive_number_example=SpecialFloatValue(
        ref_float_value=3.5,
        sign_encoding=str(0),
        exponent_encoding=format(4, "03b"),
        significand_encoding=format(3, "02b"),
        description="positive example",
    ),
    negative_number_example=SpecialFloatValue(
        ref_float_value=-2.0,
        sign_encoding=str(1),
        exponent_encoding=format(4, "03b"),
        significand_encoding=format(0, "02b"),
        description="negative example",
    ),
)


# FLOAT4_E2M1: 1/2/1, bias 1; max finite 6; no Inf/NaN.
# torch.float4_e2m1fn_x2 packs two 4-bit elements per byte; register_bit_width is the 8-bit packed register.
# OCP Microscaling Formats (MX) Specification v1.0:
# https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
# ml_dtypes:
# https://github.com/jax-ml/ml_dtypes/blob/main/README.md#float4_e2m1fn
# ONNX:
# https://onnx.ai/onnx/technical/float4.html
# FLOAT4_E2M1FN encoding constants.
_FLOAT4_E2M1FN_SIGNIFICAND_ALL_ONES: Final[int] = (1 << 1) - 1
_FLOAT4_E2M1FN_SMALLEST_SUBNORMAL: Final[float] = math.ldexp(1.0, -1)
_FLOAT4_E2M1FN_LARGEST_SUBNORMAL: Final[float] = math.ldexp(_FLOAT4_E2M1FN_SIGNIFICAND_ALL_ONES, -1)
_FLOAT4_E2M1FN_LARGEST_POWER_OF_TWO: Final[float] = math.ldexp(1.0, 2)
_FLOAT4_E2M1FN_LARGEST_INTEGER: Final[float] = 6.0
_FLOAT4_E2M1FN_LARGEST_CONSECUTIVE_INTEGER: Final[float] = 4.0


FLOAT4_E2M1FN_x2_SPEC: Final[FloatSpec] = FloatSpec(
    torch_dtype=torch.float4_e2m1fn_x2,
    is_packed=True,
    register_bit_width=torch.finfo(torch.float4_e2m1fn_x2).bits,
    sign=BitField(start=3, end=3),
    exponent=BitField(start=1, end=2),
    significand=BitField(start=0, end=0),
    exponent_bias=1,
    has_implicit_leading_bit=True,
    eps=0.5,
    positive_zero=SpecialFloatValue(
        ref_float_value=0.0,
        sign_encoding=str(0),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(0, "01b"),
        description="positive zero",
    ),
    negative_zero=SpecialFloatValue(
        ref_float_value=-0.0,
        sign_encoding=str(1),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(0, "01b"),
        description="negative zero",
    ),
    positive_infinity=None,
    negative_infinity=None,
    positive_nan=None,
    negative_nan=None,
    positive_largest_normal=SpecialFloatValue(
        ref_float_value=6.0,
        sign_encoding=str(0),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(1, "01b"),
        description="largest positive normal",
    ),
    positive_smallest_normal=SpecialFloatValue(
        ref_float_value=1.0,
        sign_encoding=str(0),
        exponent_encoding=format(1, "02b"),
        significand_encoding=format(0, "01b"),
        description="smallest positive normal",
    ),
    negative_largest_normal=SpecialFloatValue(
        ref_float_value=-6.0,
        sign_encoding=str(1),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(1, "01b"),
        description="most negative normal",
    ),
    negative_smallest_normal=SpecialFloatValue(
        ref_float_value=-1.0,
        sign_encoding=str(1),
        exponent_encoding=format(1, "02b"),
        significand_encoding=format(0, "01b"),
        description="negative normal closest to zero",
    ),
    positive_largest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT4_E2M1FN_LARGEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(1, "01b"),
        description="largest positive subnormal",
    ),
    positive_smallest_subnormal=SpecialFloatValue(
        ref_float_value=_FLOAT4_E2M1FN_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(1, "01b"),
        description="smallest positive subnormal",
    ),
    negative_largest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT4_E2M1FN_LARGEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(1, "01b"),
        description="most negative subnormal",
    ),
    negative_smallest_subnormal=SpecialFloatValue(
        ref_float_value=-_FLOAT4_E2M1FN_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(1, "01b"),
        description="negative subnormal closest to zero",
    ),
    positive_largest_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT4_E2M1FN_LARGEST_POWER_OF_TWO,
        sign_encoding=str(0),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(0, "01b"),
        description="largest positive power of two",
    ),
    positive_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=1.0,
        sign_encoding=str(0),
        exponent_encoding=format(1, "02b"),
        significand_encoding=format(0, "01b"),
        description="smallest positive normal power of two",
    ),
    positive_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=_FLOAT4_E2M1FN_SMALLEST_SUBNORMAL,
        sign_encoding=str(0),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(1, "01b"),
        description="smallest positive subnormal power of two",
    ),
    negative_largest_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT4_E2M1FN_LARGEST_POWER_OF_TWO,
        sign_encoding=str(1),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(0, "01b"),
        description="negative power of two with largest magnitude",
    ),
    negative_smallest_normal_power_of_two=SpecialFloatValue(
        ref_float_value=-1.0,
        sign_encoding=str(1),
        exponent_encoding=format(1, "02b"),
        significand_encoding=format(0, "01b"),
        description="negative normal power of two with smallest magnitude",
    ),
    negative_smallest_subnormal_power_of_two=SpecialFloatValue(
        ref_float_value=-_FLOAT4_E2M1FN_SMALLEST_SUBNORMAL,
        sign_encoding=str(1),
        exponent_encoding=format(0, "02b"),
        significand_encoding=format(1, "01b"),
        description="negative subnormal power of two with smallest magnitude",
    ),
    positive_largest_integer=SpecialFloatValue(
        ref_float_value=_FLOAT4_E2M1FN_LARGEST_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(1, "01b"),
        description="largest positive integer-valued float",
    ),
    negative_largest_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT4_E2M1FN_LARGEST_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(1, "01b"),
        description="most negative integer-valued float",
    ),
    positive_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=_FLOAT4_E2M1FN_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(0),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(0, "01b"),
        description="largest positive consecutive integer",
    ),
    negative_largest_consecutive_integer=SpecialFloatValue(
        ref_float_value=-_FLOAT4_E2M1FN_LARGEST_CONSECUTIVE_INTEGER,
        sign_encoding=str(1),
        exponent_encoding=format(3, "02b"),
        significand_encoding=format(0, "01b"),
        description="most negative consecutive integer",
    ),
    positive_number_example=SpecialFloatValue(
        ref_float_value=3.0,
        sign_encoding=str(0),
        exponent_encoding=format(2, "02b"),
        significand_encoding=format(1, "01b"),
        description="positive example",
    ),
    negative_number_example=SpecialFloatValue(
        ref_float_value=-2.0,
        sign_encoding=str(1),
        exponent_encoding=format(2, "02b"),
        significand_encoding=format(0, "01b"),
        description="negative example",
    ),
)


# Both torch dtypes and their string names are accepted; FP6 only has string names.
FLOAT_SPEC_TABLE: Final[dict[torch.dtype | str, FloatSpec]] = {
    torch.float64: FLOAT64_SPEC,
    "float64": FLOAT64_SPEC,
    torch.float32: FLOAT32_SPEC,
    "float32": FLOAT32_SPEC,
    torch.float16: FLOAT16_SPEC,
    "float16": FLOAT16_SPEC,
    torch.bfloat16: BFLOAT16_SPEC,
    "bfloat16": BFLOAT16_SPEC,
    torch.float8_e4m3fn: FLOAT8_E4M3FN_SPEC,
    "float8_e4m3fn": FLOAT8_E4M3FN_SPEC,
    torch.float8_e4m3fnuz: FLOAT8_E4M3FNUZ_SPEC,
    "float8_e4m3fnuz": FLOAT8_E4M3FNUZ_SPEC,
    torch.float8_e5m2: FLOAT8_E5M2_SPEC,
    "float8_e5m2": FLOAT8_E5M2_SPEC,
    torch.float8_e5m2fnuz: FLOAT8_E5M2FNUZ_SPEC,
    "float8_e5m2fnuz": FLOAT8_E5M2FNUZ_SPEC,
    torch.float8_e8m0fnu: FLOAT8_E8M0FNU_SPEC,
    "float8_e8m0fnu": FLOAT8_E8M0FNU_SPEC,
    torch.float4_e2m1fn_x2: FLOAT4_E2M1FN_x2_SPEC,
    "float4_e2m1fn_x2": FLOAT4_E2M1FN_x2_SPEC,
    "float6_e2m3fn": FLOAT6_E2M3_SPEC,
    "float6_e3m2fn": FLOAT6_E3M2_SPEC,
}

__all__ = ["FLOAT_SPEC_TABLE"]
