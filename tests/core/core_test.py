# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.core`."""

import dataclasses

import pytest

from mokit.core import MokitConfigBase

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
