# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`mokit.serialization`."""

import pickle
from pathlib import Path

import pytest
import torch

from mokit.serialization import torch_add_safe_global

# ==============================================================================
# torch_add_safe_global
# ==============================================================================

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


@torch_add_safe_global
class Registered:
    """A class registered for weights-only loading."""

    def __init__(self, value: int) -> None:
        """Store ``value``."""
        self.value = value


class Unregistered:
    """A class that is not registered for weights-only loading."""

    def __init__(self, value: int) -> None:
        """Store ``value``."""
        self.value = value


def _round_trip(obj: object, path: Path) -> object:
    """Save ``obj`` and load it back with the weights-only loader."""
    torch.save(obj, path)
    return torch.load(path, weights_only=True)


# ------------------------------------------------------------------------------
# Decorator
# ------------------------------------------------------------------------------


def test_decorator_returns_the_class() -> None:
    """Test that the decorator returns the decorated class unchanged."""

    @torch_add_safe_global
    class Decorated:
        """A decorated class."""

    assert Decorated.__name__ == "Decorated"


# ------------------------------------------------------------------------------
# Round trip
# ------------------------------------------------------------------------------


def test_registered_instance_round_trips(tmp_path: Path) -> None:
    """Test that a registered instance survives a weights-only round trip."""
    loaded = _round_trip(Registered(7), tmp_path / "registered.pt")
    assert isinstance(loaded, Registered)
    assert loaded.value == 7


def test_unregistered_class_is_rejected(tmp_path: Path) -> None:
    """Test that an unregistered class is rejected by the weights-only loader."""
    with pytest.raises(pickle.UnpicklingError, match="Unsupported global"):
        _round_trip(Unregistered(7), tmp_path / "unregistered.pt")
