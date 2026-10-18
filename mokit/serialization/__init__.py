# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""Serialization utilities."""

import torch


def torch_add_safe_global[T: type](cls: T) -> T:
    """
    Allow ``cls`` in PyTorch's weights-only unpickling, and return it.

    Checkpoints reference the classes they contain, so each one must be allowlisted
    for deserialization when restoring.
    """
    torch.serialization.add_safe_globals([cls])
    return cls
