# SPDX-FileCopyrightText: 2026 CaoSY
# SPDX-License-Identifier: BSD-3-Clause

"""MOKIT package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mokit")
except PackageNotFoundError:
    # This file being executed already implies MOKIT package is accessible.
    # The use should not be blocked because MOKIT is not found in the search
    # path. Simply say the version is not known.
    __version__ = "unknown"

from mokit import quantization

__all__ = ["quantization"]
